import sys
sys.path.append("/home/raulteixeira/repos/CSPZ/scripts/")

import NoiseSOM as ns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
start_time = time.time()

def flux2mag(flux):
    return -2.5 * np.log10(flux) + 30


df = pd.read_table('/project2/chihway/raulteixeira/data/deepfields.csv.gz', sep=',', header=0)


# Mask flagged regions
mask = df.MASK_FLAGS_NIR==0
mask &= df.MASK_FLAGS==0
mask &= df.FLAGS_NIR==0
mask &= df.FLAGS==0
mask &= df.FLAGSTR=="b'ok'"
mask &= df.FLAGSTR_NIR=="b'ok'"
df = df[mask]
df = df.drop(columns=[
    "MASK_FLAGS",
    "MASK_FLAGS_NIR",
    "FLAGS",
    "FLAGS_NIR",
    "FLAGSTR",
    "FLAGSTR_NIR",
])

deep_bands_ = ["U","G","R","I","Z","J","H","KS"]
# remove crazy colors, defined as two 
# consecutive colors (e.g u-g, g-r, r-i, etc) 
# that have a value smaler than -1
mags_d = np.zeros((len(df),len(deep_bands_)))
magerrs_d = np.zeros((len(df),len(deep_bands_)))

for i,band in enumerate(deep_bands_):
    print(i,band)
    mags_d[:,i] = flux2mag(df['BDF_FLUX_DERED_CALIB_%s'%band])

colors = np.zeros((len(df),len(deep_bands_)-1))
for i in range(len(deep_bands_)-1):
    colors[:,i] = mags_d[:,i] - mags_d[:,i+1]

normal_colors = np.mean(colors > -1, axis=1) == 1
normal_colors.sum()

df = df[normal_colors]

# mask faint objects, i < 25
df = df[flux2mag(df.BDF_FLUX_DERED_CALIB_I.values) < 25]

i = flux2mag(df.BDF_FLUX_DERED_CALIB_I.values)
r = flux2mag(df.BDF_FLUX_DERED_CALIB_R.values)
z = flux2mag(df.BDF_FLUX_DERED_CALIB_Z.values)
k = flux2mag(df.BDF_FLUX_DERED_CALIB_KS.values)


plt.hexbin((r-z), (z-k), gridsize=100, mincnt=1, bins='log')
_t = np.linspace(-1, 5, 100)
plt.plot(_t, 0.5*_t, color='r', ls='--')
plt.xlabel(r"$r-z$")
plt.ylabel(r"$z-K$")
plt.show()

# mask stars based on (z−K) > 0.5×(r −z) color cut

df = df[(z-k) > 0.5*(r-z)]

deep_data=df

deep_data.loc[:,"BDF_MAG_DERED_CALIB_I"]=flux2mag(deep_data.loc[:, 'BDF_FLUX_DERED_CALIB_I'])

def balrog_sigmoid(x, x0):
    """Sigmoid function
    Parameters
    ----------
    x : float or array-like
        Points at which to evaluate the function.
    x0 : float or array-like
        Location of transition.
    Returns
    -------
    sigmoid : scalar or array-like, same shape as input
    """
    return 1.0 - 1.0 / (1.0 + np.exp(-4.0 * (x - x0)))

def mock_balrog_sigmoid(
    deep_data, 
    sigmoid_x0,
    N,
    ref_mag_col = "BDF_MAG_DERED_CALIB_I"
):
    """
    Function for selecting deep field galaxies at a rate that follows a sigmoid function that smoothly transitions from 1 for bright objects, to a value of 0 for faint objects. 
    Parameters
    ----------
    deep_data : pandas dataframe
        Pandas dataframe containing the deep field data.
    sigmoid_x0 : float
        Magnitude value at which the sigmoid function transitions from 1 to 0.
    N : int
        Number of galaxies to be drawn.
    ref_mag_col : string
        Column name of the reference magnitude in deep_data
    Returns
    -------
    deep_balrog_selection : pandas dataframe
        Pandas dataframe containing a list of N deep field objects to be injected by Balrog.
    """
    np.random.seed()
    mag_ref = deep_data.loc[:, ref_mag_col].values
    weights = balrog_sigmoid(mag_ref, sigmoid_x0)
    weights/=sum(weights)
    selected_objects = np.random.choice(len(deep_data), N, p=weights, replace=True)
    
    deep_balrog_selection = deep_data.iloc[selected_objects]
    return deep_balrog_selection
                          
    
# New Pandas Dataframe with only detected galaxies
# A value of 23.0 returns something similar to Balrog Y3
# A value of 23.5 maybe is similar to the WL sample in Y6.
# A value of 21.5 is maybe optimal for LSS samples in Y6.
#deep_balrog_mocked = mock_balrog_sigmoid(deep_data, 23.0, nTrain)

nTrain = int(1e2)
deep_balrog_mocked = mock_balrog_sigmoid(deep_data, 23.0, nTrain)

bands = ['U','G','R','I','Z','J','H','KS']
fluxes_d = np.zeros((len(deep_balrog_mocked),len(bands)))
fluxerrs_d = np.zeros((len(deep_balrog_mocked),len(bands)))

for i,band in enumerate(bands):
    print(i,band)
    fluxes_d[:,i] = deep_balrog_mocked['BDF_FLUX_DERED_CALIB_%s'%band]
    fluxerrs_d[:,i] = deep_balrog_mocked['BDF_FLUX_ERR_DERED_CALIB_%s'%band]

# Train the SOM with this set (takes a few hours on laptop!)
#len(fluxes_d)

# Scramble the order of the catalog for purposes of training
#indices = np.random.choice(nTrain, size=nTrain, replace=False)
indices = np.random.choice(nTrain, size=nTrain, replace=True)
hh = ns.hFunc(nTrain, sigma=(30,1))
metric = ns.AsinhMetric(lnScaleSigma=0.4, lnScaleStep=0.03)

som = ns.NoiseSOM(metric, fluxes_d[indices,:], fluxerrs_d[indices,:], \
    learning=hh, \
    shape=(48,48), \
    wrap=False,logF=True, \
    initialize='sample', \
    minError=0.02)

path_cats='/project2/chihway/raulteixeira/data/'
# And save the resultant weight matrix
np.save("%s/som_deep_48_48_1e2.npy"%path_cats,som.weights)
print("--- %s seconds ---" % (time.time() - start_time))