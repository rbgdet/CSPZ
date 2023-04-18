import sklearn as skl
import sklearn_som
from sklearn_som.som import SOM
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time

def flux2mag(flux):
    return -2.5 * np.log10(flux) + 30

def luptize(flux, var, s, zp):
    # s: measurement error (variance) of the flux (with zero pt zp) of an object at the limiting magnitude of the survey
    # a: Pogson's ratio
    # b: softening parameter that sets the scale of transition between linear and log behavior of the luptitudes
    #print(s.shape)
    a = 2.5 * np.log10(np.exp(1)) 
    b = a**(1./2) * s 
    mu0 = zp -2.5 * np.log10(b)

    # turn into luptitudes and their errors
    #print(mu0.shape, flux.shape, b.shape)
    lupt = mu0 - a * np.arcsinh(flux / (2 * b))
    lupt_var = a ** 2 * var / ((2 * b) ** 2 + flux ** 2)
    return lupt, lupt_var

def luptize_wide(flux, var=0, zp=22.5):
    """ The flux must be four dimensional and must be given in the order [f_i, f_g, f_r, f_z] to match the ordering of the softening parameter b """
    #lim_mags = np.array([22.9, 23.7, 23.5, 22.2]) # old
    # see ../test/full_run_on_data/limiting_mags_in_data.ipynb
    lim_mags = np.array([22.92, 23.7, 23.49, 22.28]) # g band is copied from commented array above because g band is not in up to date mastercat
    s = (10**((zp - lim_mags) / 2.5)) / 10  # des limiting mag is 10 sigma

    return luptize(flux, var, s, zp)

def luptize_deep(flux, var=0, zp=22.5):
    """The flux must be 8 dimensional and must be given in the order [f_i, f_g, f_r, f_z, f_u, f_Y, f_J, f_H, f_K] to match the ordering of the softening parameter b """
    #lim_mags_des = np.array([22.9, 23.7, 23.5, 22.2, 25]) # old
    #lim_mags_vista = np.array([24.6, 24.5, 24.0, 23.5]) # old
    lim_mags_des = np.array([24.66, 25.57, 25.27, 24.06, 24.64])
    lim_mags_vista = np.array([24.02, 23.69, 23.58]) # y band value is copied from array above because Y band is not in the up to date catalog
    s_des = (10**((zp-lim_mags_des)/2.5)) / 10  # des limiting mag is 10 sigma
    s_vista = (10**((zp-lim_mags_vista)/2.5)) / 10  # vista limiting mag is 10 sigma

    s = np.concatenate([s_des, s_vista])

    return luptize(flux, var, s, zp)

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

# mask stars based on (z−K) > 0.5×(r −z) color cut

df = df[(z-k) > 0.5*(r-z)]

deep=df

deep_bands = ['I', 'G', 'R', 'Z', 'U', 'J', 'H', 'KS']
deep_fluxes = deep[(f'BDF_FLUX_DERED_CALIB_{band}' for band in deep_bands)].values
deep_vars = deep[(f'BDF_FLUX_ERR_DERED_CALIB_{band}' for band in deep_bands)].values

deep_lupts = np.array([luptize_deep(deep_flux, var=deep_var) for (deep_flux, deep_var) in zip(deep_fluxes, deep_vars)])
deep_luptitudes, deep_lupts_var = deep_lupts[:,0], deep_lupts[:,1]

deep_input = [0]*8
deep_lupt_i = deep_luptitudes.T[0]
deep_input[0] = deep_lupt_i
j=1
for deep_lupt in deep_luptitudes.T[1:]:
    deep_input[j]=deep_lupt-deep_lupt_i
    j+=1
    
deep_input = np.array(deep_input)

#normalizing i-band luptitude
deep_median_i = np.median(deep_lupt_i)
deep_std_i = np.std(deep_lupt_i)

deep_nlupt_i = (deep_lupt_i-deep_median_i)/deep_std_i

deep_medians = np.median(deep_input, axis=1)
deep_stds = np.std(deep_input, axis=1)
deep_n_input = (deep_input.T-deep_medians)/deep_stds
#comment out line below to use unit variance
deep_n_input = (deep_n_input-deep_n_input.min())/(deep_n_input.max()-deep_n_input.min())

n = 48
som_deep = SOM(m=n, n=n, dim=8)

np.random.seed(42)
som_deep.fit(deep_n_input, epochs=7)
pred = som_deep.predict(deep_n_input)

deep_weights = som_deep.weights
print(deep_weights.shape)

np.savetxt('/project2/chihway/raulteixeira/data/deepfields_weights_7E_01.txt', deep_weights)
specz = pd.read_csv('/project2/chihway/raulteixeira/data/deepfields_specz.csv.gz')

specz['cell_wide']=pred

#UV stands for unit variance
specz.to_csv('/project2/chihway/raulteixeira/data/deepfields_specz_SOMpred_7E_01.csv.gz')