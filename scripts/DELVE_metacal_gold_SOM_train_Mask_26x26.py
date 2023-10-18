import sys
sys.path.append("/home/raulteixeira/repos/CSPZ/scripts/")
import NoiseSOM as ns
import numpy as np
import pandas as pd
import h5py
import matplotlib.pyplot as plt
import scipy
import sys
import time
start_time = time.time()

#with h5py.File('/project2/chihway/data/decade/metacal_gold_combined_20230531.hdf') as f: #last one used
with h5py.File('/project2/chihway/data/decade/metacal_gold_combined_20230613.hdf') as f: #unused
    
    flux_r, flux_i, flux_z = np.array(f['mcal_flux_noshear']).T
    flux_err_r, flux_err_i, flux_err_z = np.array(f['mcal_flux_err_noshear']).T

    Mask = np.load('/project2/chihway/raulteixeira/data/metacal_gold_mask.npy')
    
    #These are the fluxes with all metacal cuts applied
    flux_r = flux_r[Mask]
    flux_i = flux_i[Mask]
    flux_z = flux_z[Mask]
    flux_err_r = flux_err_r[Mask]
    flux_err_i = flux_err_i[Mask]
    flux_err_z = flux_err_z[Mask]

fluxes_d = np.array([flux_r, flux_i, flux_z]).T
fluxerrs_d = np.array([flux_err_r, flux_err_i, flux_err_z]).T

def flux2mag(flux):
    return -2.5 * np.log10(flux) + 30

bands = ['r', 'i', 'z']

cols = [f'flux_{band}' for band in bands]+[f'flux_err_{band}' for band in bands]
df = pd.DataFrame(np.hstack([fluxes_d, fluxerrs_d]), columns=cols)

mags_d = np.zeros((len(df),len(bands)))
magerrs_d = np.zeros((len(df),len(bands)))

for i,band in enumerate(bands):
    print(i,band)
    mags_d[:,i] = flux2mag(fluxes_d[:,i])

colors = np.zeros((len(fluxes_d),len(bands)-1))
for i in range(len(bands)-1):
    colors[:,i] = mags_d[:,i] - mags_d[:,i+1]

normal_colors = np.mean(colors > -1, axis=1) == 1
normal_colors.sum()

df = df[normal_colors]

#mask faint objects, i < 25
df = df[flux2mag(df.flux_i.values) < 25]

data=df

data.loc[:,"mag_i"]=flux2mag(data.loc[:, 'flux_i'])
                          
# New Pandas Dataframe with only detected galaxies
# A value of 23.0 returns something similar to Balrog Y3
# A value of 23.5 maybe is similar to the WL sample in Y6.
# A value of 21.5 is maybe optimal for LSS samples in Y6.
#balrog_mocked = mock_balrog_sigmoid(deep_data, 23.0, nTrain)

nTrain = int(2e6)

# Scramble the order of the catalog for purposes of training
#indices = np.random.choice(nTrain, size=nTrain, replace=False)
indices = np.random.choice(nTrain, size=nTrain, replace=False)
hh = ns.hFunc(nTrain, sigma=(30,1))
metric = ns.AsinhMetric(lnScaleSigma=0.4, lnScaleStep=0.03)

som = ns.NoiseSOM(metric, fluxes_d[indices,:], fluxerrs_d[indices,:], \
    learning=hh, \
    shape=(26,26), \
    wrap=False,logF=True, \
    initialize='sample', \
    minError=0.02)

path_cats='/project2/chihway/raulteixeira/data/'
# And save the resultant weight matrix
np.save("%s/som_delve_metacal_gold_26x26_2e6.npy"%path_cats,som.weights)
print(time.time()-start_time, ' seconds')