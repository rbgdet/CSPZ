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

n_deep = 48
outpath = '/project/chihway/raulteixeira/data'

balrog_det_DF = pd.read_hdf('%s/deep_Balrog_DR3_1_1_fp_GOLD_WL_FG.hdf5'%outpath, key='df')

bands = list('UGRIZJH')+['KS']

fluxes_d = np.array([balrog_det_DF[f'BDF_FLUX_DERED_CALIB_{band}'].values for band in bands]).T
fluxerrs_d = np.array([balrog_det_DF[f'BDF_FLUX_ERR_DERED_CALIB_{band}'].values for band in bands]).T

def flux2mag(flux):
    return -2.5 * np.log10(flux) + 30

# cols = [f'flux_{band.lower()}' for band in bands]+[f'flux_err_{band.lower()}' for band in bands]
# df = pd.DataFrame(np.hstack([fluxes_d, fluxerrs_d]), columns=cols)

# mags_d = np.zeros((len(df),len(bands)))
# magerrs_d = np.zeros((len(df),len(bands)))

# for i,band in enumerate(bands):
#     print(i,band)
#     mags_d[:,i] = flux2mag(fluxes_d[:,i])

# colors = np.zeros((len(fluxes_d),len(bands)-1))
# for i in range(len(bands)-1):
#     colors[:,i] = mags_d[:,i] - mags_d[:,i+1]

# normal_colors = np.mean(colors > -1, axis=1) == 1
# normal_colors.sum()

# df = df[normal_colors]

# #mask faint objects, i < 25
balrog_det_DF = balrog_det_DF[flux2mag(balrog_det_DF.BDF_FLUX_DERED_CALIB_I.values) < 25]
# data=df

# data.loc[:,"mag_i"]=flux2mag(data.loc[:, 'flux_i'])
                          
# New Pandas Dataframe with only detected galaxies
# A value of 23.0 returns something similar to Balrog Y3
# A value of 23.5 maybe is similar to the WL sample in Y6.
# A value of 21.5 is maybe optimal for LSS samples in Y6.
#balrog_mocked = mock_balrog_sigmoid(deep_data, 23.0, nTrain)

nTrain = len(fluxes_d)#int(2e6)

# Scramble the order of the catalog for purposes of training
#indices = np.random.choice(nTrain, size=nTrain, replace=False)
indices = np.random.choice(nTrain, size=nTrain, replace=False)
hh = ns.hFunc(nTrain, sigma=(30,1))
metric = ns.AsinhMetric(lnScaleSigma=0.4, lnScaleStep=0.03)

som = ns.NoiseSOM(metric, fluxes_d[indices,:], fluxerrs_d[indices,:], \
    learning=hh, \
    shape=(n_deep,n_deep), \
    wrap=False,logF=True, \
    initialize='sample', \
    minError=0.02)

# And save the resultant weight matrix
np.save("%s/som_weights_des_DF_balrog_pass_gold+wl+fg_48x48_112523.npy"%outpath,som.weights)
print(time.time()-start_time, ' seconds')