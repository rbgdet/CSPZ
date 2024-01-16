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
np.random.seed(42)
n_deep = 64
outpath = '/project/chihway/raulteixeira/data'

DF = pd.read_csv('/project/chihway/raulteixeira/data/deepfields.csv.gz')
det_ids = np.load('%s/BalrogoftheDECADE_121723_detected_ids.npz'%outpath)['arr_0']
DF = DF[np.isin(DF.ID.values, det_ids)]
nTrain=len(DF)
bands = list('UGRIZJH')+['KS']

fluxes_d = np.array([DF[f'BDF_FLUX_DERED_CALIB_{band}'].values for band in bands]).T
fluxerrs_d = np.array([DF[f'BDF_FLUX_ERR_DERED_CALIB_{band}'].values for band in bands]).T

def flux2mag(flux):
    return -2.5 * np.log10(flux) + 30

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
np.save("%s/som_weights_des_DF_121723_64x64.npy"%outpath,som.weights)
print(time.time()-start_time, ' seconds')