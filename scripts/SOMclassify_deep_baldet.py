import numpy as np
import pickle
import matplotlib.pyplot as plt
import sys
#sys.path.append('/global/homes/c/carles/highz/src/gary_som/')
sys.path.append("/home/raulteixeira/repos/CSPZ/scripts/")
import NoiseSOM as ns
import multiprocessing as mp
import time
import warnings
import os
from schwimmbad import MPIPool
import pandas as pd

n_deep = 48
outpath = '/project/chihway/raulteixeira/data'

balrog_det_DF = pd.read_csv('%s/BalrogOfTheDECADE_20231002/deep_fields_obj_detected_BalrogoftheDECADE_20231002.csv.gz'%outpath)

#load deep catalog
bands = list('UGRIZJH')+['KS']
fluxes_d = np.array([balrog_det_DF[f'BDF_FLUX_DERED_CALIB_{band}'].values for band in bands]).T
fluxerrs_d = np.array([balrog_det_DF[f'BDF_FLUX_ERR_DERED_CALIB_{band}'].values for band in bands]).T
    
nTrain=len(fluxes_d)
# Here we just the weights and initialize the SOM.
som_weights = np.load("%s/som_des_DF_baldet_gold_48x48_112123.npy"%outpath)
hh = ns.hFunc(nTrain,sigma=(30,1))
metric = ns.AsinhMetric(lnScaleSigma=0.4,lnScaleStep=0.03)
som = ns.NoiseSOM(metric,None,None, \
    learning=hh, \
    shape=(n_deep,n_deep), \
    wrap=False,logF=True, \
    initialize=som_weights, \
    minError=0.02)

def fun(inputs):
    f, fe = inputs
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cells_test,dist_test = som.classify(f, fe)
    return cells_test
#"""
pool = MPIPool()
if not pool.is_master():
    pool.wait()
    sys.exit(0)
#"""    

filename = "%s/som_DES_DF_baldet_112123_48x48.npz"%(outpath)

t0 = time.time()

'''
def fun(_indx):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cells_test,dist_test = som.classify(fluxes_d[_indx,:], fluxerrs_d[_indx,:])
    return cells_test
p = mp.Pool(32)
cells_test = np.concatenate(p.map(fun, indices))
p.terminate()
'''
#breakpoint()

#np.save(outpath+'chosen_coadd_ids.npy', chosen_ids)

nranks = pool.comm.Get_size() - 1
indices = np.array_split(np.arange(len(fluxes_d)), nranks)
inputs = []
for _indx in indices:
    inputs.append([fluxes_d[_indx,:], fluxerrs_d[_indx,:]])

cells_final = np.concatenate(pool.map(fun, inputs))

np.savez(filename,cells=cells_final)
t1 = time.time()
print(t1-t0)
