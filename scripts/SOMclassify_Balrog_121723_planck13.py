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
import h5py
nwide=32
outpath = '/project/chihway/raulteixeira/data'

catpath = '/project/chihway/dhayaa/DECADE/BalrogOfTheDECADE_20231216.hdf5'
with h5py.File(catpath) as f:
    flux_r, flux_i, flux_z = np.array(f['mcal_flux_noshear_dered_planck13']).T
    flux_err_r, flux_err_i, flux_err_z = np.array(f['mcal_flux_err_noshear_dered_planck13']).T
    
fluxes_d = np.array([flux_r, flux_i, flux_z]).T
fluxerrs_d = np.array([flux_err_r, flux_err_i, flux_err_z]).T
    
nTrain=int(2e6)
# Here we just input the weights and initialize the SOM.
som_weights = np.load("%s/som_delve_metacal_gold_32x32_seed42_nTrain2e6_121723.npy"%outpath)
hh = ns.hFunc(nTrain,sigma=(30,1))
metric = ns.AsinhMetric(lnScaleSigma=0.4,lnScaleStep=0.03)
som = ns.NoiseSOM(metric,None,None, \
    learning=hh, \
    shape=(nwide,nwide), \
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

filename = "%s/som_BalrogoftheDECADE_121923_32x32_planck13.npz"%(outpath)

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