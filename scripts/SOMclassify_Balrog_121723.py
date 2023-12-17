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
outpath = '/project/chihway/raulteixeira/data'

df = pd.read_csv('%s/BalrogoftheDECADE_112523_MASKED_DET_CONTAM_1dot5_fluxes+errs+ID+id+tilename.csv.gz'%outpath)

flux_r, flux_i, flux_z = df['mcal_flux_r_noshear_dered_SFD98'].values, df['mcal_flux_i_noshear_dered_SFD98'].values, df['mcal_flux_z_noshear'].values

flux_err_r, flux_err_i, flux_err_z = df['mcal_flux_r_err_noshear_dered_SFD98'].values, df['mcal_flux_i_err_noshear_dered_SFD98'].values, df['mcal_flux_z_err_noshear_dered_SFD98'].values
    
fluxes_d = np.array([flux_r, flux_i, flux_z]).T
fluxerrs_d = np.array([flux_err_r, flux_err_i, flux_err_z]).T
    
nTrain=int(2e6)
# Here we just input the weights and initialize the SOM.
som_weights = np.load("%s/som_delve_metacal_gold_26x26_2e6.npy"%outpath)
hh = ns.hFunc(nTrain,sigma=(30,1))
metric = ns.AsinhMetric(lnScaleSigma=0.4,lnScaleStep=0.03)
som = ns.NoiseSOM(metric,None,None, \
    learning=hh, \
    shape=(26,26), \
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

filename = "%s/som_BalrogoftheDECADE_112523_26x26.npz"%(outpath)

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