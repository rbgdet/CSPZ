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
nTrain = int(2e6)
length = 108904669 # Given by Chihway
indices = np.random.choice(length, size=nTrain, replace=False)

def flux2mag(flux):
    return -2.5 * np.log10(flux) + 30

with h5py.File('/project/chihway/data/decade/metacal_gold_combined_20231212.hdf') as f:
    
    flux_r, flux_i, flux_z = np.array(f['mcal_flux_noshear_dered_sfd98']).T[:,indices]
    flux_err_r, flux_err_i, flux_err_z = np.array(f['mcal_flux_err_noshear_dered_sfd98']).T[:,indices]
    
    #These are the fluxes with all metacal cuts applied
    flux_r = flux_r
    flux_i = flux_i
    flux_z = flux_z
    flux_err_r = flux_err_r
    flux_err_i = flux_err_i
    flux_err_z = flux_err_z

fluxes_d = np.array([flux_r, flux_i, flux_z]).T
fluxerrs_d = np.array([flux_err_r, flux_err_i, flux_err_z]).T

# Scramble the order of the catalog for purposes of training 
#Raul: I am scrambling the order when sourcing the catalog for memory-saving purposes

#indices = np.random.choice(nTrain, size=nTrain, replace=False)
hh = ns.hFunc(nTrain, sigma=(30,1))
metric = ns.AsinhMetric(lnScaleSigma=0.4, lnScaleStep=0.03)

som = ns.NoiseSOM(metric, fluxes_d, fluxerrs_d, \
    learning=hh, \
    shape=(26,26), \
    wrap=False,logF=True, \
    initialize='sample', \
    minError=0.02)

path_cats='/project2/chihway/raulteixeira/data/'
# And save the resultant weight matrix
np.save("%s/som_delve_metacal_gold_26x26_2e6.npy"%path_cats,som.weights)
print(time.time()-start_time, ' seconds')