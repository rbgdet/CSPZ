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
import glob

with h5py.File('/project/chihway/data/decade/metacal_gold_combined_20231212.hdf') as f:
    
    flux_r, flux_i, flux_z = np.array(f['mcal_flux_noshear_dered_sfd98']).T
    flux_err_r, flux_err_i, flux_err_z = np.array(f['mcal_flux_err_noshear_dered_sfd98']).T
    mag_r = 30 - 2.5*np.log10(flux_r)
    mag_i = 30 - 2.5*np.log10(flux_i)
    mag_z = 30 - 2.5*np.log10(flux_z)

    mcal_pz_mask = ((mag_i < 23.5) & (mag_i > 18) & 
                    (mag_r < 26)   & (mag_r > 15) & 
                    (mag_z < 26)   & (mag_z > 15) & 
                    (mag_r - mag_i < 4)   & (mag_r - mag_i > -1.5) & 
                    (mag_i - mag_z < 4)   & (mag_i - mag_z > -1.5))

    SNR     = np.array(f['mcal_s2n_noshear'])
    T_ratio = np.array(f['mcal_T_ratio_noshear'])
    T       = np.array(f['mcal_T_noshear'])
    flags   = np.array(f['mcal_flags'])
    
    #sg = np.array(f['sg_bdf'])
    fg = np.array(f['FLAGS_FOREGROUND'])
    
    g1, g2  = np.array(f['mcal_g_noshear']).T

    #Metacal cuts based on DES Y3 ones (from here: https://des.ncsa.illinois.edu/releases/y3a2/Y3key-catalogs)
    SNR_Mask   = (SNR > 10) & (SNR < 1000)
    Tratio_Mask= T_ratio > 0.5
    T_Mask     = T < 10
    Flag_Mask  = flags == 0
    #SG_Mask = (sg>=4)
    FG_Mask = (fg==0)

    Other_Mask = np.invert((T > 2) & (SNR < 30)) & np.invert((np.log10(T) < (22.25 - mag_r)/3.5) & (g1**2 + g2**2 > 0.8**2))

    Mask = mcal_pz_mask & SNR_Mask & Tratio_Mask & T_Mask & Flag_Mask & Other_Mask & FG_Mask #& SG_Mask

    np.save('/project/chihway/raulteixeira/data/metacal_gold_mask_all.npy', Mask)