import numpy as np
import pandas as pd
import time

def flux2mag(flux):
    return -2.5 * np.log10(flux) + 30

t0 = time.time()

f = np.load('/project/chihway/raulteixeira/data/Balrog_Extra_Cols_20231120.npz')['arr_0']
df = pd.read_csv('/project/chihway/raulteixeira/data/som_BalrogOfTheDECADE_20231002_26x26_ids+cells+fluxes.csv.gz')

fluxcols    = ['mcal_FLUX_i', 'mcal_FLUX_r', 'mcal_FLUX_z', ]
fluxerrcols = ['mcal_FLUX_i_ERR', 'mcal_FLUX_r_ERR', 'mcal_FLUX_z_ERR']

flux_r, flux_i, flux_z = (df[fluxcol] for fluxcol in fluxcols)
flux_err_r, flux_err_i, flux_err_z = (df[fluxerrcol] for fluxerrcol in fluxerrcols)

# flux_err_r, flux_err_i, flux_err_z = np.array(f['mcal_flux_err_noshear']).T
mag_r = flux2mag(flux_r)
mag_i = flux2mag(flux_i)
mag_z = flux2mag(flux_z)

mcal_pz_mask = ((mag_i < 23.5) & (mag_i > 18) & 
                (mag_r < 26)   & (mag_r > 15) & 
                (mag_z < 26)   & (mag_z > 15) & 
                (mag_r - mag_i < 4)   & (mag_r - mag_i > -1.5) & 
                (mag_i - mag_z < 4)   & (mag_i - mag_z > -1.5))

SNR     = f['SNR']
T_ratio = f['T_ratio']
T       = f['T']
flags   = f['flags']

fg = np.array(f['FLAGS_FOREGROUND'])

g1, g2  = f['g1'], f['g2']

#Metacal cuts based on DES Y3 ones (from here: https://des.ncsa.illinois.edu/releases/y3a2/Y3key-catalogs)
SNR_Mask   = (SNR > 10) & (SNR < 1000)
Tratio_Mask= T_ratio > 0.5
T_Mask     = T < 10
Flag_Mask  = flags == 0
#SG_Mask = (sg>=4)
FG_Mask = (fg==0)

Other_Mask = np.invert((T > 2) & (SNR < 30)) & np.invert((np.log10(T) < (22.25 - mag_r)/3.5) & (g1**2 + g2**2 > 0.8**2))

Mask = mcal_pz_mask & SNR_Mask & Tratio_Mask & T_Mask & Flag_Mask & Other_Mask & FG_Mask #& SG_Mask

np.savez('/project/chihway/raulteixeira/data/Balrog_20231120_MASK.npz', Mask)

print('time cut:', time.time()-t0, 'seconds')