import numpy as np
import h5py
import time
import pandas as pd
import matplotlib.pyplot as plt
import glob
import scipy
import pickle as pkl
import matplotlib as mpl
from astropy.coordinates import SkyCoord
from astropy import units as u
import healpy as hp
import scipy
from scipy.optimize import differential_evolution
from matplotlib.colors import LogNorm
outpath = '/project/chihway/raulteixeira/data/'
n_deep, n_wide = 64, 32
N_deep, N_wide = n_deep**2, n_wide**2
nbins = 50
zmin, zmax = 0., 2.
deltaz = (zmax-zmin)/nbins
assert ((zmin>=0) & (zmax>=zmin) & (nbins>0) & (n_deep>0) & (n_wide>0))
z_edges = np.linspace(zmin,zmax,nbins) #np.arange(zmin,zmax,deltaz)
zs = .5*np.array([z_edges[i]+z_edges[i+1] for i in range(nbins-1)])
bins_0 = [.36, .63, .86]

def hist(group):
    '''
    Returns the counts of each bin in a histogram of sample a (numerical list-like object)
    '''
    return list(np.histogram(group.Z, bins=z_edges, density=False, weights=group.w)[0])

def flux2mag(flux):
    '''
    Converts fluxes to magnitudes
    '''
    return -2.5 * np.log10(flux) + 30

def bin_loss(bins=bins_0, n_bins=4):
    bins = np.concatenate(([0], bins, [2]))
    binned_counts = np.zeros((n_bins, N_wide))

    redshifts = (balrog_deep[['Z', 'wide_cells']][balrog_deep['Gold_Mask']])
    Zs = redshifts['Z'].values
    bin_masks = [((bins[i]<Zs) & (Zs<bins[i+1])) for i in range(n_bins)]
    bin_groups = [redshifts[bin_masks[i]].groupby('wide_cells').agg([len]) for i in range(n_bins)]
    binned_cell_idx = [np.array(bin_group.Z.index.astype(int)) for bin_group in bin_groups]
    for i in range(n_bins):
        binned_counts[i][binned_cell_idx[i]] = bin_groups[i].Z.len

    tomo_idx = []
    for i in range(n_bins):
        tomo_idx.append(np.arange(N_wide)[np.argmax(binned_counts, axis=0)==i])

    share = [np.sum(counts_wide[tomo_idx[i]]) for i in range(n_bins)]
    norm_share = share/sum(share)
    return np.sum((norm_share - 0.25)**2)

def tb_share(bins=bins_0):
    bins = np.concatenate(([0], bins, [2]))
    binned_counts = np.zeros((n_bins, N_wide))

    redshifts = (balrog_deep[['Z', 'wide_cells']][balrog_deep['Gold_Mask']])
    Zs = redshifts['Z'].values
    bin_masks = [((bins[i]<Zs) & (Zs<bins[i+1])) for i in range(n_bins)]
    bin_groups = [redshifts[bin_masks[i]].groupby('wide_cells').agg([len]) for i in range(n_bins)]
    binned_cell_idx = [np.array(bin_group.Z.index.astype(int)) for bin_group in bin_groups]
    for i in range(n_bins):
        binned_counts[i][binned_cell_idx[i]] = bin_groups[i].Z.len

    tomo_idx = []
    for i in range(n_bins):
        tomo_idx.append(np.arange(N_wide)[np.argmax(binned_counts, axis=0)==i])

    share = [np.sum(counts_wide[tomo_idx[i]]) for i in range(n_bins)]
    norm_share = share/sum(share)
    return norm_share

deep = pd.read_hdf('%s/DES_DF_baldet_121923_64x64_cells.hdf'%outpath, key='df')
redshift_DF = pd.read_csv('/project/chihway/raulteixeira/data/deepfields_with_redshifts.csv.gz')
deep = deep.merge(redshift_DF[['ID', 'Z']], on='ID', how='left')

with h5py.File('/project/chihway/dhayaa/DECADE/BalrogOfTheDECADE_20231216.hdf5') as f:
    ID, ids, tilename = np.array(f['ID']), np.array(f['id']), np.array(f['tilename'])
    
    flux_r, flux_i, flux_z = np.array(f['mcal_flux_noshear_dered_sfd98']).T
    flux_err_r, flux_err_i, flux_err_z = np.array(f['mcal_flux_err_noshear_dered_sfd98']).T

    # flux_err_r, flux_err_i, flux_err_z = np.array(f['mcal_flux_err_noshear']).T
    mag_r = flux2mag(flux_r)
    mag_i = flux2mag(flux_i)
    mag_z = flux2mag(flux_z)

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

    Balrog_Gold_Mask = mcal_pz_mask & SNR_Mask & Tratio_Mask & T_Mask & Flag_Mask & Other_Mask #& FG_Mask #& SG_Mask
    
    true_ra, true_dec = np.array(f['true_ra']), np.array(f['true_dec'])
    detected     = np.array(f['detected'])
    d_contam_arcsec  = np.array(f['d_contam_arcsec'])
    contmask = d_contam_arcsec > 1.5
    
    data = np.array([ID, ids, tilename, true_ra, true_dec, flux_r, flux_i, flux_z, flux_err_r, flux_err_i, flux_err_z]).T
    cols = ['ID', 'id', 'tilename', 'true_ra', 'true_dec']+[f'flux_{band}' for band in 'riz']+[f'flux_err_{band}' for band in 'riz']
    balrog = pd.DataFrame(data=data, columns=cols)
    balrog['Gold_Mask'] = Balrog_Gold_Mask
    balrog['FG_Mask'] = FG_Mask
    balrog['Cont_Mask'] = contmask
    balrog['detected'] = detected
    balrog['wide_cells'] = np.load("%s/som_BalrogoftheDECADE_121923_32x32.npz"%outpath)['cells']
# add balrog dr3_1_1 footprint cut

print(f'fraction of uncontaminated foreground balrog injections that pass the gold cut and WL selection:\
      {np.mean((balrog["Gold_Mask"])[contmask&balrog["FG_Mask"]]):.2}')

balrog_clean = balrog[balrog['Cont_Mask']&balrog["FG_Mask"]]

balrog_deep = balrog_clean.merge(deep, on='ID', how='left') # merge balrog w/ deep catalog for later
                                                      # will have same length as barlog catalog

deep_columns = list(deep.columns)
deep_columns.remove('deep_cells')

missing_cells = np.arange(N_deep)[~np.isin(np.arange(N_deep), balrog_deep['deep_cells'])]
n_missing_cells = len(missing_cells)
for i in range(n_missing_cells):
    balrog_deep.loc[len(balrog_deep)+i]=\
    [np.nan if col in cols+deep_columns\
     else (0. if col!='deep_cells' else missing_cells[i]) for col in balrog_deep.columns]

balrog_deep['Gold_Mask'] = balrog_deep['Gold_Mask'].astype(bool)

wide = pd.read_hdf(f"%s/classify_sfd98_2p/som_metacal_all_gold_wide_32x32_full.hdf5"%outpath, key='df')
wide['COADD_OBJECT_ID']=wide['id'].astype(int)

#p(\hat{c}) - i.e. probability that a galaxy in the wide fields will belong to a cell c 
square_p_c_hat = wide[['COADD_OBJECT_ID', 'wide_cells']].groupby('wide_cells').agg([len])
counts_wide = square_p_c_hat.COADD_OBJECT_ID.len

bins_0 = np.random.normal([0.358, 0.631, 0.872], scale=.1)

result_DE = differential_evolution(bin_loss, bounds=[(0,2), (0,2), (0,2)], x0=bins_0)
bins_DE = result_DE.x
print(result_DE)

n_bins = 4
N_wide=32**2
Wide_bins = np.zeros(N_wide, dtype = int).flatten()
bins = np.concatenate(([0], bins_DE, [2]))
binned_counts = np.zeros((n_bins, N_wide))

catalog = balrog_deep[['Z', 'wide_cells']][balrog_deep['Gold_Mask'].astype(bool)]
Zs = catalog['Z']

for i in range(N_wide):
    Balrog_in_this_cell = catalog['wide_cells'].values == i
    redshift_in_this_cell = catalog['Z'].values[Balrog_in_this_cell] #Deep/true Redshift of all gals in this cell
    bcounts_in_this_cell  = np.histogram(redshift_in_this_cell, bins)[0] #How many deep galaxy counts per z-bin
    Wide_bins[i]          = np.argmax(bcounts_in_this_cell) #Which bin is most populated by galaxies from this cell?

print(bins)
# plt.title('DES Y3 Binning')
# plt.imshow(Wide_bins.reshape(n_wide, n_wide))

wide['TomoBin'] = -99*np.ones(len(wide))
for i in range(4):
    wide['TomoBin'][np.isin(wide['wide_cells'], np.arange(0, N_wide)[Wide_bins==i])]=i
(wide['TomoBin']==-99).any()

wide[['id', 'TomoBin']].to_hdf('%s/classify_sfd98_2p/DR3_1_ID+TomoBin.hdf5'%outpath, key='df')