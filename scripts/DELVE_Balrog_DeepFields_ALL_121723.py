import pandas as pd
import numpy as np
import h5py
outpath = '/project/chihway/raulteixeira/data'

def flux2mag(flux):
    '''
    Converts fluxes to magnitudes
    '''
    return -2.5 * np.log10(flux) + 30

def deep_cuts(df):
    '''
    places color cuts on deep field catalog
    Credit: Alex Alarcon
    '''
    
    #Mask flagged regions -- not needed, saved deep catalog already has flag cuts in place
    mask = df.MASK_FLAGS_NIR==0
    mask &= df.MASK_FLAGS==0
    mask &= df.FLAGS_NIR==0
    mask &= df.FLAGS==0
    mask &= df.FLAGSTR=="b'ok'"
    mask &= df.FLAGSTR_NIR=="b'ok'"
    #df = df[mask]
    df = df.drop(columns=[
        "MASK_FLAGS",
        "MASK_FLAGS_NIR",
        "FLAGS",
        "FLAGS_NIR",
        "FLAGSTR",
        "FLAGSTR_NIR",
    ])
    deep_bands_ = ["U","G","R","I","Z","J","H","KS"]
    # remove crazy colors, defined as two 
    # consecutive colors (e.g u-g, g-r, r-i, etc) 
    # that have a value smaler than -1
    mags_d = np.zeros((len(df),len(deep_bands_)))
    magerrs_d = np.zeros((len(df),len(deep_bands_)))

    for i,band in enumerate(deep_bands_):
        #print(i,band)
        mags_d[:,i] = flux2mag(df['BDF_FLUX_DERED_CALIB_%s'%band])

    colors = np.zeros((len(df),len(deep_bands_)-1))
    for i in range(len(deep_bands_)-1):
        colors[:,i] = mags_d[:,i] - mags_d[:,i+1]

    normal_colors = np.mean(colors > -1, axis=1) == 1
    normal_colors.sum()

    i = flux2mag(df.BDF_FLUX_DERED_CALIB_I.values)
    r = flux2mag(df.BDF_FLUX_DERED_CALIB_R.values)
    z = flux2mag(df.BDF_FLUX_DERED_CALIB_Z.values)
    k = flux2mag(df.BDF_FLUX_DERED_CALIB_KS.values)

    return mask&(normal_colors) 
    #remove (flux2mag(df.BDF_FLUX_DERED_CALIB_I.values) < 25)&((z-k) > 0.5*(r-z)), balrog is a better selection

raw_DF = pd.read_csv('/project/chihway/raulteixeira/data/deepfields.csv.gz')
redshift_DF = pd.read_csv('/project/chihway/raulteixeira/data/deepfields_with_redshifts.csv.gz')
raw_DF_with_redshifts = raw_DF.merge(redshift_DF[['ID', 'Z']], on='ID', how='left')

raw_DF_with_redshifts.to_csv('/project/chihway/raulteixeira/data/RAW_deepfields_with_redshifts.csv.gz')

deep_mask = deep_cuts(raw_DF_with_redshifts)
print(deep_mask.sum())
clean_DF_with_redshifts = raw_DF_with_redshifts[deep_mask]

with h5py.File('/project/chihway/dhayaa/DECADE/BalrogOfTheDECADE_20231216.hdf5') as f:
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

    Mask = mcal_pz_mask & SNR_Mask & Tratio_Mask & T_Mask & Flag_Mask & Other_Mask & FG_Mask #& SG_Mask
    
    detected = np.array(f['detected']).astype(bool)
    
    assert (Mask==(Mask&detected)).all()
    
    ID = np.array(f['ID'])
    det_ids = ID[Mask]
    det_DF_ids = clean_DF_with_redshifts.ID.values[np.isin(clean_DF_with_redshifts.ID.values, det_ids)]
    print(det_DF_ids.shape)
    np.savez('%s/BalrogoftheDECADE_121723_detected_ids.npz'%outpath, det_DF_ids)
print('DONE')