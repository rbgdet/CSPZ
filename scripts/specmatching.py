import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord
from astropy import units as u
import glob

# Loop over all files

filenames = glob.glob("/home/raulteixeira/scratch-midway2/dr3_1_1_bdf*")

def get_dataframes(filenames):
    all_dataframes = []
    for filename in filenames:
        # Open catalogs
        all_dataframes.append(pd.read_hdf(filename, key='data/table', mode='r'))

    data_df = pd.concat(all_dataframes)
    
    return data_df

# Open catalogs
data_df = get_dataframes(filenames)
data_df.memory_usage()
print('data_df')
specz = pd.read_table("/project2/chihway/delve_shear/BRPORTAL_E_6315_18670.csv", sep=',')
print('specz')

# Match spectra to DELVE catalog
spec_cd = SkyCoord(ra=specz['RA'].values*u.degree, dec=specz['DEC'].values*u.degree)
delve_cd = SkyCoord(ra=data_df['RA'].values*u.deg, dec=data_df['DEC'].values*u.deg)
idx, d2d, d3d = delve_cd.match_to_catalog_sky(spec_cd)
good_matches = d2d < 1.0*u.arcsec
print('matching')

print(len(np.unique(idx[good_matches])), np.count_nonzero(good_matches))

# Add spectra to DELVE catalog
data_df['zspec'] = np.nan
data_df.loc[good_matches, 'zspec'] = specz.iloc[idx[good_matches], specz.columns.get_loc('Z')].values
print('added spectra')

# # Save each tile separately
# unique_tilenames = np.unique(data_df.TILENAME)
# unique_tilenames.shape

# for tilename in unique_tilenames:
#     data_df_sub = data_df[data_df.TILENAME==tilename]
#     data_df_sub.to_csv('/home/raulteixeira/scratch-midway2/CosmicShearData/spectiles/'+tilename+'.csv.gz', sep=',', header=True, index=False)

# Save one file containing the subset with matched spectra.
data_df_spec = data_df[np.isfinite(data_df.zspec.values)]
data_df_spec.to_csv('/home/raulteixeira/scratch-midway2/CosmicShearData/DELVE_BDF_data_with_zspec022023.csv.gz', sep=',', header=True, index=False)
print('done')