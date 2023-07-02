import numpy as np
import pandas as pd
import h5py
import time

start_time=time.time()
METACAL_COADD_ID = np.load('/project2/chihway/raulteixeira/data/metcal_cut_ids_20230521_v3.npy')
bpz_columns = ['COADD_OBJECT_ID', 'BDF_FLUX_G', 'BDF_FLUX_R', 'BDF_FLUX_I', 'BDF_FLUX_Z',\
               'BDF_FLUX_ERR_G', 'BDF_FLUX_ERR_R', 'BDF_FLUX_ERR_I', 'BDF_FLUX_ERR_Z', 'BDF_MAG_I']

for i in range(1,50):
    print(f'{i:02}')
    bdf_df = pd.read_hdf(f'/project2/chihway/raulteixeira/data/bdf_photometry/dr3_1_1_bdf_0000{i:02}.h5') 
    BDF_COADD_ID = bdf_df.COADD_OBJECT_ID.values
    #Find the intersection of objects between the two arrays.
    #This generally selects objects that are both in the metacal catalog and in THIS specific BDF file.
    unique, index_bdf, index_metacal = np.intersect1d(BDF_COADD_ID, METACAL_COADD_ID, assume_unique = True, return_indices = True)
    #Get the ids from BDF and from metacal of the same objects
    # you can use index_bdf to apply the metacal cuts on your bdf catalog
    bdf_df = bdf_df.iloc[index_bdf][bpz_columns]
    if i==1: print(bdf_df.columns)
    bdf_df.to_hdf(f'/project2/chihway/raulteixeira/data/bdf_photometry/dr3_1_1_bdf_metacal_cut_BPZ_input_{i:02}.h5', key='df')
    print(f'time until file {i:02}', time.time()-start_time, 'seconds')

print(time.time()-start_time, 'seconds')