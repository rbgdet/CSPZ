import pandas as pd
import numpy as np
outpath = '/project/chihway/raulteixeira/data'

# DF stands for Deep Fields

balrog = pd.read_csv('%s/som_BalrogOfTheDECADE_20231002_26x26_ids+cells+fluxes.csv.gz'%outpath)
balrog_rest = pd.read_csv('%s/som_BalrogOfTheDECADE_20231002_26x26_REST.csv.gz'%outpath)
balrog['detected']=balrog_rest['detected'] #detected objects

balrog_fp_mask = (balrog['true_ra'] < 180) & (balrog['true_dec'] > -25) #footprint mask for DR3_1_1
balrog = balrog[balrog_fp_mask]

det_ids = np.unique(balrog[balrog['detected'].values.astype(bool)]['ID']) # what we want for DF training
# np.savez('%s/BalrogOfTheDECADE_20231002/BalrogoftheDECADE_20231002_detected_ids.npz'%outpath, det_ids)

det_ids = np.load('%s/BalrogOfTheDECADE_20231002/BalrogoftheDECADE_20231002_detected_ids.npz'%outpath)['arr_0']
raw_DF = pd.read_csv('/project2/chihway/raulteixeira/data/deepfields.csv.gz')

balrog_det_DF = raw_DF[np.isin(raw_DF.ID.values, det_ids)]

# change to hdf
# balrog_det_DF.to_csv('%s/BalrogOfTheDECADE_20231002/deep_fields_obj_detected_BalrogoftheDECADE_20231002.csv.gz'%outpath)


