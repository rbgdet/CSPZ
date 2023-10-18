import numpy as np
import h5py
import time
import pandas as pd
import matplotlib.pyplot as plt
import glob

fnames = glob.glob(f'/project2/chihway/raulteixeira/data/BPZ_bdf/hdfn_gen/CWWSB4/BPZ_BDF_HDFN_CWWSB4_*_mcal_gold.h5')

dfs = []
for fname in fnames:
    df_i = pd.DataFrame()
    with h5py.File(fname) as f:
        for key in list(f.keys()):
            df_i[key] = np.array(f[key])
    dfs.append(df_i)
    
df_ = pd.concat(dfs)

df_['COADD_OBJECT_ID']=df_['ID']

df=pd.read_csv('/project/chihway/raulteixeira/data/som_metacal_gold_wide_48x48_ids+cells+fluxes.csv.gz')

df = df.merge(df_, on='COADD_OBJECT_ID')

df.to_csv('/project/chihway/raulteixeira/data/BPZ+SOM_mcal_gold_wide_48x48_ids+cells+fluxes.csv.gz')