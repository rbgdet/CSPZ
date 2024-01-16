import numpy as np
import pandas as pd
outpath = '/project/chihway/raulteixeira/data'

# WIDE

#sfd98
for sheartype in ['1p', '1m', '2p', '2m']:
    dfs = []
    for i in range(109):
        df = pd.read_hdf(f'%s/classify_sfd98_{sheartype}/cat_{i:03}.hdf5'%outpath, key='df')
        cells = np.load(f"%s/classify_sfd98_{sheartype}/som_metacal_all_gold_wide_32x32_{i:03}.npz"%outpath)['cells']
        df['wide_cells'] = cells
        dfs.append(df)
    df = pd.concat(dfs)
    df.to_hdf(f"%s/classify_sfd98_{sheartype}/som_metacal_all_gold_wide_32x32_full.hdf5"%outpath, key='df')
    del df, dfs, cells