import numpy as np
import pandas as pd
import time

cells = []
dfs=[]
t0 = time.time()

for i in range(90):
    cells.append(np.load(f'/project/chihway/raulteixeira/data/som_BalrogOfTheDECADE_20231002_26x26_{i:02}.npz')['cells'])
    dfs.append(pd.read_csv(f'/project/chihway/raulteixeira/data/BalrogOfTheDECADE_20231002/catbalrog_{i:02}.csv.gz'))
    print(time.time()-t0, 'seconds')
    
cells = np.concatenate(cells)
df = pd.concat(dfs)
df['wide_cells'] = cells

balrog = pd.read_csv('/project/chihway/raulteixeira/data/BalrogOfTheDECADE_20231002.csv.gz')

notincols = balrog.columns[~np.isin(balrog.columns, df.columns)].values
compcols = np.concatenate((['ID'], notincols))

balrog = balrog[compcols]

print('df set', time.time()-t0, 'seconds')

df.to_csv('/project/chihway/raulteixeira/data/som_BalrogOfTheDECADE_20231002_26x26_ids+cells+fluxes.csv.gz')

balrog.to_csv('/project/chihway/raulteixeira/data/som_BalrogOfTheDECADE_20231002_26x26_REST.csv.gz')

print('saving done', time.time()-t0, 'seconds')