import numpy as np
import pandas as pd

df = pd.read_csv('/project/chihway/raulteixeira/data/BPZ+SOM_mcal_gold_wide_48x48_ids+cells+fluxes.csv.gz')

fun = 'mean'
feat = 'Z_SAMP'

square = df[[feat, 'cells']].groupby('cells').agg(['mean', 'median', 'std', len])
im_z = square[(feat, fun)].values

ids = np.argsort(im_z.flatten())

fun = 'len'
feat = 'Z_SAMP'

square = df[[feat, 'cells']].groupby('cells').agg(['mean', 'median', 'std', len])
im_len = square[(feat, fun)].values

percentages = im_len.flatten()[ids].cumsum()/len(df)

masks = []
for lim1, lim2 in zip([0, .25, .5, .75], [.25, .5, .75, 1.]):
    masks.append(ids[(lim1<percentages) & (percentages<=lim2)])

df['TomoBin'] = np.zeros(len(df))

for i, mask in enumerate(masks):
    df['TomoBin'][np.isin(df['cells'], ids[mask])] = i+1

df.to_csv('/project/chihway/raulteixeira/data/BPZ+SOM_mcal_gold_wide_48x48_ids+cells+fluxes+TomoBins.csv.gz')