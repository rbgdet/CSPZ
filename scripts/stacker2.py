import matplotlib.pyplot as plt
import numpy as np
import shelve
import pandas as pd
import h5py
import glob
import astropy
from astropy.table import Table as Table
    
start = 50; end = 75
    
fnamelist = glob.glob('/home/raulteixeira/scratch-midway2/CosmicShearData/bpztiles/output/probs/*METACAL*')

tiles = []
for fname in fnamelist:
    tiles.append(fname.split('_')[-1].split('.')[0])

tiles = np.unique(tiles)

tile = tiles[0]
path = f'/home/raulteixeira/scratch-midway2/CosmicShearData/bpztiles/output/probs/pz_METACAL4_probs_{tile}'
probs = shelve.open(path)
IDs=list(probs.keys())[2:]
zs = probs[IDs[0]][0]

POSTERIOR1 = POSTERIOR2 = POSTERIOR3 = np.zeros_like(zs)
all_samples1 = all_samples2 = all_samples3 = np.array([])

Zs1 = Zs2 = Zs3 = IDs1 = IDs2 = IDs3 = np.array([])
all_samples1 = all_samples2 = all_samples3 = np.array([])

for tile in tiles[start:end]:
    # opening probs files
    path = f'/home/raulteixeira/scratch-midway2/CosmicShearData/bpztiles/output/probs/pz_cosmos_Laigle_py3_METACAL4_probs_{tile}'
    probs1 = shelve.open(path)

    path = f'/home/raulteixeira/scratch-midway2/CosmicShearData/bpztiles/output/probs/pz_hdfn_gen_METACAL4_probs_{tile}'
    probs2 = shelve.open(path)

    path = f'/home/raulteixeira/scratch-midway2/CosmicShearData/bpztiles/output/probs/pz_sva1_weights_METACAL4_probs_{tile}'
    probs3 = shelve.open(path)

    # opening samples files
    path = f'/home/raulteixeira/scratch-midway2/CosmicShearData/bpztiles/output/pzs/pz_cosmos_Laigle_py3_METACAL4_{tile}_ITS.txt'
    samples1 = np.loadtxt(path)

    path = f'/home/raulteixeira/scratch-midway2/CosmicShearData/bpztiles/output/pzs/pz_hdfn_gen_METACAL4_{tile}_ITS.txt'
    samples2 = np.loadtxt(path)

    path = f'/home/raulteixeira/scratch-midway2/CosmicShearData/bpztiles/output/pzs/pz_sva1_weights_METACAL4_{tile}_ITS.txt'
    samples3 = np.loadtxt(path)
    
    #BPZ h5 file
    filename = f'/home/raulteixeira/scratch-midway2/CosmicShearData/bpztiles/output/pzs/pz_cosmos_Laigle_py3_METACAL4_{tile}.h5'
    with h5py.File(filename) as f:
        bpzres1={}
        for key in list(f.keys()):
            bpzres1[key] = np.array(f[key])
            
    filename = f'/home/raulteixeira/scratch-midway2/CosmicShearData/bpztiles/output/pzs/pz_hdfn_gen_METACAL4_{tile}.h5'
    with h5py.File(filename) as f:
        bpzres2={}
        for key in list(f.keys()):
            bpzres2[key] = np.array(f[key])
    
    filename = f'/home/raulteixeira/scratch-midway2/CosmicShearData/bpztiles/output/pzs/pz_sva1_weights_METACAL4_{tile}.h5'
    with h5py.File(filename) as f:
        bpzres3={}
        for key in list(f.keys()):
            bpzres3[key] = np.array(f[key])
            
    #IDs
    iIDs1=list(probs1.keys())[2:]
    iIDs2=list(probs2.keys())[2:]
    iIDs3=list(probs3.keys())[2:]
    length1 = len(iIDs1)
    length2 = len(iIDs2)
    length3 = len(iIDs3)
    
    all_samples1 = np.concatenate((all_samples1, samples1))
    all_samples2 = np.concatenate((all_samples2, samples2))
    all_samples3 = np.concatenate((all_samples3, samples3))
       
    Zs1 = np.concatenate((Zs1, bpzres1['Z_B']))
    IDs1 = np.concatenate((IDs1, bpzres1['ID']))
    Zs2 = np.concatenate((Zs2, bpzres2['Z_B']))
    IDs2 = np.concatenate((IDs2, bpzres2['ID']))
    Zs3 = np.concatenate((Zs3, bpzres3['Z_B']))
    IDs3 = np.concatenate((IDs3, bpzres3['ID']))
            
    posteriortotal1 = posteriortotal2 = posteriortotal3 = np.zeros_like(zs)

    for i, ID in enumerate(iIDs1):
        if i%1000==0: print(i, 'out of', length1)
        posterior = np.sum(probs1[ID][2]*probs1[ID][1], axis=1)
        norm = np.sum(posterior)
        posterior /= norm
        posteriortotal1 += posterior
    
    POSTERIOR1 += posteriortotal1
    
    for i, ID in enumerate(iIDs2):
        if i%1000==0: print(i, 'out of', length2)
        posterior = np.sum(probs2[ID][2]*probs2[ID][1], axis=1)
        norm = np.sum(posterior)
        posterior /= norm
        posteriortotal2 += posterior
    
    POSTERIOR2 += posteriortotal2
    
    for i, ID in enumerate(iIDs3):
        if i%1000==0: print(i, 'out of', length3)
        posterior = np.sum(probs3[ID][2]*probs3[ID][1], axis=1)
        norm = np.sum(posterior)
        posterior /= norm
        posteriortotal3 += posterior
    
    POSTERIOR3 += posteriortotal3

BPZs1 = Table()
BPZs1['COADD_OBJECT_ID'], BPZs1['Z_B1'], BPZs1['Z_SAMP1'] = IDs1, Zs1, all_samples1
BPZs2 = Table()
BPZs2['COADD_OBJECT_ID'], BPZs2['Z_B2'], BPZs1['Z_SAMP2'] = IDs2, Zs2, all_samples2
BPZs3 = Table()
BPZs3['COADD_OBJECT_ID'], BPZs3['Z_B3'], BPZs1['Z_SAMP3'] = IDs3, Zs3, all_samples3

cat = Table.read('/home/raulteixeira/scratch-midway2/CosmicShearData/pzinput/fits/shear_test_coadd_object_gold.fits')

BPZs = [BPZs1, BPZs2, BPZs3]
for BPZ in BPZs:
    cat = astropy.table.join(cat, BPZ, keys='COADD_OBJECT_ID')

cat.write('/home/raulteixeira/scratch-midway2/CosmicShearData/bpztiles/BPZcat_after25_2.fits')

np.savetxt('/home/raulteixeira/scratch-midway2/CosmicShearData/bpztiles/post1_after25_2.txt', POSTERIOR1)
np.savetxt('/home/raulteixeira/scratch-midway2/CosmicShearData/bpztiles/post2_after25_2.txt', POSTERIOR2)
np.savetxt('/home/raulteixeira/scratch-midway2/CosmicShearData/bpztiles/post3_after25_2.txt', POSTERIOR3)
