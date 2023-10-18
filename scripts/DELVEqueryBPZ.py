import easyaccess
import numpy as np
import fitsio
import os
import subprocess as sp
import pandas as pd

bpzdir = '/home/raulteixeira/repos/DESC_BPZ/scripts/bpz.py'

metadata = np.genfromtxt('/project/chihway/chihway/shearcat/Tilelist/11072023/new_final_list_DR3_1_1.txt', dtype='str', delimiter=",")[1:]

def flux2mag(flux):
    return -2.5 * np.log10(flux) + 30

cols = ['COADD_OBJECT_ID', 'BDF_FLUX_G', 'BDF_FLUX_R', 'BDF_FLUX_I', 'BDF_FLUX_Z', 'BDF_FLUX_ERR_G', 'BDF_FLUX_ERR_R', 'BDF_FLUX_ERR_I', 'BDF_FLUX_ERR_Z', 'BDF_MAG_I']

m0col = 'BDF_MAG_I'
f0col = 'BDF_FLUX_I'

prior = 'sva1_weights'

for i in range(5):#len(metadata)):   
    tile = metadata[i][0]
    print(i, tile)

    h5dir = '/home/raulteixeira/scratch-midway2/CosmicShearData/bpztiles/pzinput/gold_'+str(tile)+'.h5'
    parsdir = f'/project/chihway/raulteixeira/data/DELVE_gold_{str(tile)}.pars'
    
    decade_query = "select a.COADD_OBJECT_ID, a.RA, a.DEC, a.FLUX_AUTO_G, a.FLUX_AUTO_R, a.FLUX_AUTO_I, a.FLUX_AUTO_Z, a.FLUXERR_AUTO_G, a.FLUXERR_AUTO_R, a.FLUXERR_AUTO_I, a.FLUXERR_AUTO_Z, b.BDF_FLUX_G, b.BDF_FLUX_R, b.BDF_FLUX_I, b.BDF_FLUX_Z, b.BDF_FLUX_ERR_G, b.BDF_FLUX_ERR_R, b.BDF_FLUX_ERR_I, b.BDF_FLUX_ERR_Z from DECADE.DR3_1_COADD_OBJECT_SUMMARY a, DR3_1_SOF b where a.COADD_OBJECT_ID=b.COADD_OBJECT_ID and a.TILENAME='"+str(tile)+"';"

    # get the decade qa data as a pandas dataframe
    conn = easyaccess.connect(section='decade')
    decade_df = conn.query_to_pandas(decade_query)
    data = decade_df.to_records()
    print(len(data))

    pos_mask = data[f0col]>0
    data=data[pos_mask]
    dframe_i = pd.DataFrame()
    if len(data)>0:            
        for label in cols:
            if label!=m0col: dframe_i[label] = data[label]
            else: dframe_i[m0col] = flux2mag(data[f0col])

        dframe_i.to_hdf(h5dir, key='df')
        list_lines = ['COLUMNS    /home/raulteixeira/repos/DESC_BPZ/tests/CosmicShearPZ_BDF_Flux.columns\n'
                  , f'OUTPUT\t    /project/chihway/raulteixeira/data/BPZ_gold_DELVE/pz_{prior}_DELVE_gold_{tile}.h5\n'
                  , 'SPECTRA     CWWSB4.list\n'
                  , f'PRIOR\t     {prior}\n'
                  , 'DZ          0.01\n'
                  , 'ZMIN        0.005\n'
                  , 'ZMAX        3.505\n'
                  , 'MAG         no\n'
                  , 'NEW_AB      no\n'
                  , 'MADAU\t    no #TURN OFF MADAU!!!!\n'
                  , 'EXCLUDE     none\n'
                  , 'CHECK       yes\n'
                  , '#ZC          1.0,2.0\n'
                  , '#FC          0.2,0.4\n'
                  , 'VERBOSE     no\n'
                  , '#INTERP      0\n'
                  , 'ODDS        0.68\n'
                  , f'PROBS      no\n'
                  , f'PROBS2     no\n'
                  , f'PROBS_LITE no\n'
                  , 'GET_Z       yes\n'
                  , 'INTERACTIVE yes\n'
                  , 'PLOTS       no\n'
                  , 'SAMPLING yes\n'
                  , 'NSAMPLES 1\n'
                  , 'SEED 42\n'
                  , '#ONLY_TYPE   yes\n']

        pars = open(parsdir, mode='w')
        pars.writelines(list_lines)
        pars.close()

        command = f'python -u {bpzdir} {h5dir} -P {parsdir}'
        sp.run(command, shell = True)
        sp.run(f'rm {h5dir}', shell = True)
        sp.run(f'rm {pars}', shell = True)
    else: print(f'{str(tile)} empty tile')
