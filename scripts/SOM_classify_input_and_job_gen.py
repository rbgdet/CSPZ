import numpy as np
import h5py
import time
import pandas as pd
import subprocess as sp

# def band(i):
#     if i==0: return 'R'
#     elif i==1: return 'I'
#     elif i==2: return 'Z'
#     else: 
#         raise ValueError

# mask = np.load('/project/chihway/raulteixeira/data/metacal_gold_mask.npy')

# with h5py.File('/project/chihway/data/decade/metacal_gold_combined_20230613.hdf') as f:
#     print(f.keys())
#     flux_r, flux_i, flux_z = np.array(f['mcal_flux_noshear']).T
#     flux_err_r, flux_err_i, flux_err_z = np.array(f['mcal_flux_err_noshear']).T
    
#     fluxes_d = np.array([flux_r, flux_i, flux_z]).T
#     fluxerrs_d = np.array([flux_err_r, flux_err_i, flux_err_z]).T

#     df = pd.DataFrame()
#     df['COADD_OBJECT_ID'] = np.array(f['COADD_OBJECT_ID'])[mask]

# for i, (flux_d, fluxerr_d) in enumerate(zip(fluxes_d.T, fluxerrs_d.T)):
#     print(i)
#     df[f'FLUX_{band(i)}']=flux_d[mask]
#     df[f'FLUX_ERR_{band(i)}']=fluxerr_d[mask]

    
# #df = df.loc[mask]

# dfs = np.array_split(df, 47)
        
# for i, df_ in enumerate(dfs):
#     df_.to_csv(f'/project/chihway/raulteixeira/data/metacal_gold_fluxes+ids_{i:02}.csv.gz')

# file = open('/home/raulteixeira/repos/CSPZ/scripts/SOMclassify_metacal.py', 'r')

# string = file.read()

# list_lines = string.splitlines()

# string.split('\n')

# start_time=time.time()
# for i in range(47):
#     file = open('/home/raulteixeira/repos/CSPZ/scripts/SOMclassify_metacal.py', 'r')

#     string = file.read()

#     list_lines = string.splitlines()
    
#     file.close()
    
#     list_lines[17]=f"df = pd.read_csv('/project/chihway/raulteixeira/data/metacal_gold_fluxes+ids_{i:02}.csv.gz')"
#     list_lines[29]="som_weights = np.load('%s/som_delve_metacal_gold_26x26_2e6.npy'%outpath)"
#     list_lines[52]=f'filename = "%s/som_metacal_gold_wide_26x26_{i:02}.npz"%(outpath)'
    
#     list_lines = [l + '\n' for l in list_lines]
    
#     pars = open(f'/home/raulteixeira/repos/CSPZ/scripts/SOMclassify_metacal/script_mcalgold_{i:02}_26x26.py', mode='w')
#     pars.writelines(list_lines)
#     pars.close()
    
#     print(time.time()-start_time)

start_time=time.time()

for i in range(47):   
    lines = ['#!/bin/sh', '#SBATCH -t 2:00:00',
             '#SBATCH --partition=broadwl',
             '#SBATCH --account=pi-chihway',
             f'#SBATCH --job-name=SOM{i:02}',
             '#SBATCH --nodes=1',
             '#SBATCH --ntasks-per-node=28', 
             'source activate',
             'conda activate sompz',
             f'mpirun -n 28 python /home/raulteixeira/repos/CSPZ/scripts/SOMclassify_metacal/script_mcalgold_{i:02}_26x26.py']

    lines = [l + '\n' for l in lines]
    
    bash_job_path = f'/home/raulteixeira/jobs/SOMclassify_metacal/script_mcalgold_{i:02}.sh'
    pars = open(bash_job_path, mode='w')
    pars.writelines(lines)
    pars.close()
    sp.run(f'sbatch {bash_job_path}', shell=True)
    
    print(time.time()-start_time)
