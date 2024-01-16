import numpy as np
import h5py
import time
import pandas as pd
import subprocess as sp

file = open('/home/raulteixeira/repos/CSPZ/scripts/SOMclassify_metacal_seed.py', 'r')

string = file.read()

list_lines = string.splitlines()

string.split('\n')

start_time=time.time()
for i in range(109):
    file = open('/home/raulteixeira/repos/CSPZ/scripts/SOMclassify_metacal_seed.py', 'r')

    string = file.read()

    list_lines = string.splitlines()
    
    file.close()
    
    list_lines[20]=f"df = pd.read_hdf('%s/classify_sfd98_1p/cat_{i:03}.hdf5'%outpath, key='df')"
    list_lines[22]=f"flux_r, flux_i, flux_z = df['mcal_flux_1p_dered_sfd98_r'].values,\
    df['mcal_flux_1p_dered_sfd98_i'].values, df['mcal_flux_1p_dered_sfd98_z'].values"
    list_lines[23]=f"flux_err_r, flux_err_i, flux_err_z = df['mcal_flux_err_1p_dered_sfd98_r'].values,\
    df['mcal_flux_err_1p_dered_sfd98_i'].values, df['mcal_flux_err_1p_dered_sfd98_z'].values"
    list_lines[55]=f'filename = "%s/classify_sfd98_1p/som_metacal_all_gold_wide_32x32_{i:03}.npz"%(outpath)'
    
    list_lines = [l + '\n' for l in list_lines]
    
    pars = open(f'/home/raulteixeira/repos/CSPZ/scripts/SOMclassify_metacal_all_sfd98_1p/script_mcalgold_{i:03}_32x32.py', mode='w')
    pars.writelines(list_lines)
    pars.close()
    
    print(time.time()-start_time)

start_time=time.time()

for i in range(109):   
    lines = ['#!/bin/sh', '#SBATCH -t 10:00:00',
             '#SBATCH --partition=broadwl',
             '#SBATCH --account=pi-chihway',
             f'#SBATCH --job-name=1P{i:03}',
             '#SBATCH --nodes=1',
             '#SBATCH --ntasks-per-node=28', 
             'source activate',
             'conda activate sompz',
             f'mpirun -n 28 python /home/raulteixeira/repos/CSPZ/scripts/SOMclassify_metacal_all_sfd98_1p/script_mcalgold_{i:03}_32x32.py']

    lines = [l + '\n' for l in lines]
    
    bash_job_path = f'/home/raulteixeira/jobs/SOMclassify_metacal_all_sfd98_1p/script_mcalgold_{i:03}.sh'
    pars = open(bash_job_path, mode='w')
    pars.writelines(lines)
    pars.close()
    sp.run(f'sbatch {bash_job_path}', shell=True)
    
    print(time.time()-start_time)
