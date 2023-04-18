import numpy as np
import subprocess as sp
import copy
import sys
import argparse
from astropy.table import Table

#Get all input arguments from command line
#so like my_script.py --Arg1 Val1 --Arg2 Val2 --Flag1 --Flag2
my_parser = argparse.ArgumentParser()

#First some required statements. These have to be passed everytime
#my_parser.add_argument('--tilename',  action='store', type = str,   required = True)
my_parser.add_argument('--spectra',   action='store', type = str,   required = True)
my_parser.add_argument('--prior',     action='store', type = str,   required = True)
#my_parser.add_argument('--OutPath',   action='store', type = str,   required = True)
my_parser.add_argument('--BatchSize', action='store', type = str,   required = True)
my_parser.add_argument('--nbatches',  action='store', type = str,   required = True)

#Optional ones. Default values given and can be varied if needed.
my_parser.add_argument('--dz',        action='store', type = float, default = 0.01)
my_parser.add_argument('--zmin',      action='store', type = float, default = 0.005)
my_parser.add_argument('--zmax',      action='store', type = float, default = 3.505)
my_parser.add_argument('--Nsamples',  action='store', type = int,   default = 1)

#Parse everything
args = vars(my_parser.parse_args())

#Print input args to help any future debugging
print('-------INPUT PARAMS----------')
for p in args.keys():
    print('%s : %s'%(p.upper(), args[p]))
print('-----------------------------')
print('-----------------------------')

#tile = args['tilename']

batch_size = int(args['BatchSize'])
n_batches = int(args['nbatches'])

spectra = args['spectra']
prior = args['prior']

dz = args['dz']
zmin = args['zmin']
zmax = args['zmax']
nsamples = args['Nsamples']


#f = open("/home/raulteixeira/repos/CSPZ/scripts/submit_bpz_batch.sh", "r")
#baselines = f.readlines()
for j in range(n_batches):
    list_lines = ['#!/bin/bash',
                'for ((i=%i;i<%i;i++))'%(j*batch_size,(j+1)*batch_size),
                '',
                'do',
                'echo $i',
                '',
                'cd /scratch/midway2/raulteixeira/CosmicShearData/',
                'mkdir tile_${i}',
                'cd tile_${i}',
                'cp /home/raulteixeira/repos/CSPZ/scripts/download_tile.py ./.',
                'cp /home/raulteixeira/repos/CSPZ/scripts/bpztilerun_template.py ./.',
                'cp /home/raulteixeira/repos/CSPZ/scripts/tile_DR3_1_1.csv ./.',
                '',
                'python download_tile.py ${i}',
                '',
                'echo \"#!/bin/sh',
                '#SBATCH -t 00:20:00',
                '#SBATCH --partition=chihway',
                '#SBATCH --account=pi-chihway',
                '#SBATCH --job-name=BPZ_${i}',
                '#SBATCH --nodes=1',
                '#SBATCH --ntasks-per-node=28',

                'python bpztilerun_template.py --tilename ${i}'+\
                            ' --spectra %s --prior %s'%(spectra, prior)+\
                            ' --OutPath /scratch/midway2/raulteixeira/CosmicShearData/tile_${i}/pzout_${i}.h5'+\
                            ' --dz %3f --zmin %3f --zmax %3f --Nsamples %i'%(dz, zmin, zmax, nsamples),
                '',
                'rm -rf /scratch/midway2/raulteixeira/CosmicShearData/tile_${i}/decade.ncsa.illinois.edu',
                'rm /scratch/midway2/raulteixeira/CosmicShearData/tile_${i}/*py',
                'rm /scratch/midway2/raulteixeira/CosmicShearData/tile_${i}/tile_DR3_1_1.csv',
                'rm /scratch/midway2/raulteixeira/CosmicShearData/tile_${i}/table.h5\">submit',
                '',
                'sbatch submit',
                '',
                'done']
    
    list_lines = [l + '\n' for l in list_lines]

    script = open('/home/raulteixeira/repos/CSPZ/scripts/submit_bpz_batch_%i.sh'%j, mode='w')
    script.writelines(list_lines)  
    script.close()
    sp.run('bash /home/raulteixeira/repos/CSPZ/scripts/submit_bpz_batch_%i.sh'%j, shell = True)


# for i in range(n_batches):
# 	start = i*batch_size
# 	end = (i+1)*batch_size
# 	print('python script START: ', start, '\n', 'python script END: ', end)
# 	list_lines = copy.deepcopy(baselines)
# 	list_lines[2] = list_lines[2].replace('i=0;i<1', f'i={start}'+f';i<{end}')
# 	print(list_lines[2])
# 	#sp.run(f'cp /home/raulteixeira/repos/CSPZ/scripts/submit_bpz_batch.sh /home/raulteixeira/repos/CSPZ/scripts/submit_bpz_batch_{i}.sh', shell = True)
	
# 	script = open(f'/home/raulteixeira/repos/CSPZ/scripts/submit_bpz_batch_{i}.sh', mode='w')
# 	script.writelines(list_lines) 
# 	script.close()
# 	sp.run(f'bash /home/raulteixeira/repos/CSPZ/scripts/submit_bpz_batch_{i}.sh', shell = True)
	
# 	#sp.run(f'rm /home/raulteixeira/repos/CSPZ/scripts/submit_bpz_batch_{i}.sh', shell = True)
