import numpy as np
import pandas as pd
import h5py
from astropy.utils.data import download_file  #import file from URL
import astropy.table
import subprocess as sp
import glob
import sys
import argparse

#Get all input arguments from command line
#so like my_script.py --Arg1 Val1 --Arg2 Val2 --Flag1 --Flag2
my_parser = argparse.ArgumentParser()

#First some required statements. These have to be passed everytime
my_parser.add_argument('--tilename',  action='store', type = str,   required = True)
my_parser.add_argument('--spectra',   action='store', type = str,   required = True)
my_parser.add_argument('--prior',     action='store', type = str,   required = True)
my_parser.add_argument('--OutPath',   action='store', type = str,   required = True)

#Optional ones. Default values given and can be varied if needed.
my_parser.add_argument('--dz',        action='store', type = float, default = 0.01)
my_parser.add_argument('--zmin',      action='store', type = float, default = 0.005)
my_parser.add_argument('--zmax',      action='store', type = float, default = 3.505)
my_parser.add_argument('--Nsamples',  action='store', type = int,   default = 1)

#Print input args to help any future debugging
print('-------INPUT PARAMS----------')
for p in args.keys():
    print('%s : %s'%(p.upper(), args[p]))
print('-----------------------------')
print('-----------------------------')

#Parse everything
args = vars(my_parser.parse_args())


tile = args['tilename']


column_path =  '/home/raulteixeira/photoz/code/software/DESC_BPZ/tests/CosmicShearPZ.columns'
PROB_path   = f'/home/raulteixeira/scratch-midway2/CosmicShearData/bpztiles/output/probs/PZ_OUTPUT_sgY3_probs_{tile}_finer_test'

#Better to pass output dir as argument to script.
#If you want to run 10 different versions of same tile, you should have a way to
#easily output them to 10 different filenames/directories from this script
output_path = args['OutPath'] #f'/home/raulteixeira/scratch-midway2/CosmicShearData/bpztiles/output/pzs/pz_sgY3_{tile}_finer_test.h5'

list_lines = ['COLUMNS         %s\n'%column_path,
             f'OUTPUT          %s\n'%output_path,
              'SPECTRA         %s\n'%args['spectra'], #eg CWWSB4.list 
              'PRIOR           %s\n'%args['prior'], #eg hdfn_gen
              'DZ              %s\n'%args['dz'], 
              'ZMIN            %s\n'%args['zmin'],
              'ZMAX            %s\n'%args['zmax'], 
              'MAG             yes\n', 
              'NEW_AB          no\n', 
              'MADAU           no #TURN OFF MADAU!!!!\n',
              'EXCLUDE         none\n', 
              'CHECK           yes\n', 
              '#ZC             1.0,2.0\n', 
              '#FC             0.2,0.4\n',
              'VERBOSE         no\n', 
              '#INTERP         0\n', 
              'ODDS            0.68\n',
             f'PROBS           %s\n'%PROB_path,
             f'PROBS2          no\n',
             f'PROBS_LITE      no\n', 
              'GET_Z           yes\n', 
              'INTERACTIVE     yes\n', 
              'PLOTS           no\n',
              'SAMPLING        yes\n',
              'NSAMPLES        %d\n'%args['Nsamples'],
              '#ONLY_TYPE      yes\n']

#Can't modify the example.pars directly.
#If we run in parallel then multiple instances of script
#will be rewriting same file at once and that won't work. 
#So make temporary copy first.

sp.run(f'cp /home/raulteixeira/photoz/code/software/DESC_BPZ/tests/DELVEdata_example.pars /home/raulteixeira/photoz/code/software/DESC_BPZ/tests/TEMP_{tile}.pars', shell = True)

pars = open(f'/home/raulteixeira/photoz/code/software/DESC_BPZ/tests/TEMP_{tile}.pars', mode='w')
pars.writelines(list_lines)
pars.close()

catalog_name = f'/home/raulteixeira/scratch-midway2/CosmicShearData/bpztiles/pzinput/pzinput_sgY3_{tile}.h5'
#running bpz using subprocess
sp.run(f'python /home/raulteixeira/repos/DESC_BPZ/scripts/bpz.py ' + catalog_name + ' -P /home/raulteixeira/photoz/code/software/DESC_BPZ/tests/TEMP_{tile}.pars', shell = True)

#Remove the temporary pars file
sp.run(f'rm /home/raulteixeira/photoz/code/software/DESC_BPZ/tests/TEMP_{tile}.pars', shell = True)

print("end")
