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

list_lines = ['COLUMNS         %s'%column_path,
             f'OUTPUT          %s'%output_path,
              'SPECTRA         %s'%args['spectra'], #eg CWWSB4.list 
              'PRIOR           %s'%args['prior'], #eg hdfn_gen
              'DZ              %s'%args['dz'], 
              'ZMIN            %s'%args['zmin'],
              'ZMAX            %s'%args['zmax'], 
              'MAG             yes', 
              'NEW_AB          no', 
              'MADAU           no #TURN OFF MADAU!!!!',
              'EXCLUDE         none', 
              'CHECK           yes', 
              '#ZC             1.0,2.0', 
              '#FC             0.2,0.4',
              'VERBOSE         no', 
              '#INTERP         0', 
              'ODDS            0.68',
             f'PROBS           %s'%PROB_path,
             f'PROBS2          no',
             f'PROBS_LITE      no', 
              'GET_Z           yes', 
              'INTERACTIVE     yes', 
              'PLOTS           no',
              'SAMPLING        yes',
              'NSAMPLES        %d'%args['Nsamples'],
              '#ONLY_TYPE      yes']

#I removed all the "\n" from each line purely for visual considerations.
#Now adding them back in with a single list comprehension line.
list_lines = [l + '\n' for l in list_lines]

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
