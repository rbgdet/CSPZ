import numpy as np
import pandas as pd
import h5py
from astropy.utils.data import download_file  #import file from URL
import astropy.table
import subprocess as sp
import glob
import sys
import argparse
from astropy.table import Table

#Get all input arguments from command line
#so like my_script.py --Arg1 Val1 --Arg2 Val2 --Flag1 --Flag2
my_parser = argparse.ArgumentParser()

#First some required statements. These have to be passed everytime
my_parser.add_argument('--tilename',  action='store', type = str,   required = True)
my_parser.add_argument('--spectra',   action='store', type = str,   required = True)
my_parser.add_argument('--prior',     action='store', type = str,   required = True)
my_parser.add_argument('--OutPath',   action='store', type = str,   required = True)
#my_parser.add_argument('--mag',       action='store', type = str,   required = True)

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

tile = args['tilename']

bandlist = ['g', 'r', 'i', 'z']
BANDLIST = ['G', 'R', 'I', 'Z']

tables=[]
h5dir = f'/scratch/midway2/raulteixeira/CosmicShearData/tile_{tile}/table.h5'

print(tile, type(tile))
metadata = np.genfromtxt('tile_DR3_1_1.csv', dtype='str', delimiter=",")[1:][int(tile)]

tilename = metadata[0][2:-1]
path = metadata[1][2:-1]
print(path)
p_number = path.split('/')[-1]
for band, BAND in zip(bandlist, BANDLIST):
	fitsdir = f'/scratch/midway2/raulteixeira/CosmicShearData/tile_{tile}/decade.ncsa.illinois.edu/deca_archive/' + path + f'/cat/{tilename}_r5918{p_number}_{band}_cat.fits'

	table_j = Table.read(fitsdir)
	#print(table_j.columns)
	table_j[f'FLUX_AUTO_{BAND}']=table_j['FLUX_AUTO']
	table_j[f'FLUXERR_AUTO_{BAND}']=table_j['FLUXERR_AUTO']
	if band=='g': tables.append(table_j['NUMBER', f'FLUX_AUTO_{BAND}', f'FLUXERR_AUTO_{BAND}'])
	else: tables.append(table_j[f'FLUX_AUTO_{BAND}', f'FLUXERR_AUTO_{BAND}'])
table = astropy.table.hstack(tables)
print(table.columns)
subtable = table['NUMBER', 'FLUX_AUTO_G', 'FLUX_AUTO_R'\
                         , 'FLUX_AUTO_I', 'FLUX_AUTO_Z', 'FLUXERR_AUTO_G', 'FLUXERR_AUTO_R'\
                         , 'FLUXERR_AUTO_I', 'FLUXERR_AUTO_Z']

dframe = pd.DataFrame(data=subtable['NUMBER'], columns = ['NUMBER'])
for label in subtable.columns:
	dframe[label] = subtable[label]

dframe.to_hdf(h5dir, key='df')

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
              'MAG             no'#%s'%args['mag'], 
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

sp.run(f'cp /home/raulteixeira/photoz/code/software/DESC_BPZ/tests/DELVEdata_Flux.pars /home/raulteixeira/photoz/code/software/DESC_BPZ/tests/TEMP_{tile}.pars', shell = True)

pars = open(f'/home/raulteixeira/photoz/code/software/DESC_BPZ/tests/TEMP_{tile}.pars', mode='w')
pars.writelines(list_lines)
pars.close()

catalog_name = f'/home/raulteixeira/scratch-midway2/CosmicShearData/bpztiles/pzinput/pzinput_sgY3_{tile}.h5'

#running bpz using subprocess
sp.run(f'python /home/raulteixeira/repos/DESC_BPZ/scripts/bpz.py ' + catalog_name + ' -P /home/raulteixeira/photoz/code/software/DESC_BPZ/tests/TEMP_{tile}.pars', shell = True)

#Remove the temporary pars file
sp.run(f'rm /home/raulteixeira/photoz/code/software/DESC_BPZ/tests/TEMP_{tile}.pars', shell = True)

print("end")
