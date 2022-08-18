import numpy as np
import pandas as pd
import h5py
from astropy.utils.data import download_file  #import file from URL
import astropy.table
import subprocess as sp
import glob
import sys

tile = 'DES1156-3706'

list_lines = ['COLUMNS     /home/raulteixeira/repos/DESC_BPZ/tests/CosmicShearPZ.columns\n'
              , f'OUTPUT\t    /home/raulteixeira/scratch-midway2/CosmicShearData/bpztiles/output/pzs/pz_test_{tile}.h5\n'
              , 'SPECTRA     CWWSB4.list\n', 'PRIOR\t  hdfn_gen\n', 'DZ          0.01\n', 'ZMIN        0.005\n'
              , 'ZMAX        3.505\n', 'MAG         yes\n', 'NEW_AB      no\n', 'MADAU\t    no #TURN OFF MADAU!!!!\n'
              , 'EXCLUDE     none\n', 'CHECK       yes\n', '#ZC          1.0,2.0\n', '#FC          0.2,0.4\n'
              , 'VERBOSE     no\n', '#INTERP      0\n', 'ODDS        0.68\n'
              , f'PROBS      /home/raulteixeira/scratch-midway2/CosmicShearData/bpztiles/output/probs/pz_test_probs_{tile}\n'
              , f'PROBS2     no\n'
              , f'PROBS_LITE no\n', 'GET_Z       yes\n', 'INTERACTIVE yes\n', 'PLOTS       no\n'
              , 'SAMPLING yes\n', 'NSAMPLES 1\n', 'SEED 42\n'
              , '#ONLY_TYPE   yes\n']
pars = open('/home/raulteixeira/repos/DESC_BPZ/tests/DELVEdata_example.pars', mode='w')
pars.writelines(list_lines)
pars.close()

print("pars.close()")

catalog_name = f'/home/raulteixeira/scratch-midway2/CosmicShearData/bpztiles/pzinput/pzinput_METACAL_4_bands_{tile}.h5'
#running bpz using subprocess
sp.run(['python', '/home/raulteixeira/repos/DESC_BPZ/scripts/bpz.py', catalog_name, '-P', '/home/raulteixeira/repos/DESC_BPZ/tests/DELVEdata_example.pars'], capture_output=False)

print("end")
