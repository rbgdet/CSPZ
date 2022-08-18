#WIP

#imports

import numpy as np
import pandas as pd
import h5py
import astropy.table
from astropy.table import Table
import astropy.io.fits as pf
#import matplotlib.pyplot as plt



class BPZRun:
    '''
    This is a class to run bpz from a fits catalog.
    Inputs: 
    - FITS catalog
    - List of strings with photometric band (and errors) column names


    Outputs:
    - FITS catalog with BPZ estimated redshifts and other BPZ outputs
        See github.com/LSSTDESC/DESC_BPZ and github.com/rbgdet/DESC_BPZ 
    '''
    def __init__(self, catalog, bands, band_errs):
        self.catalog = catalog
        self.bands = bands
        self.band_errs = band_errs
    
    def 