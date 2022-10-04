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
    def __init__(self, catname, idcol, bands, band_errs, groupcol):
        self.cat = Table.read(catname)
        self.idcol= idcol
        self.bands = bands
        self.band_errs = band_errs
        self.groupcol = groupcol #can be tile or healpix pixel; should be STRING with tile/pixel column name
    
    def h5gen(self, h5dir='~/HDF5FileBPZ.h5', tiling=False):
        columns = [self.idcol, self.groupcol] + self.bands + self.band_errs #list of columns
        if tiling:
            for group in np.unique(self.cat[groupcol]):
                masktile=self.catalog['TILENAME']==group
                table_i = self.cat[masktile][columns] #masked table
                dframe_i = table_i.to_pandas() #converting to DF
                        
            dframe_i.to_hdf(h5dir.split('.')[0]+tile+'.h5', key='df')
        else:
            table = self.cat[columns]
            dframe = table.to_pandas()

            dframe.to_hdf(h5dir.split('.')[0]+tile+'.h5', key='df')


    def columns(self):
        '''
        generates .columns file
        '''
        
    def pars(self):
        '''
        generates .pars file
        '''
        
    def bpzrun(self, cut=True, cuts=None):
    
    def merge(self):
    
    def cuts(self, cuts=None, idcutfname=None):
        '''
        cuts='Y3', 'Y6', 'idcut' --- 'idcut' can be of any kind, need to provide file with ids
        '''
        if cuts==None: pass
        else:
            
    
    def stats(self):
    
            
            
            
            
            
            
            
            
            
            
            
            
            
        