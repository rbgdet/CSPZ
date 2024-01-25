import fitsio as fits
import numpy as np
import h5py as h
import treecorr
from astropy.cosmology import FlatLambdaCDM
cosmo = FlatLambdaCDM(H0=70, Om0=0.3, Tcmb0=2.725)
from astropy import units as u

########################################
#reading all catalogs                                                                                                          
metacal_cat =  h.File('/project/chihway/data/decade/metacal_gold_combined_20231212.hdf','r')
mask_cat = h.File('/project/chihway/data/decade/metacal_gold_combined_mask_20231212.hdf','r')
binning_cat = h.File('/project/chihway/raulteixeira/data/DR3_1_ID+TomoBin.hdf5','r')

boss_cat_file = '/project/chihway/data/decade/BOSS_eBOSS.fits'
boss_random_file = '/project/chihway/data/decade/BOSS_eBOSS_rnd.fits'
boss_cat = fits.read(boss_cat_file) #get Z, RA, DEC from here
boss_random = fits.read(boss_random_file)

#checking metacal matches the binning file:                                                                                   
noshear_mask = mask_cat['baseline_mcal_mask_noshear'][:]
metacal_ids = metacal_cat['id'][noshear_mask>0]
tomo_ids = binning_cat['df']['block0_values'][:,0].astype(int)
assert np.all(metacal_ids==tomo_ids)

#since the above is fine, get the tomographic bins:                                                                           
tomo_bins = binning_cat['df']['block0_values'][:,1].astype(int)
assert np.all(np.unique(tomo_bins)==np.arange(4)) #check there are 4 bins 0-indexed

#the IDs and the ordering is correcty between the metacal catalog and the binning catalog
#now check that the actual bins are the same:
assert np.all( noshear_mask[noshear_mask>0] == tomo_bins+1 )

############
#the meat of the calculation: for every tomo bin, split the BOSS & random sample 
#into slices of 0.025 in redshift and cross-correlate
nbins = 20
bin_slop=0.01
bin_edges = np.linspace(0.1,1.1,41) #slices of 0.025 in redshift

def angle_min_max(redshift): #will use this to minimum and maximum theta over which treecorr will run
        physical_min, physical_max = 1.5*u.Mpc, 5.0*u.Mpc #in Mpc
        physical_min = 0.9*physical_min #simply enlarging the bottom and top bins a bit
        physical_max = 1.1*physical_max 

        theta_min = physical_min/cosmo.angular_diameter_distance(redshift)
        theta_min = theta_min*u.rad
        theta_min = theta_min.to(u.degree)

        theta_max = physical_max/cosmo.angular_diameter_distance(redshift)
        theta_max = theta_max*u.rad
        theta_max = theta_max.to(u.degree)

        return theta_min.value, theta_max.value #in degrees
        
for Bin in [3,4]:#np.arange(1,5): #will use the 1-indexed bin assignments in noshear_mask since they match the binning catalog 
        
        thisbin = noshear_mask==Bin
        print(f'For bin {Bin}, will cross-correlate {sum(thisbin)} galaxies')
        metacal_ra_thisbin = metacal_cat['RA'][:][thisbin]	
        metacal_dec_thisbin = metacal_cat['DEC'][:][thisbin]

        for slice_lower_edge,slice_upper_edge in zip(bin_edges[:-1],bin_edges[1:]):

                Z_middle = 0.5*(slice_upper_edge+slice_lower_edge)
                min_theta,max_theta = angle_min_max(Z_middle) #gets the minimum and maximum theta from the mean redshift of the thin slice

                #get the galaxies in the boss random and actual catalogs that sit within the thin redshift slice:
                boss_slicing = np.where( (boss_cat['Z']>=slice_lower_edge) * (boss_cat['Z']<slice_upper_edge))[0]
                boss_random_slicing = np.where( (boss_random['Z']>=slice_lower_edge) * (boss_random['Z']<slice_upper_edge))[0]

                boss_ra = boss_cat['RA'][boss_slicing]
                boss_dec = boss_cat['DEC'][boss_slicing]
                boss_w = boss_cat['WEIGHT_FKP'][boss_slicing] #check which weight to use!
                NotNan = ~np.isnan(boss_w) #some boss galaxies have NaN FKP weights, get those out
                boss_ra = boss_ra[NotNan]
                boss_dec = boss_dec[NotNan]
                boss_w = boss_w[NotNan]
                
                boss_random_ra = boss_random['RA'][boss_random_slicing]
                boss_random_dec = boss_random['DEC'][boss_random_slicing]
                boss_random_w = boss_random['WEIGHT_FKP'][boss_random_slicing]
                
                print(len(boss_random_slicing),'random boss galaxies and ',len(boss_slicing),'real galaxies')
                
                DD = treecorr.NNCorrelation(min_sep=min_theta, max_sep=max_theta,nbins=nbins,sep_units='degrees',bin_slop=bin_slop)
                DR = treecorr.NNCorrelation(min_sep=min_theta, max_sep=max_theta,nbins=nbins,sep_units='degrees',bin_slop=bin_slop)
        
                unknown_cat = treecorr.Catalog(ra=metacal_ra_thisbin,dec=metacal_dec_thisbin,ra_units='deg',dec_units='deg')
                reference_cat = treecorr.Catalog(ra=boss_ra, dec=boss_dec,w=boss_w,ra_units='deg',dec_units='deg')
                random_cat = treecorr.Catalog(ra=boss_random_ra, dec=boss_random_dec, w=boss_random_w, ra_units='deg',dec_units='deg')

                DD.process(unknown_cat,reference_cat)
                DR.process(unknown_cat,random_cat)

                
                outname = 'outputs/output_1.5to5.0Mpc_bslop%1.3f_ref%1.3fz%1.3f_metacalbin%d.txt'%(bin_slop,slice_lower_edge,slice_upper_edge,Bin)
                print('Writing %s (measured between %1.3f and %1.3f deg)'%(outname,min_theta,max_theta))
                DD.write(outname,dr=DR,rr=DR) #including RR simply because treecorr wants it to be there for writing

                del DD, DR, unknown_cat, reference_cat, random_cat



