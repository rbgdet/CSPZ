import sys
if sys.version_info[0] < 3:
    raise Exception("Must be using Python 3")
    
    
import numpy as np
import pandas as pd
import pickle
import time
import os

from datetime import datetime
today = datetime.today()
today = today.strftime('%B%d')
outpath = '/project/chihway/raulteixeira/data/'

from multiprocessing import Pool

### This will produce Nsamples X 64 samples
Nsamples = int(2e3)
#data_dir = '/global/cscratch1/sd/aamon/sompz_data/v0.50/'

#redshift_sample_dir = data_dir
#redshift_file_name = 'redshift_deep_balrog_incl_cell_assignment'
#redshift_sample_dir = '/global/cscratch1/sd/aamon/sompz_data/zsamples/'
#redshift_file_name = 'PhOpt_PAU_C30elsewhere_deep_balrog'
#redshift_file_name = 'opt_spec_C30elsewhere_deep_balrog'
redshift_file_name =  'optPrime_spec_PAU_C30elsewhere_deep_balrog'


# out_path = f'/global/cscratch1/sd/alexalar/desy3data/Nz_samples/v0.50/3sdir_fid_zsamples_test1/{redshift_file_name}/'

# if not os.path.exists(out_path):
#     os.makedirs(out_path)



### Comment this line if you don't want to save the summary h5 file.
save_h5 = f'/project/chihway/raulteixeira/data/SOMPZ_{redshift_file_name}_{today}.h5'
    
#####################################
### Load catalogs and essential matrices.
#####################################
    
### Balrog files ###
# balrog_file= data_dir + 'deep_balrog_incl_cell_assignment_incw.pkl'
# balrog_file2= data_dir + 'deep_balrog_incl_cell_assignment_incw2.pkl'

# balrog_data1= pickle.load(open(balrog_file, 'rb'), encoding='latin1')
# balrog_data2= pickle.load(open(balrog_file2, 'rb'), encoding='latin1')
# balrog_data=pd.concat([balrog_data1, balrog_data2], ignore_index=True)
# ## This computes the lensingXresponse weight for each galaxy, removing the Balrog injection rate.
# balrog_data['weight_response_shear'] = balrog_data['injection_counts']*balrog_data['overlap_weight']


# spec_file= redshift_sample_dir +'%s.pkl'%redshift_file_name
# spec_file2= redshift_sample_dir +'%s2.pkl'%redshift_file_name
# spec_data1= pickle.load(open(spec_file, 'rb'), encoding='latin1')
# spec_data2= pickle.load(open(spec_file2, 'rb'), encoding='latin1')
# spec_data=pd.concat([spec_data1, spec_data2], ignore_index=True)

## Add the overlap_weight to the redshift sample

# needed_columns = ['overlap_weight','cell_deep', 'cell_wide_unsheared']
# needed_columns = [x for x in needed_columns if x not in spec_data.columns.values]

# spec_data = spec_data.merge(balrog_data[['bal_id']+needed_columns], on='bal_id')
spec_data = pd.read_hdf('%s/DES_DF_baldet_121923_64x64_cells_with_redshifts_colnames4uncertainties_corrected.hdf'%outpath, key='df') #is this the catalog with just objects that have redshifts?? - Raul
spec_data['cell_deep']=spec_data['cell_deep'].astype(int)
spec_data['cell_wide_unsheared']=spec_data['cell_wide_unsheared'].astype(int)
spec_data['overlap_weight']=np.ones_like(spec_data['overlap_weight'])

# ## This computes the lensingXresponse weight for each galaxy, removing the Balrog injection rate.
spec_data['weight_response_shear'] = spec_data['injection_counts']*spec_data['overlap_weight']

### Load dictionary containing which wide cells belong to which tomographic bin
tomo_bins_wide_modal_even = pickle.load(open('%s/tomo_bins_wide_cells.pickle'%outpath, 'rb'), encoding='latin1')

### Load p(chat) with all weights included: Balrog, response, shear.
#pchat = np.load(data_dir+'pchat_modal_even.npy')
pchat = np.load('%s/p_c_hat_bal_01092023.npz'%outpath)['p_c_hat_bal'].flatten() #here I don't have response nor shear weights

### Load p(c|chat) with all weights included: Balrog, response, shear.
pc_chat = np.load('%s/p_cchat_01092023.npz'%outpath)['p_cchat'].T
#COMMENTED OUT UNTIL WE GET SHEAR WEIGHTS
# pcchat = np.zeros_like(pc_chat)
# np.add.at(pcchat, 
#           (spec_data.cell_deep.values.astype(int),spec_data.cell_wide_unsheared.values.astype(int)),
#           spec_data.overlap_weight.values)
# #pc_chat = pcchat/np.sum(pcchat,axis=0)
# pc_chat_new = pcchat/np.sum(pcchat,axis=0)
#
#assert np.allclose(pc_chat_new,pc_chat)

### Define the redshift binning. This is currently set by the sample variance.

min_z   = 0.01
max_z   = 5
delta_z = 0.05
zbins   = np.arange(min_z,max_z+delta_z,delta_z)
zbinsc  = zbins[:-1]+(zbins[1]-zbins[0])/2.


#####################################
### compute N(z,c) and N(c), R(z,c), R(c) 
### and bin conditionalization versions.
#####################################

def return_Nzc(df):
    """
    - This function returns the counts Nzc=N(z,c) in each bin z and cell c.
    - The input is a pandas Dataframe containing a redshift sample. 
    - The redshift sample must have redshift and deep cell assignment.
    - It computes the balrog probability defined as #detections/#injections 
    to weight the counts of each galaxy in N(z,c).
    """

    redshift_sample = df[['injection_counts','true_id','cell_deep', 'Z']].groupby('true_id').agg('mean').reset_index()
    unique_id, unique_counts = np.unique(df.true_id.values, return_counts=True)
    redshift_sample = redshift_sample.merge(pd.DataFrame({'true_id':unique_id, 'unique_counts':unique_counts}),on='true_id')
    redshift_sample['balrog_prob'] = redshift_sample['unique_counts']/redshift_sample['injection_counts']
    zid = np.digitize(redshift_sample.Z.values, zbins)-1
    zid = np.clip(zid, 0, len(zbinsc)-1)
    redshift_sample['zid'] = zid
    redshift_sample_groupby = redshift_sample[['balrog_prob','zid','cell_deep']].groupby(['zid','cell_deep']).agg('sum')

    Nzc = np.zeros((len(zbins)-1,64*64))
    for index, row in redshift_sample_groupby.iterrows():
        if (index[0]<0)|(index[0]>len(zbins)-1): continue
        Nzc[int(index[0]),int(index[1])] = row.balrog_prob
    return Nzc

def return_Nc(df):
    """
    - This function returns the counts Nc=N(c) in each cell c.
    - The input is a pandas Dataframe containing a deep sample. 
    - The deep sample must have a deep cell assignment.
    - It computes the balrog probability defined as #detections/#injections 
    to weight the counts of each galaxy in N(c).
    """

    redshift_sample = df[['injection_counts','true_id','cell_deep']].groupby('true_id').agg('mean').reset_index()
    unique_id, unique_counts = np.unique(df.true_id.values, return_counts=True)
    redshift_sample = redshift_sample.merge(pd.DataFrame({'true_id':unique_id, 'unique_counts':unique_counts}),on='true_id')
    redshift_sample['balrog_prob'] = redshift_sample['unique_counts']/redshift_sample['injection_counts']
    redshift_sample_groupby = redshift_sample[['balrog_prob','cell_deep']].groupby(['cell_deep']).agg('sum')

    Nc = np.zeros((64*64))
    for index, row in redshift_sample_groupby.iterrows():
        Nc[int(index)] = row.balrog_prob
    return Nc

def return_Rzc(df):
    """
    - This function returns the average lensingXshear weight in each bin z and cell c, Rzc= <ResponseXshear>(z,c)
    - The average is weighted by the balrog probability of each galaxy, defined as #detections/#injections.
    """
    redshift_sample = df[['injection_counts','true_id','cell_deep', 'Z', 'weight_response_shear','overlap_weight']].groupby('true_id').agg('mean').reset_index()
    unique_id, unique_counts = np.unique(df.true_id.values, return_counts=True)
    redshift_sample = redshift_sample.merge(pd.DataFrame({'true_id':unique_id, 'unique_counts':unique_counts}),on='true_id')
    redshift_sample['balrog_prob'] = redshift_sample['unique_counts']/redshift_sample['injection_counts']
    zid = np.digitize(redshift_sample.Z.values, zbins)-1
    zid = np.clip(zid, 0, len(zbinsc)-1)
    redshift_sample['zid'] = zid
    redshift_sample['weight_response_shear_balrogprob'] = redshift_sample['weight_response_shear']*redshift_sample['balrog_prob']


    redshift_sample_groupby = redshift_sample[['weight_response_shear_balrogprob', 'balrog_prob','zid','cell_deep']].groupby(['zid','cell_deep']).agg('sum')
    Rzc = np.zeros((len(zbins)-1,64*64))
    for index, row in redshift_sample_groupby.iterrows():
        if (index[0]<0)|(index[0]>len(zbins)-1): continue
        Rzc[int(index[0]),int(index[1])] = row.weight_response_shear_balrogprob/row.balrog_prob
    return Rzc

def return_Rc(df):
    """
    - This function returns the average lensingXshear weight in each cell c, Rc= <ResponseXshear>(c)
    - The average is weighted by the balrog probability of each galaxy, defined as #detections/#injections.
    """
    redshift_sample = df[['injection_counts','true_id','cell_deep', 'weight_response_shear','overlap_weight']].groupby('true_id').agg('mean').reset_index()
    unique_id, unique_counts = np.unique(df.true_id.values, return_counts=True)
    redshift_sample = redshift_sample.merge(pd.DataFrame({'true_id':unique_id, 'unique_counts':unique_counts}),on='true_id')
    redshift_sample['balrog_prob'] = redshift_sample['unique_counts']/redshift_sample['injection_counts']
    redshift_sample['weight_response_shear_balrogprob'] = redshift_sample['weight_response_shear']*redshift_sample['balrog_prob']


    redshift_sample_groupby = redshift_sample[['weight_response_shear_balrogprob', 'balrog_prob','cell_deep']].groupby(['cell_deep']).agg('sum')
    Rc = np.zeros(64*64)
    for index, row in redshift_sample_groupby.iterrows():
        Rc[int(index)] = row.weight_response_shear_balrogprob/row.balrog_prob
    return Rc


### Counts in the redshift sample (weighted by balrog, but not weighted by responseXlensing weights.)
### Including condition on tomographic bin.
Nzc = return_Nzc(spec_data)
Nzc_0 = return_Nzc(spec_data[spec_data.cell_wide_unsheared.isin(tomo_bins_wide_modal_even[0])])
Nzc_1 = return_Nzc(spec_data[spec_data.cell_wide_unsheared.isin(tomo_bins_wide_modal_even[1])])
Nzc_2 = return_Nzc(spec_data[spec_data.cell_wide_unsheared.isin(tomo_bins_wide_modal_even[2])])
Nzc_3 = return_Nzc(spec_data[spec_data.cell_wide_unsheared.isin(tomo_bins_wide_modal_even[3])])

### Counts in the deep sample (weighted by balrog, but not weighted by responseXlensing weights.)
### Including condition on tomographic bin.
Nc = return_Nc(spec_data)
Nc_0 = return_Nc(spec_data[spec_data.cell_wide_unsheared.isin(tomo_bins_wide_modal_even[0])])
Nc_1 = return_Nc(spec_data[spec_data.cell_wide_unsheared.isin(tomo_bins_wide_modal_even[1])])
Nc_2 = return_Nc(spec_data[spec_data.cell_wide_unsheared.isin(tomo_bins_wide_modal_even[2])])
Nc_3 = return_Nc(spec_data[spec_data.cell_wide_unsheared.isin(tomo_bins_wide_modal_even[3])])

### If after the bin condition there are no redshift counts in a deep cell, don't apply the bin condition in that deep cell.
sel_0 = ((np.sum(Nzc, axis=0)>0) & (np.sum(Nzc_0, axis=0)==0))
sel_1 = ((np.sum(Nzc, axis=0)>0) & (np.sum(Nzc_1, axis=0)==0))
sel_2 = ((np.sum(Nzc, axis=0)>0) & (np.sum(Nzc_2, axis=0)==0))
sel_3 = ((np.sum(Nzc, axis=0)>0) & (np.sum(Nzc_3, axis=0)==0))
Nzc_0[:,sel_0] = Nzc[:,sel_0].copy()
Nzc_1[:,sel_1] = Nzc[:,sel_1].copy()
Nzc_2[:,sel_2] = Nzc[:,sel_2].copy()
Nzc_3[:,sel_3] = Nzc[:,sel_3].copy()

### Average responseXlensing in each deep cell and redshift bin. The responseXlensing of each galaxy is weighted by its balrog probability.
### Including condition on tomographic bin.

Rzc = return_Rzc(spec_data)
Rzc_0 = return_Rzc(spec_data[spec_data.cell_wide_unsheared.isin(tomo_bins_wide_modal_even[0])])
Rzc_1 = return_Rzc(spec_data[spec_data.cell_wide_unsheared.isin(tomo_bins_wide_modal_even[1])])
Rzc_2 = return_Rzc(spec_data[spec_data.cell_wide_unsheared.isin(tomo_bins_wide_modal_even[2])])
Rzc_3 = return_Rzc(spec_data[spec_data.cell_wide_unsheared.isin(tomo_bins_wide_modal_even[3])])


### Average responseXlensing in each deep cell in the REDSHIFT sample. The responseXlensing of each galaxy is weighted by its balrog probability.
### Including condition on tomographic bin.
Rc_redshift = return_Rc(spec_data)
Rc_0_redshift = return_Rc(spec_data[spec_data.cell_wide_unsheared.isin(tomo_bins_wide_modal_even[0])])
Rc_1_redshift = return_Rc(spec_data[spec_data.cell_wide_unsheared.isin(tomo_bins_wide_modal_even[1])])
Rc_2_redshift = return_Rc(spec_data[spec_data.cell_wide_unsheared.isin(tomo_bins_wide_modal_even[2])])
Rc_3_redshift = return_Rc(spec_data[spec_data.cell_wide_unsheared.isin(tomo_bins_wide_modal_even[3])])

### Average responseXlensing in each deep cell in the DEEP sample. The responseXlensing of each galaxy is weighted by its balrog probability.
### Including condition on tomographic bin.
Rc_deep = return_Rc(spec_data)
Rc_0_deep = return_Rc(spec_data[spec_data.cell_wide_unsheared.isin(tomo_bins_wide_modal_even[0])])
Rc_1_deep = return_Rc(spec_data[spec_data.cell_wide_unsheared.isin(tomo_bins_wide_modal_even[1])])
Rc_2_deep = return_Rc(spec_data[spec_data.cell_wide_unsheared.isin(tomo_bins_wide_modal_even[2])])
Rc_3_deep = return_Rc(spec_data[spec_data.cell_wide_unsheared.isin(tomo_bins_wide_modal_even[3])])

### We do not need the balrog and redshift samples. We can delete them.
del spec_data


def return_bincondition_fraction_Nzt_redshiftsample(redshift_sample_Nzt):
    """This function returns the fraction of counts in Nzt with over 
    without bin condition, for each tomographic bin.
    """
    pz_c_0 = redshift_sample_Nzt[0]/np.sum(redshift_sample_Nzt[0],axis=0)
    pz_c_1 = redshift_sample_Nzt[1]/np.sum(redshift_sample_Nzt[1],axis=0)
    pz_c_2 = redshift_sample_Nzt[2]/np.sum(redshift_sample_Nzt[2],axis=0)
    pz_c_3 = redshift_sample_Nzt[3]/np.sum(redshift_sample_Nzt[3],axis=0)
    pz_c_4 = redshift_sample_Nzt[4]/np.sum(redshift_sample_Nzt[4],axis=0)
    
    pz_c_0[~np.isfinite(pz_c_0)] = 0
    pz_c_1[~np.isfinite(pz_c_1)] = 0
    pz_c_2[~np.isfinite(pz_c_2)] = 0
    pz_c_3[~np.isfinite(pz_c_3)] = 0
    pz_c_4[~np.isfinite(pz_c_4)] = 0
    
    gzt_0 = pz_c_1/pz_c_0
    gzt_1 = pz_c_2/pz_c_0
    gzt_2 = pz_c_3/pz_c_0
    gzt_3 = pz_c_4/pz_c_0


    gzt_0[~np.isfinite(gzt_0)] = 0
    gzt_1[~np.isfinite(gzt_1)] = 0
    gzt_2[~np.isfinite(gzt_2)] = 0
    gzt_3[~np.isfinite(gzt_3)] = 0
    
    return np.array([gzt_0, gzt_1, gzt_2, gzt_3])


def return_bincondition_fraction_Nt_deepsample(deep_sample_Nt):
    """This function returns the fraction of counts in Nt with over 
    without bin condition, for each tomographic bin. Deep sample.
    """    
    gt_0 = deep_sample_Nt[1]/deep_sample_Nt[0]
    gt_1 = deep_sample_Nt[2]/deep_sample_Nt[0]
    gt_2 = deep_sample_Nt[3]/deep_sample_Nt[0]
    gt_3 = deep_sample_Nt[4]/deep_sample_Nt[0]

    gt_0[~np.isfinite(gt_0)] = 0
    gt_1[~np.isfinite(gt_1)] = 0
    gt_2[~np.isfinite(gt_2)] = 0
    gt_3[~np.isfinite(gt_3)] = 0

    return np.array([gt_0, gt_1, gt_2, gt_3])


fraction_Nzt = return_bincondition_fraction_Nzt_redshiftsample(np.array([Nzc, Nzc_0, Nzc_1, Nzc_2, Nzc_3]))
fraction_Nt_D = return_bincondition_fraction_Nt_deepsample(np.array([Nc, Nc_0, Nc_1, Nc_2, Nc_3]))
bincond_combined = fraction_Nzt*fraction_Nt_D[:,None,:]


def return_bincondition_weight_Rzt_combined(redshift_sample_Rzt, redshift_sample_Rt, deep_sample_Rt):
    """This function returns the final average responseXshear weight in each deep cell and redshift bin: Rzc.
    Response weight = Response to shear of the balrog injection of a deep galaxy.
    Shear weight = Weight to optimize of signal to noise of some shear observable. 
    - final Rzt = <Rzt>r * <Rt>r / <Rt>d
    where: 
    - <Rzt>r: average weight in z,c in the redshift sample.
    - <Rt>r: average weight in c in the redshift sample.
    - <Rt>d: average weight in c in the deep sample.
    It basically rescales the weight in Rzt such that it matches the average weight according to the deep sample.
    """    
    Rzt_factor_0 = deep_sample_Rt[1]/redshift_sample_Rt[1]
    Rzt_factor_1 = deep_sample_Rt[2]/redshift_sample_Rt[2]
    Rzt_factor_2 = deep_sample_Rt[3]/redshift_sample_Rt[3]
    Rzt_factor_3 = deep_sample_Rt[4]/redshift_sample_Rt[4]

    Rzt_factor_0[~np.isfinite(Rzt_factor_0)] = 0
    Rzt_factor_1[~np.isfinite(Rzt_factor_1)] = 0
    Rzt_factor_2[~np.isfinite(Rzt_factor_2)] = 0
    Rzt_factor_3[~np.isfinite(Rzt_factor_3)] = 0

    Rzt_0_final = np.einsum('zt,t->zt', redshift_sample_Rzt[1], Rzt_factor_0)
    Rzt_1_final = np.einsum('zt,t->zt', redshift_sample_Rzt[2], Rzt_factor_1)
    Rzt_2_final = np.einsum('zt,t->zt', redshift_sample_Rzt[3], Rzt_factor_2)
    Rzt_3_final = np.einsum('zt,t->zt', redshift_sample_Rzt[4], Rzt_factor_3)
    return np.array([Rzt_0_final, Rzt_1_final, Rzt_2_final, Rzt_3_final])

redshift_sample_Rzt = np.array([Rzc, Rzc_0, Rzc_1, Rzc_2, Rzc_3])
redshift_sample_Rt = np.array([Rc_redshift, Rc_0_redshift, Rc_1_redshift, Rc_2_redshift, Rc_3_redshift])
deep_sample_Rt = np.array([Rc_deep, Rc_0_deep, Rc_1_deep, Rc_2_deep, Rc_3_deep])
Rt_combined = return_bincondition_weight_Rzt_combined(redshift_sample_Rzt, redshift_sample_Rt, deep_sample_Rt)


#####################################
### Load Sample Variance from theory. 
### Compute superphenotypes and N(T,c,Z) matrices.
#####################################

### Load the sample variance theory ingredient. This estimates the ratio between Shot noise and sample variance.

#sv_th = np.load('/global/cscratch1/sd/alexalar/desy3data/cosmos_sample_variance.npy')[0]
sv_th = np.load('/project/chihway/dhayaa/DECADE/Alex_NERSC_files/cosmos_sample_variance.npy')[0]
sv_th = np.diagonal(sv_th)[:]
sv_th = sv_th[:len(zbinsc)]
assert sv_th.shape[0]==len(zbinsc)
#sv_th_new = np.load('/global/cscratch1/sd/alexalar/desy3data/marco_sv_v2/sample_variance.npy')
sv_th_new = np.load('/project/chihway/dhayaa/DECADE/Alex_NERSC_files/sample_variance.npy')
sv_th_new_diag = np.array([np.diagonal(x) for x in sv_th_new])

sv_th_new_final = np.linalg.pinv(np.sum(np.array([np.linalg.pinv(x) for x in sv_th_new]),axis=0))
sv_th_new_final_diag = np.diagonal(sv_th_new_final)

sv_th_new_diag = sv_th_new_diag[:,:len(zbinsc)]
sv_th_new_final_diag = sv_th_new_final_diag[:len(zbinsc)]

nts = Nc.copy()
nzt = Nzc.copy()
nz,nt = nzt.shape

# Removing types that don't have galaxies
maskt = (np.sum(nzt,axis=0)>0.)
nts = nts[maskt]
nzt = nzt[:,maskt]

# What is the redshift of each type?
# Computing the mean redshift per type
zmeant = np.zeros(nzt.shape[1])
for i in range(nzt.shape[1]):
    zmeant[i] = np.average(np.arange(len(zbinsc)),weights=nzt.T[i])
zmeant = np.rint(zmeant)

#sv_th_v2 = np.load('/global/cscratch1/sd/alexalar/desy3data/marco_sv_v2/sv_th_v2.npy')

varn_th = 1 + np.sum(nzt,axis=1)*sv_th
#varn_th_deep_v2 = 1 + np.sum(nzt/np.sum(nzt,axis=0) * nts,axis=1)*sv_th_v2
varn_th_deep_v2 = 1 + np.sum(nzt/np.sum(nzt,axis=0) * nts,axis=1)*sv_th_new_final_diag


def make_nzT(nzti, njoin, plot=False):
    zmeanti = np.zeros(nzti.shape[1])
    for i in range(nzti.shape[1]):
        try: zmeanti[i] = np.average(np.arange(len(zbinsc)),weights=nzti.T[i])
        except: zmeanti[i] = np.random.randint(len(zbinsc))
    zmeanti = np.rint(zmeant)

    nzTi = np.zeros((len(zbinsc),int(len(zbinsc)/njoin)))
    for i in range(int(len(zbinsc)/njoin)):
        nzTi[:,i] = np.sum(nzti[:,((zmeant>=njoin*i)&(zmeant<njoin*i+njoin))],axis=1)

    if plot:
        plt.figure()
        for i in range(int(len(zbinsc)/njoin)):
            plt.plot(zbinsc,nzTi[:,i])
        plt.show()
    
    return nzTi

def make_nT(nzti, nti, njoin):
    zmeanti = np.zeros(nzti.shape[1])
    for i in range(nzti.shape[1]):
        try: zmeanti[i] = np.average(np.arange(len(zbinsc)),weights=nzti.T[i])
        except: zmeanti[i] = np.random.randint(len(zbinsc))
    zmeanti = np.rint(zmeant)

    nTi = np.zeros(int(len(zbinsc)/njoin))
    for i in range(int(len(zbinsc)/njoin)):
        nTi[i] = np.sum(nti[((zmeant>=njoin*i)&(zmeant<njoin*i+njoin))])
    return nTi

def corr_metric(pzT):
    pzT = pzT/pzT.sum()
    overlap = np.zeros((pzT.shape[1],pzT.shape[1]))
    for i in range(pzT.shape[1]):
        for j in range(pzT.shape[1]):
            overlap[i,j] = np.sum(pzT[:,i]*pzT[:,j])
    overlap = overlap/np.diagonal(overlap)[:,None]
    metric = np.linalg.det(overlap)**(float(pzT.shape[1])/float(len(zbinsc)))
    return metric

### Decide which phenotypes go to which superphenotype
########
### Choose number of superphenotypes
nT = 6
########
bins = {str(b):[] for b in range(nT)}
j = 0 
sumbin = 0
nTs = np.zeros(len(zbinsc))
for i in range(len(zbinsc)):
    sumbin += np.sum(nzt[:,((zmeant==i))],axis=1).sum()
    nTs[i] = np.sum(nzt[:,((zmeant==i))],axis=1).sum()
    if (sumbin <= np.sum(nzt)/(nT-1))|(j==nT-1):
        bins[str(j)].append(i)
        #continue
    else:
        j += 1
        bins[str(j)].append(i)
        sumbin = np.sum(nzt[:,((zmeant==i))],axis=1).sum()
        
        
### Compute p(T), p(z,T) for the superphenotypes
nzTi = np.zeros((len(zbinsc),nT))
nTi = np.zeros((nT))
for i in range(nT):
    nzTi[:,i] = np.sum(make_nzT(nzt,1,False)[:,bins[str(i)]],axis=1)
    nTi[i] = np.sum(make_nT(nzt,nts,1)[bins[str(i)]])
    
print ('Correlation metric = %.3f'%corr_metric(nzTi))


#####################################
### Sampling.
### Prepare matrices for efficient sampling.
#####################################

### Define p(c,chat|bhat)/[p(c|bhat)p(chat|bhat)] --  conditioned on tomographic bin
fcchat = pc_chat.T/pc_chat.sum()

fcchat_0 = fcchat[tomo_bins_wide_modal_even[0]]
fcchat_1 = fcchat[tomo_bins_wide_modal_even[1]]
fcchat_2 = fcchat[tomo_bins_wide_modal_even[2]]
fcchat_3 = fcchat[tomo_bins_wide_modal_even[3]]
fcchat_0 /= np.multiply.outer(np.sum(fcchat_0,axis=1), np.sum(fcchat_0,axis=0))
fcchat_1 /= np.multiply.outer(np.sum(fcchat_1,axis=1), np.sum(fcchat_1,axis=0))
fcchat_2 /= np.multiply.outer(np.sum(fcchat_2,axis=1), np.sum(fcchat_2,axis=0))
fcchat_3 /= np.multiply.outer(np.sum(fcchat_3,axis=1), np.sum(fcchat_3,axis=0))

fcchat_0[~np.isfinite(fcchat_0)] = 0
fcchat_1[~np.isfinite(fcchat_1)] = 0
fcchat_2[~np.isfinite(fcchat_2)] = 0
fcchat_3[~np.isfinite(fcchat_3)] = 0

### Define p(chat|bhat) --  conditioned on tomographic bin
fchat_0 = pchat[tomo_bins_wide_modal_even[0]]
fchat_1 = pchat[tomo_bins_wide_modal_even[1]]
fchat_2 = pchat[tomo_bins_wide_modal_even[2]]
fchat_3 = pchat[tomo_bins_wide_modal_even[3]]

z2Tmap = np.zeros((len(zmeant))).astype(int)
for i in range(nT):
    z2Tmap[np.isin(zmeant.astype(int),bins[str(i)])] = i
    
Fcchat_0 = fcchat_0*fchat_0[:,None]
Fcchat_1 = fcchat_1*fchat_1[:,None]
Fcchat_2 = fcchat_2*fchat_2[:,None]
Fcchat_3 = fcchat_3*fchat_3[:,None]

try:
    print(save_h5)
    store = pd.HDFStore(save_h5)
    store['nzt'] = pd.DataFrame(nzt)
    store['nzTi'] = pd.DataFrame(nzTi)
    store['nTi'] = pd.Series(nTi)
    store['nts'] = pd.Series(nts)
    store['bincond_combined_0'] = pd.DataFrame(bincond_combined[:,:,maskt][0])
    store['bincond_combined_1'] = pd.DataFrame(bincond_combined[:,:,maskt][1])
    store['bincond_combined_2'] = pd.DataFrame(bincond_combined[:,:,maskt][2])
    store['bincond_combined_3'] = pd.DataFrame(bincond_combined[:,:,maskt][3])
    store['R_combined_0'] = pd.DataFrame(Rt_combined[:,:,maskt][0])
    store['R_combined_1'] = pd.DataFrame(Rt_combined[:,:,maskt][1])
    store['R_combined_2'] = pd.DataFrame(Rt_combined[:,:,maskt][2])
    store['R_combined_3'] = pd.DataFrame(Rt_combined[:,:,maskt][3])
    store['sv_th'] = pd.Series(sv_th)
    store['sv_th_deep'] = pd.Series(sv_th_new_final_diag)
    store['varn_th'] = pd.Series(varn_th)
    store['varn_th_deep'] = pd.Series(varn_th_deep_v2)
    store['fcchat_0'] = pd.DataFrame(fcchat_0[:,maskt])
    store['fcchat_1'] = pd.DataFrame(fcchat_1[:,maskt])
    store['fcchat_2'] = pd.DataFrame(fcchat_2[:,maskt])
    store['fcchat_3'] = pd.DataFrame(fcchat_3[:,maskt])
    store['fchat_0'] = pd.Series(fchat_0)
    store['fchat_1'] = pd.Series(fchat_1)
    store['fchat_2'] = pd.Series(fchat_2)
    store['fchat_3'] = pd.Series(fchat_3)
    store['z2Tmap'] = pd.Series(z2Tmap)
    store['maskt'] = pd.Series(maskt)
    store.close()
except:
    pass

#assert False

def return_nzsamples_fromfzt(fzt_dummy):
    fzt = np.zeros((4096,len(zbinsc))).T
    fzt[:,maskt] = fzt_dummy.T

    ### Multiply the f_{zc} by:
    ### - Rzt: the average weight (includes response and shear weight).
    ### - gzt: the fraction probability for each tomographic bin.
    ### to add the bin condition and the average response and shear weights.
    fzt_0 = fzt * bincond_combined[0] * Rt_combined[0]
    fzt_1 = fzt * bincond_combined[1] * Rt_combined[1]
    fzt_2 = fzt * bincond_combined[2] * Rt_combined[2]
    fzt_3 = fzt * bincond_combined[3] * Rt_combined[3]

    fzt_0 /= np.sum(fzt_0)
    fzt_1 /= np.sum(fzt_1)
    fzt_2 /= np.sum(fzt_2)
    fzt_3 /= np.sum(fzt_3)

    fzt_0[~np.isfinite(fzt_0)] = 0
    fzt_1[~np.isfinite(fzt_1)] = 0
    fzt_2[~np.isfinite(fzt_2)] = 0
    fzt_3[~np.isfinite(fzt_3)] = 0

    ### SOMPZ: Equals Eq.2 in https://www.overleaf.com/project/5e8b5a7d3431a1000126471a
    nz_0 = np.einsum('zt,dt->z', fzt_0, Fcchat_0)
    nz_1 = np.einsum('zt,dt->z', fzt_1, Fcchat_1)
    nz_2 = np.einsum('zt,dt->z', fzt_2, Fcchat_2)
    nz_3 = np.einsum('zt,dt->z', fzt_3, Fcchat_3)

    nz_0 /= nz_0.sum()
    nz_1 /= nz_1.sum()
    nz_2 /= nz_2.sum()
    nz_3 /= nz_3.sum()

    nz_samples = np.array([nz_0, nz_1, nz_2, nz_3])
    return nz_samples


nt = sum(maskt)
nz=len(zbinsc)
N_Tcz_Rsample = np.zeros((nT,nt,nz))
for i in range(nT):
    sel = z2Tmap==i
    N_Tcz_Rsample[i, sel] = nzt.T[sel]

N_Tc_Dsample = np.zeros((nT,nt))
for i in range(nT):
    sel = z2Tmap==i
    N_Tc_Dsample[i, sel] = nts[sel]
    
    
alpha = 1e-300

N_T_Rsample = np.sum(N_Tcz_Rsample, axis=(1,2))
N_z_Rsample = np.sum(N_Tcz_Rsample, axis=(0,1))
N_Tz_Rsample = np.sum(N_Tcz_Rsample, axis=(1))
N_cz_Rsample = np.sum(N_Tcz_Rsample, axis=(0))

N_T_Dsample = np.sum(N_Tc_Dsample, axis=(1))
N_c_Dsample = np.sum(N_Tc_Dsample, axis=(0))

lambda_z_step1 = varn_th_deep_v2.copy()
lambda_z_step2 = varn_th.copy()
lambda_mean = np.sum(lambda_z_step1*N_z_Rsample/N_z_Rsample.sum())
lambda_mean_R = np.sum(lambda_z_step2*N_z_Rsample/N_z_Rsample.sum())
lambda_T = np.array([np.sum(lambda_z_step2 * x/x.sum()) for x in N_Tz_Rsample])

onecell = np.sum(N_cz_Rsample>0,axis=1) == 1
N_cz_Rsample_onecell = (N_cz_Rsample/np.sum(N_cz_Rsample,axis=1)[:,None])[onecell]


def draw_3sdir_onlyR():
    
    ### step1
    f_T = np.random.dirichlet(N_T_Rsample/lambda_mean_R+alpha)

    ### step2
    f_z_T = np.array([np.random.dirichlet(x/lambda_T[i]+alpha) for i,x in enumerate(N_Tz_Rsample)])

    ### step3
    f_cz_Rsample = np.random.dirichlet(N_cz_Rsample.reshape(np.prod(N_cz_Rsample.shape))+alpha).reshape(N_cz_Rsample.shape)
    f_cz = np.zeros((nt,nz))
    for k in range(N_Tcz_Rsample.shape[0]):
        sel = z2Tmap==k
        dummy = f_cz_Rsample[sel] 
        dummy = dummy/np.sum(dummy,axis=0)
        dummy[np.isnan(dummy)] = 0
        f_cz[sel] += np.einsum('cz,z->cz', dummy, f_z_T[k])* f_T[k]
        
    return f_cz


def draw_3sdir_newmethod():
    ### step1
    f_T = np.random.dirichlet(N_T_Dsample/lambda_mean+alpha)

    ### step2
    f_cT = np.zeros(nt)
    for k in range(nT):
        sel = z2Tmap==k
        f_cT[sel] = np.random.dirichlet(N_Tc_Dsample[k,sel]+alpha) * f_T[k]
        
    ### step3
    f_cz = draw_3sdir_onlyR()
    f_z_c = f_cz/np.sum(f_cz,axis=1)[:,None]
    f_z_c[onecell] = N_cz_Rsample_onecell
    
    ### compute f_{zc}
    f_cz = f_z_c * f_cT[:,None]
    return f_cz




def aux_fun(i):
    np.random.seed()
    #t0 = time.time()
    nz_samples_newmethod = np.zeros((Nsamples,4, len(zbinsc)))

    #t0 = time.time()
    for i_sample in range(Nsamples):
        f_zt = draw_3sdir_newmethod()
        nz_samples_newmethod[i_sample] = return_nzsamples_fromfzt(f_zt)

    #t1 = time.time()
    #print(t1-t0)
    return nz_samples_newmethod  
        
    
p = Pool(28) #changed to 28 for midway --Raul (original: 64)
nz_samples_newmethod = np.concatenate(p.map(aux_fun, range(28)), axis=0) #changed to 28 for midway --Raul (original: 64)
p.terminate()

sel = np.sum(np.isnan(nz_samples_newmethod),axis=(1,2))==0
nz_samples_newmethod = nz_samples_newmethod[sel]

np.save(outpath+'nz_samples_newmethod.npy', nz_samples_newmethod)

def draw_3sdir_step3_p_zT_onlyR():
    
    ### step1
    f_T = np.random.dirichlet(N_T_Dsample/lambda_mean+alpha)

    ### step2
    f_z_T = np.array([np.random.dirichlet(x/lambda_T[i]+alpha) for i,x in enumerate(N_Tz_Rsample)])

    ### step3
    f_cz_Rsample = np.random.dirichlet(N_cz_Rsample.reshape(np.prod(N_cz_Rsample.shape))+alpha).reshape(N_cz_Rsample.shape)
    f_cz = np.zeros((nt,nz))
    for k in range(N_Tcz_Rsample.shape[0]):
        sel = z2Tmap==k
        dummy = f_cz_Rsample[sel] 
        dummy = dummy/np.sum(dummy,axis=0)
        dummy[np.isnan(dummy)] = 0
        f_cz[sel] += np.einsum('cz,z->cz', dummy, f_z_T[k])* f_T[k]
        
    return f_cz
    
def aux_fun_2(i):
    np.random.seed()
    #t0 = time.time()
    nz_samples_newmethod = np.zeros((Nsamples,4, len(zbinsc)))

    t0 = time.time()
    for i_sample in range(Nsamples):
        f_zt = draw_3sdir_step3_p_zT_onlyR()
        nz_samples_newmethod[i_sample] = return_nzsamples_fromfzt(f_zt)

    #t1 = time.time()
    #print(t1-t0)
    return nz_samples_newmethod 

p = Pool(28) #changed to 28 for midway --Raul (original: 64)
nz_samples = np.concatenate(p.map(aux_fun_2, range(28)), axis=0) #changed to 28 for midway --Raul (original: 64)
p.terminate()

sel = np.sum(np.isnan(nz_samples),axis=(1,2))==0
nz_samples = nz_samples[sel]

np.save(outpath+'nz_samples.npy', nz_samples)
