#%%
import os
import cobra
import pandas as pd
import seaborn as sns
from functions import *
import matplotlib.pyplot as plt
import matplotlib


matplotlib.rcParams.update({'font.size':15,
                            'legend.loc':'upper right'})

savefolder= 'figures/savefig'

#load models and label

path_lib = 'models/PDGSM_2class/'
path_ref = 'models/HMRdatabase.xml'

# get tissue label 
label = [filename.split('_')[0] for filename in sorted(os.listdir(path_lib))]

#load models 
ref_model = cobra.io.read_sbml_model(path_ref)
rxn, met, gen, graphlist, flx = load_library(path_lib, ref_model, graph=True, sampling = False)
flx = pd.read_csv('csv/flx_mat_PDGSM.csv',sep = ',', index_col='Unnamed: 0')

# compute pairwise metabolic distance matrix 
pw_JD = jaccard_DM(rxn)
pw_GK = gKernel_DM(graphlist)
pw_flx = flux_DM(flx)

mantels_rez = mantel_test([pw_JD, pw_GK, pw_flx],'PDGSM') 

# save
pw_JD.to_csv('csv/PDGSM_dm_jd.tsv', sep = '\t')
pw_GK.to_csv('csv/PDGSM_dm_gk.tsv', sep = '\t')
pw_flx.to_csv('csv/PDGSM_dm_flx.tsv', sep = '\t')


# Distance matrices
dm_jd = pd.read_csv('csv/PDGSM_dm_jd.tsv', sep='\t',index_col='Unnamed: 0' )
dm_gk = pd.read_csv('csv/PDGSM_dm_gk.tsv', sep='\t',index_col='Unnamed: 0')
dm_flx = pd.read_csv('csv/PDGSM_dm_flx.tsv', sep='\t',index_col='Unnamed: 0')

# Fix some mistakes...
dm_gk.columns = dm_gk.index = dm_jd.columns
dm_flx.fillna(0, inplace=True)
#%%
tr = {'L': 'Liver' , 'S': 'Skin'}
label = [tr[filename.split('_')[0]] for filename in dm_jd.index]

names = ['Jd','Gk','Flx']
## Heatmaps and PCA plots
for i , m in enumerate([dm_jd, dm_gk, dm_flx]):
    sns.clustermap(m)
    filename = savefolder+'/PDGSM_heatmap_'+names[i]+'.png'
    plt.savefig(filename,bbox_inches='tight')
    e = embed(m,'KernelPCA')
    fig = plt.figure(figsize=(10,7))
    sns.scatterplot(x=e.iloc[:, 0], y=e.iloc[:, 1], hue=label, s=90)
    plt.legend(loc='center right', bbox_to_anchor=(1.25, 0.94), ncol=1)
    plt.savefig(savefolder+'/PDGSM_KPCA_'+names[i]+'.png',bbox_inches='tight')

# %%

# %%
