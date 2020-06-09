#%%
import os
import cobra
from functions import *
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib

matplotlib.rcParams.update({'font.size':15,'legend.loc':'upper right'})
savefolder= 'figures/savefig'

#load models and label
path_lib = 'models/AGORA_1.03/'
path_ref = 'models/AGORA_universe.xml'
tax = 'csv/agora_taxonomy.tsv'

# get AGORA label ('Family')
models_taxonomy = pd.read_csv(tax,sep = '\t')
label = models_taxonomy['mclass']
label.name = 'Taxonomic class'

#reduce number of classes (classes with less than 10 elements are merged into "other")
for s in list(label.value_counts()[label.value_counts()<10].index):
    label.replace(s,'Other', inplace=True)

#load models 
ref_model = cobra.io.read_sbml_model(path_ref)
rxn, met, gen, graphlist, flx = load_library(path_lib, ref_model,graph=True,sampling = False)
#flx.to_csv('csv/smpl_mat_AGORA.tsv', sep = '\t')
flx = pd.read_csv('csv/flx_mat_AGORA.csv',sep = ',', index_col='Unnamed: 0')

# compute pairwise matrices
pw_JD = jaccard_DM(rxn)
pw_GK = gKernel_DM(graphlist)
pw_flx = flux_DM(flx)
#save
pw_JD.to_csv('csv/Agora_dm_jd.csv',sep=',')
pw_GK.to_csv('csv/Agora_dm_gk.csv',sep=',')
pw_flx.to_csv('csv/Agora_dm_flx.csv',sep=',')

pw_GK.columns = pw_GK.index = pw_JD.index
# Distance matrices
dm_jd = pd.read_csv('csv/Agora_dm_jd.csv', sep=',',index_col='Unnamed: 0' )
dm_gk = pd.read_csv('csv/Agora_dm_gk.csv', sep=',',index_col='Unnamed: 0')
dm_flx = pd.read_csv('csv/Agora_dm_flx.csv', sep=',',index_col='Unnamed: 0')
dm_gk.index = dm_gk.columns = dm_jd.index


#%%

# Mantel's test 
mantels_rez = mantel_test([pw_JD, pw_GK, pw_flx],'AGORA')

## AGORA full dataset plot
embedding = embed(dm_jd, 'KernelPCA')

# KPCA scatterplot (full dataset)
g = sns.scatterplot(embedding.iloc[:,0],embedding.iloc[:,1],hue=label,legend = 'brief',palette='colorblind')
box = g.get_position() # get position of figure
g.set_position([box.x0, box.y0, box.width * 1.25, box.height * 1.25]) # resize position
# Put a legend to the right side
plt.legend(loc='center right', bbox_to_anchor=(1.47, 0.5), ncol=1)
plt.savefig(savefolder+'/AGORA_KPCA_fullset.png',bbox_inches='tight')
plt.show(g)


#%%

# Make subset of AGORA (only Bacilli and Clostridia) 
to_keep = (models_taxonomy['mclass']=='Bacilli') | (models_taxonomy['mclass']=='Clostridia')

dm_jd_s = dm_jd.loc[to_keep.values,to_keep.values]
dm_gk_s = dm_gk.loc[to_keep.values,to_keep.values]
dm_flx_s = dm_flx.loc[to_keep.values,to_keep.values]


label = list(models_taxonomy['mclass'].loc[to_keep])
names = ['Jd','Gk','Flx']

## Heatmaps and PCA plots of AGORA subset
for i , m in enumerate([dm_jd_s, dm_gk_s, dm_flx_s]):
    sns.clustermap(m)
    filename = savefolder+'/AGORA_heatmap_'+names[i]+'.png'
    plt.savefig(filename,bbox_inches='tight')
    e = embed(m,'KernelPCA')
    fig = plt.figure(figsize=(10,7))
    sns.scatterplot(x=e.iloc[:, 0], y=e.iloc[:, 1], hue=label, s=90)
    plt.legend(loc='center right', bbox_to_anchor=(1.25, 0.94), ncol=1)
    plt.savefig(savefolder+'/AGORA_KPCA_'+names[i]+'.png',bbox_inches='tight')
    


# %%
'''
label = models_taxonomy['mclass']
label.name = 'Taxonomic class'

#reduce number of classes (classes with less than 10 elements are merged into "other")
for s in list(label.value_counts()[label.value_counts()<10].index):
    label.replace(s,'Other', inplace=True)

embedding = embed(dm_jd, 'KernelPCA')
embedding.columns = ['PC1','PC2']

# KPCA scatterplot 
g = sns.scatterplot(embedding.iloc[:,0],embedding.iloc[:,1],hue=label,legend = 'brief',palette='colorblind')
box = g.get_position() # get position of figure
g.set_position([box.x0, box.y0, box.width * 1.25, box.height * 1.25]) # resize position
# Put a legend to the right side
plt.legend(loc='center right', bbox_to_anchor=(1.57, 0.5), ncol=1)
plt.savefig(savefolder+'/AGORA_KPCA_fullset.png',bbox_inches='tight')
plt.show()
'''
# %%
