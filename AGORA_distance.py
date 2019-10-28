#%%
import os
import cobra
import pandas as pd
import seaborn as sns
from sklearn.decomposition import KernelPCA
from functions import *
import matplotlib.pyplot as plt
from sklearn.linear_model import LassoCV
from sklearn.cluster import AgglomerativeClustering

#load models and label

path_lib = '/home/acabbia/Documents/Muscle_Model/models/AGORA_1.03/'
path_ref = '/home/acabbia/Documents/Muscle_Model/models/AGORA_universe.xml'
tax = '/home/acabbia/Documents/Muscle_Model/GSMM-distance/agora_taxonomy.tsv'

# get AGORA label ('Family')
models_taxonomy = pd.read_csv(tax,sep = '\t')

label = models_taxonomy['mclass']

for s in list(label.value_counts()[label.value_counts()<10].index):
    label.replace(s,'Other', inplace=True)

#load models 
ref_model = cobra.io.read_sbml_model(path_ref)
rxn, met, gen, graphlist, flx = load_library(path_lib, ref_model,sampling = False,FBA=True)

#%%

flx = flx.loc[(abs(flx)> 1e-9).any(1)].dropna(how='all')
flx.fillna(value=0,inplace=True)

# compute pairwise metabolic distance matrix 
pw_DM = jaccard_DM(rxn)
pw_GK = gKernel_DM(graphlist)

pw_GK.columns = pw_GK.index = pw_DM.index

M_dist = (pw_DM + pw_GK ) / 2

#%%
##plots
embedding = embedding(M_dist, 'KernelPCA')
sns.scatterplot(embedding.iloc[:,0],embedding.iloc[:,1],hue=label)
sns.clustermap(M_dist)

clust = AgglomerativeClustering(n_clusters=len(label.value_counts()),
                                affinity='precomputed', linkage='average').fit(M_dist)
cluster = clust.labels_

#%%

# # Reactions/ pathways associated with clusters?
## subset flx:  select 2 families, 'Bacilli' and 'Clostridia'

bac = flx.T[models_taxonomy['mclass']=='Bacilli']
clo = flx.T[models_taxonomy['mclass']=='Clostridia']

flx = pd.concat([bac.T,clo.T],axis=1)

# use Lasso Logistic regression to select features (reactions)
reg = LassoCV().fit(flx.T,cluster)

coeff = pd.DataFrame(reg.coef_, index = flx.index)
coeff = coeff[coeff!=0].dropna().sort_values(by=0,ascending = False)

coeff['Reaction_name'] = [ref_model.reactions.get_by_id(str(r)).name for r in coeff.index]
coeff['Pathway'] = [ref_model.reactions.get_by_id(str(r)).subsystem for r in coeff.index]
coeff = coeff.reset_index()
coeff.columns = ['Reaction ID','Lasso coeff',' Reaction name', 'Pathway']

coeff

#%%
# barplots 

for r in coeff['Reaction ID'].values:

    ax = plt.subplot(111)

    mean = flx.T[r].groupby(cluster).mean()
    std = flx.T[r].groupby(cluster).std()

    mean.plot.bar(yerr = std, capsize=6,ax = ax, grid = 'on')

    plt.suptitle('mean flux for reaction '+ r)
    plt.xlabel('Cluster')
    plt.ylabel('Flux (mmol/g(DW)/hr)')
    plt.xticks(rotation='horizontal')

    plt.savefig('/home/acabbia/Documents/Muscle_Model/tissueModels-distance/figures/hist/'+r+'.png')

    plt.show()

#%%

