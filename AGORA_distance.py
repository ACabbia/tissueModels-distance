#%%
import os
import cobra
import pandas as pd
import seaborn as sns
from sklearn.decomposition import KernelPCA
from functions import *
import matplotlib.pyplot as plt
from sklearn.linear_model import Lasso
from sklearn.cluster import AgglomerativeClustering
#from skbio.stats.distance import mantel
from itertools import product

#load models and label

path_lib = '/home/acabbia/Documents/Muscle_Model/models/AGORA_1.03/'
path_ref = '/home/acabbia/Documents/Muscle_Model/models/AGORA_universe.xml'
tax = '/home/acabbia/Documents/Muscle_Model/GSMM-distance/agora_taxonomy.tsv'

# get AGORA label ('Family')
models_taxonomy = pd.read_csv(tax,sep = '\t')
label = models_taxonomy['mclass']
label.name = 'Taxonomic class'

#reduce number of classes (classes with less than 10 elements are merged into "other")
for s in list(label.value_counts()[label.value_counts()<10].index):
    label.replace(s,'Other', inplace=True)

#load models 
ref_model = cobra.io.read_sbml_model(path_ref)
rxn, met, gen, graphlist, flx = load_library(path_lib, ref_model,sampling = True, FBA=False)
rxn.to_csv('RXN_mat_AGORA.tsv', sep = '\t')

flx.to_csv('smpl_mat_AGORA.tsv', sep = '\t')
#load flux matrix to save time
#flx = pd.read_csv('smpl_mat_AGORA.tsv')

flx = flx.loc[(abs(flx)> 1e-9).any(1)].dropna(how='all')
flx.fillna(value=0,inplace=True)
'''
# compute pairwise metabolic distance matrix 
pw_DM = jaccard_DM(rxn)
pw_GK = gKernel_DM(graphlist)
pw_flx = flux_DM(flx)

pw_GK.columns = pw_GK.index = pw_DM.index

M_dist = (pw_DM + pw_GK + pw_flx ) / 3
'''
# compute pairwise matrices
pw_JD = jaccard_DM(rxn)
pw_GK = gKernel_DM(graphlist)
pw_flx = flux_DM(flx)

pw_GK.columns = pw_GK.index = pw_JD.index

## Heatmaps and PCA plots
for m in [pw_JD, pw_GK, pw_flx]:
    sns.clustermap(m)
    plt.show()
    e = embed(m,'KernelPCA')
    ax = sns.scatterplot(x=e.iloc[:, 0], y=e.iloc[:, 1], hue=label, s=70)
    plt.show()
'''
# correlation between distance matrices (Mantel's test)
values = []
pval = []
for pair in product([pw_JD, pw_GK, pw_flx],repeat=2):
    
    sol = (mantel(pair[0],pair[1],'pearson'))
    values.append(np.round(sol[0], 3))
    pval.append(sol[1])

rez = pd.DataFrame(np.array(values).reshape((3,3)),
                   index = ['Jaccard','GraphKernel','fluxCorr'],
                   columns = ['Jaccard','GraphKernel','fluxCorr'])    
'''
#%%

##plots

embedding = embed(M_dist, 'KernelPCA')
embedding.columns = ['PC1','PC2']

# KPCA scatterplot 
g = sns.scatterplot(embedding.iloc[:,0],embedding.iloc[:,1],hue=label,legend = 'brief',palette='colorblind')
box = g.get_position() # get position of figure
g.set_position([box.x0, box.y0, box.width * 1.25, box.height * 1.25]) # resize position
# Put a legend to the right side
plt.legend(loc='center right', bbox_to_anchor=(1.47, 0.5), ncol=1)
plt.show(g)

#sns.clustermap(M_dist)

clust = AgglomerativeClustering(n_clusters=len(label.value_counts()),
                                affinity='precomputed', linkage='average').fit(M_dist)
cluster = clust.labels_

#%%
models_taxonomy.index = M_dist.index
# # Reactions/ pathways associated with clusters?
## subset flx:  select 2 families, 'Bacilli' and 'Clostridia'

bac = flx.T[models_taxonomy['mclass']=='Bacilli']
clo = flx.T[models_taxonomy['mclass']=='Clostridia']

flx = pd.concat([bac.T,clo.T],axis=1)

# use Lasso Logistic regression to select features (reactions)
reg = Lasso(alpha=10).fit(flx.T,cluster)

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

