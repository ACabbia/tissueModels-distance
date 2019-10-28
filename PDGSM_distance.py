#%%

import os
import cobra
import pandas as pd
import seaborn as sns
from sklearn.decomposition import KernelPCA
from functions import *
import matplotlib.pyplot as plt
from sklearn.linear_model import Lasso, LassoCV
from sklearn.cluster import AgglomerativeClustering

#load models and label

path_lib = '/home/acabbia/Documents/Muscle_Model/models/merged_100_2class/'
path_ref = '/home/acabbia/Documents/Muscle_Model/models/HMRdatabase.xml'

# get tissue label 
label = [filename.split('_')[0] for filename in sorted(os.listdir(path_lib))]

#load models 
ref_model = cobra.io.read_sbml_model(path_ref)

rxn, met, gen, graphlist, flx = load_library(path_lib, ref_model, sampling = False, FBA = True)
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
embed = embedding(M_dist, 'KernelPCA')
sns.scatterplot(embed.iloc[:,0],embed.iloc[:,1],hue=label)
sns.clustermap(M_dist)

clust = AgglomerativeClustering(n_clusters=3,affinity='precomputed',linkage='average').fit(M_dist)
cluster = clust.labels_

#%%
# # Reactions/ pathways associated with clusters?
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

