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

#%%
#load models and label

path_MM = '/home/acabbia/Documents/Muscle_Model/models/muscle_old_sedentary_trained/'
path_ref_MM = '/home/acabbia/Documents/Muscle_Model/models/recon2.2.xml'

# get labels (Tr/Untr) (patients_num) 
label_Tr = pd.Series([s.split('_')[2] for s in sorted(os.listdir(path_MM))])
label_num =  pd.Series([s.split('_')[3].split('.')[0] for s in sorted(os.listdir(path_MM))])

# compute pairwise matrix (jaccard)
ref_model = cobra.io.read_sbml_model(path_ref_MM)
rxn, met, gen, graphlist, smpl = load_library(path_MM, ref_model, sampling = False)

pw_DM = jaccard_DM(rxn)
pw_GK = gKernel_DM(graphlist)
#pw_flx = flux_DM(smpl)

pw_GK.columns = pw_GK.index = pw_DM.index

M_dist = (pw_DM + pw_GK ) / 2

#%%
##plots
embedding = embedding(M_dist, 'KernelPCA')
plot_trajectory(embedding, label_Tr, label_num, plot_arrows=True, plot_patient_nr=True)

clust = AgglomerativeClustering(n_clusters=4, affinity='precomputed', linkage='average').fit(M_dist)
cluster = clust.labels_


#%%

flx = pd.DataFrame()

for filename in sorted(os.listdir(path_MM)):
    print('FBA: model '+ filename)

    # FBA
    model = cobra.io.read_sbml_model(path_MM+filename)
    model = get_bounds_from_file(model,'fluxes.tsv')
    model.objective = 'ATPS4m'
    sol = model.optimize()
    flx[label] = sol.fluxes
    print('+=================================+') 

   
flx = flx.loc[(abs(flx)> 1e-9).any(1)].dropna(how='all')
flx.fillna(value=0,inplace=True)

# Reactions/ pathways associated with clusters?
#cluster = [1,1,1,0,0,0,1,1,1,0,1,1,1,1,1,0,0,0,0,0,0,0,0,1]
cluster_fitness = [2,2,2,0,0,0,1,1,0,0,1,2,2,2,2,0,0,0,1,1,0,0,1,2]

tr={0: 'Untrained', 1:'Trained'}
cl_tr = [tr[c] for c in cluster]

#%%

# use Lasso Logistic regression to select features (reactions)
reg = Lasso(alpha = 0.7).fit(flx.T,cluster)

coeff = pd.DataFrame(reg.coef_, index = flx.index)
coeff = coeff[coeff!=0].dropna().sort_values(by=0,ascending = False)


coeff['Reaction_name'] = [ref_model.reactions.get_by_id(str(r)).name for r in coeff.index]
coeff['Pathway'] = [ref_model.reactions.get_by_id(str(r)).subsystem for r in coeff.index]
coeff = coeff.reset_index()
coeff.columns = ['Reaction ID','Lasso coeff',' Reaction name', 'Pathway']

coeff

#%%
# use Lasso Logistic regression to select features (reactions)
reg = Lasso(alpha = 0.7).fit(flx.T,cluster_fitness)

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

