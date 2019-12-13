#%%
import numpy as np
import os
import cobra
import pandas as pd
import seaborn as sns
from sklearn.decomposition import KernelPCA
from functions import *
import matplotlib.pyplot as plt
from sklearn.linear_model import Lasso
from sklearn.cluster import AgglomerativeClustering
from skbio.stats.distance import mantel
from itertools import product

#load models and label

path_MM = '/home/acabbia/Documents/Muscle_Model/models/muscle_old_sedentary_trained/'
path_ref_MM = '/home/acabbia/Documents/Muscle_Model/models/recon2.2.xml'

# get labels (Tr/Untr) (patients_num) 
label_Tr = pd.Series([s.split('_')[2] for s in sorted(os.listdir(path_MM))])
label_num =  pd.Series([s.split('_')[3].split('.')[0] for s in sorted(os.listdir(path_MM))])

# load models info 
ref_model = cobra.io.read_sbml_model(path_ref_MM)
rxn, met, gen, graphlist, smpl = load_library(path_MM, ref_model, sampling = False, FBA=False)
#smpl.to_csv('sampled_fluxes.csv')
smpl = pd.read_csv('sampled_fluxes.csv', index_col='Unnamed: 0')

#%%
# compute pairwise matrices
pw_JD = jaccard_DM(rxn)
pw_GK = gKernel_DM(graphlist)
pw_flx = flux_DM(smpl)
pw_GK.columns = pw_GK.index = pw_JD.index

## Heatmaps and PCA plots
for m in [pw_JD, pw_GK, pw_flx]:
    sns.clustermap(m)
    plt.show()
    e = embed(m,'KernelPCA')
    ax = sns.scatterplot(x=e.iloc[:, 0], y=e.iloc[:, 1], hue=label_Tr, s=70)
    plt.show()

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

#%%
# Trajectory plot
embedding = embed(pw_JD, 'KernelPCA')
plot_trajectory(embedding, label_Tr, label_num, plot_arrows=True, plot_patient_nr=True)
#sns.clustermap(M_dist)

#Clustering 
clust = AgglomerativeClustering(n_clusters=4, affinity='precomputed', linkage='average').fit(pw_JD)
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
    flx[filename] = sol.fluxes
    print('+=================================+') 

   
flx = flx.loc[(abs(flx)> 1e-9).any(1)].dropna(how='all')
flx.fillna(value=0,inplace=True)

#%%
# Reactions/ pathways associated with response?

cluster = [1,1,1,0,0,0,1,1,1,0,1,1,1,1,1,0,0,0,0,0,0,0,0,1]
cluster_response = [2,2,2,0,0,0,1,1,1,0,1,2,2,2,2,0,0,0,1,1,1,0,1,2]

fit = ['01','02','03','12']
unfit = ['04','05','06','10']
responders = ['07','08','09','11']

f_vs_r_label = [0,0,0,1,1,1,1,0,0,0,0,1,1,1,1,0]
u_vs_r_label = [0,0,0,1,1,1,0,1,0,0,0,1,1,1,0,1]


tr={0: 'Untrained', 1:'Trained'}
cl_tr = [tr[c] for c in cluster]

#%%
# use Lasso Logistic regression to select features (reactions) cluster 0 vs 1

def discriminant_reactions(flx_df, label):

    reg = Lasso(alpha = 0.5).fit(flx_df.T,label)

    coeff = pd.DataFrame(reg.coef_, index = flx.index)
    coeff = coeff[coeff!=0].dropna().sort_values(by=0,ascending = False)

    coeff['Reaction_name'] = [ref_model.reactions.get_by_id(str(r)).name for r in coeff.index]
    coeff['Pathway'] = [ref_model.reactions.get_by_id(str(r)).subsystem for r in coeff.index]
    coeff = coeff.reset_index()
    coeff.columns = ['Reaction ID','Lasso coeff',' Reaction name', 'Pathway']

    return coeff
    
#%%
# use Lasso Logistic regression to select features (reactions) cluster 0 vs 1
clust_0_vs_1 = discriminant_reactions(flx, cluster)

# fit vs responders
fit_vs_responders = discriminant_reactions(select(flx,fit+responders),f_vs_r_label)

# unfit vs responders
unfit_vs_responders = discriminant_reactions(select(flx,unfit+responders),u_vs_r_label)

#%%
# barplots # actually will be KDE density plots of certain reactions
'''
for r in coeff['Reaction ID'].values:

    ax = plt.subplot(111)

    mean = flx.T[r].groupby(cluster).mean()
    std = flx.T[r].groupby(cluster).std()

    mean.plot.bar(yerr = std, capsize=6,ax = ax, grid = 'on')

    plt.suptitle('mean flux for reaction '+ r)
    plt.xlabel('Cluster')
    plt.ylabel('Flux (mmol/g(DW)/hr)')
    plt.xticks(rotation='horizontal')

    plt.savefig('/home/acabbia/Documents/Muscle_Model/tissueModels-distance/figures/supplementary_material/fig1/'+r+'.png')

    plt.show()
'''
#%%

