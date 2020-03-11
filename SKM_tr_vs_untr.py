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
from scipy.stats import kruskal

from statsmodels.stats.multitest import multipletests

import matplotlib
matplotlib.rcParams.update({'font.size':16})

def discr_rxns_analysis(flx, label, ref_model):

    rxns = []
    names = []
    pathway = []
    p = []
    df = pd.DataFrame()

    label_serie = pd.Series(label,index=flx.T.index)

    for reaction in flx.T.columns: 

        flx_0 = flx.T[reaction][label_serie == 0]
        
        flx_1 = flx.T[reaction][label_serie == 1]

        try:
            stat , pval = kruskal(flx_0,flx_1)

        except ValueError:
            print(reaction)
            pass

        rxns.append(reaction)
        names.append(ref_model.reactions.get_by_id(reaction).name)
        pathway.append(ref_model.reactions.get_by_id(reaction).subsystem)
        p.append( pval)

        #print(reaction, ref_model.reactions.get_by_id(reaction).name , pval )

    df['ID'] = rxns
    df['Reaction name'] = names
    df['Pathway'] = pathway
    df['p-value'] = p  
    df['Adjusted p-value'] = multipletests(p,alpha=0.05,method='bonferroni')[1]
    df['Rejected'] = multipletests(p,alpha=0.05,method='bonferroni')[0]

    df['p-value'] = df['p-value'].round(4)
    
    return df[df['p-value']<0.0501]

def plot_flx_means(flx, rxns_df, label):
    
    for r in rxns_df['ID'].values:

        ax = plt.subplot(111)

        mean = flx.T[r].groupby(label).mean()
        std = flx.T[r].groupby(label).std()

        mean.plot.bar(yerr = std, capsize=6,ax = ax, grid = 'on')

        plt.suptitle('mean flux for reaction '+ r)
        plt.xlabel('Cluster')
        plt.ylabel('Flux (mmol/g(DW)/hr)')
        plt.xticks(rotation='horizontal')

        plt.savefig('/home/acabbia/Documents/Muscle_Model/tissueModels-distance/figures/hists/'+r+'.png')

        plt.show()

def pathways_barplot(df,title):

    df['Pathway'].value_counts().plot.barh(figsize = (10,8))
    plt.xlabel('Number of reactions per pathway')
    plt.title(title)
    plt.show()


def plots(flx, rxns_df, label,title):
    
    plot_flx_means(flx,rxns_df,label )
    pathways_barplot(rxns_df,title)


#%%
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
    fig = plt.figure(figsize=(10,7))
    sns.scatterplot(x=e.iloc[:, 0], y=e.iloc[:, 1], hue=label_Tr, s=90)
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
plot_trajectory(embedding, label_Tr, label_num, plot_arrows=True, plot_patient_nr=False)

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
# individual response 
fit = ['01','02','03','12']
unfit = ['04','05','06','10']
responders = ['07','08','09','11']

## gives an error: no significant reactions found?
#cluster_fit_vs_unfit = [1,1,1,0,0,0,1,1,1,0,1,1,1,1,1,0,0,0,0,0,0,0,0,1]
#label_unfit_vs_resp = [0,0,0,1,1,1,0,1,0,0,0,1,1,1,0,1]
#rxns_unfit_vs_resp = discr_rxns_analysis(select(flx,unfit+responders),label_unfit_vs_resp,ref_model)

#%%

# comparison 1: Trained vs Untrained models (ground truth)
label_utr_vs_tr = [1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0]
rxns_utr_vs_tr = discr_rxns_analysis(flx,label_utr_vs_tr,ref_model)
plots(flx,rxns_utr_vs_tr,label_utr_vs_tr, 'Untrained vs Trained')

# comparison 2: fit vs unfit
label_fit_vs_unfit = [1,1,1,0,0,0,0,1,1,1,1,0,0,0,0,1]
rxns_fit_vs_unfit = discr_rxns_analysis(select(flx,fit+unfit),label_fit_vs_unfit,ref_model)
plots(select(flx,fit+unfit),rxns_fit_vs_unfit,label_fit_vs_unfit, 'Unfit vs Fit')

# comparison 3: fit vs responders
label_fit_vs_resp = [1,1,1,0,0,0,0,1,1,1,1,0,0,0,0,1]
rxns_fit_vs_resp = discr_rxns_analysis(select(flx,fit+responders),label_fit_vs_resp,ref_model)
plots(select(flx,fit+responders),rxns_fit_vs_resp,label_fit_vs_resp, 'Fit vs Responders')


# %%
def rxn_content_diff_groups(rxn,label):

    #since we're using jaccard distance, the analysis of the difference between clusters should contain the list of reactions 
    #absent/present in a cluster and not in the other   

    #find % of models in each group that contain each reaction 
    r = rxn.T.groupby(label).sum()
    r = r.T
    r.columns = ['Untrained','Trained']

    # reactions present in models of group 0 but not of group 1
    r_01 = r[(r['Untrained']!=0) & (r['Trained']==0)]

    # reactions present in models of group 1 but not group 0
    r_10 = r[(r['Untrained']==0) & (r['Trained']!=0)]

    return r_01 , r_10

def mk_diff_df(diff, ref_model):

    name = []
    pathway = []
 
    for r in list(diff.index):

        name.append(ref_model.reactions.get_by_id(r).name)
        pathway.append(ref_model.reactions.get_by_id(r).subsystem)

    diff.insert(0,'Pathway',pathway)
    diff.insert(0,'Name',name)

    diff = diff.round(2)
    return diff

#%%
# drop both models for individual nr.5 (outlier) from reaction matrix
rxns_no5 = rxn.copy()
rxns_no5.drop(['SKM_old_train_05','SKM_old_untr_05'], axis=1,inplace=True)

#remake label without those 2 models 
label_utr_vs_tr_no5 = [1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0]

df_01 , df_10 = rxn_content_diff_groups(rxns_no5,label_utr_vs_tr_no5)

rxns_Untrained = mk_diff_df(df_01,ref_model)
rxns_Trained = mk_diff_df(df_10,ref_model)

pathways_barplot(rxns_Untrained,'Reactions present only in Untrained models, by pathway')
pathways_barplot(rxns_Trained,'Reactions present only in Trained models, by pathway')

rxns_Trained.to_csv('reactions_only_Trained.csv', sep=',')
rxns_Untrained.to_csv('reactions_only_Untrained.csv', sep=',')

# %%

rxns_Trained['Trained'].value_counts().plot.barh()
plt.suptitle('Reaction occurrence in Trained models')
plt.xlabel('Number of Reactions')
plt.ylabel('Occurrence, n models/total models')
plt.show()

rxns_Untrained['Untrained'].value_counts().plot.barh()
plt.suptitle('Reaction occurrence in Untrained models')
plt.xlabel('Number of Reactions')
plt.ylabel('Occurrence, n models/total models')
plt.show()

# %%
