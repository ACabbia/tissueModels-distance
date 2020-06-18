
#%%
import numpy as np
import os
import cobra
import pandas as pd
import seaborn as sns
from functions import *
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.cluster import AgglomerativeClustering
from scipy.stats import kruskal
from Cluster_Ensembles import cluster_ensembles
from sklearn.decomposition import KernelPCA
import matplotlib

import logging
logging.getLogger("cobra.io.sbml").setLevel(logging.CRITICAL)

matplotlib.rcParams.update({'font.size':16})

def discr_rxns_analysis(flx, label):

    rxns = []
    names = []
    pathway = []
    p = []
    df = pd.DataFrame()

    label_serie = pd.Series(label,index=flx.index)              

    errors = []

    for reaction in flx.columns: 

        flx_0 = flx[reaction][label_serie == 0]
        
        flx_1 = flx[reaction][label_serie == 1]

        try:
            stat , pval = kruskal(flx_0,flx_1)

        except ValueError:
            errors.append(reaction)
            pass

        rxns.append(reaction)
        names.append(ref_model.reactions.get_by_id(reaction).name)
        pathway.append(ref_model.reactions.get_by_id(reaction).subsystem)
        p.append( pval)


    df['ID'] = rxns
    df['Reaction name'] = names
    df['Pathway'] = pathway
    df['p-value'] = p  

    df['p-value'] = df['p-value'].round(4)
    
    return df , errors

def consensus_clustering(pw_mat_list):

    names = ['Jaccard','GraphKernel','fluxCorr']
    clusterings = pd.DataFrame()

    for i, pw_mat in enumerate(pw_mat_list):
       
        clust = AgglomerativeClustering(n_clusters=4, affinity='precomputed', linkage='average').fit(pw_mat)
        clusterings[names[i]] = list(clust.labels_) 

    consensus_labels = cluster_ensembles(clusterings.T.values , N_clusters_max = 2)
    
    return consensus_labels

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

        plt.savefig('figures/savefig/'+r+'.png')

        plt.show()

def pathways_barplot(df,title):

    df['Pathway'].value_counts().plot.barh(figsize = (10,8))
    plt.xlabel('Number of reactions per pathway')
    plt.title(title)
    plt.savefig('figures/savefig/'+title+'.png')

    plt.show()


def plots(flx, rxns_df, label,title):
    
    plot_flx_means(flx,rxns_df,label )
    pathways_barplot(rxns_df,title)

def rxn_content_diff_groups(rxn,label):

    #since we're using jaccard distance, the analysis of the difference between clusters should contain the list of reactions 
    #absent/present in a cluster and not in the other   

    #find % of models in each group that contain each reaction 
    r = rxn.T.groupby(label).sum()
    r = r.T
    r.columns = ['UT','AT']

    # reactions present in models of group 0 but not of group 1
    r_01 = r[(r['UT']!=0) & (r['AT']==0)]

    # reactions present in models of group 1 but not group 0
    r_10 = r[(r['UT']==0) & (r['AT']!=0)]

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
#load models and label

path_MM = 'models/SKM_older_adults/'
path_ref_MM = 'models/recon2.2.xml'

# get labels (Tr/Untr) (patients_num) 
label_Tr = pd.Series([s.split('_')[1] for s in sorted(os.listdir(path_MM))])
label_num =  pd.Series([s.split('_')[2].split('.')[0] for s in sorted(os.listdir(path_MM))])

# load models info 
ref_model = cobra.io.read_sbml_model(path_ref_MM)
rxn, met, gen, graphlist, smpl = load_library(path_MM, ref_model, graph = True , sampling = False, FBA=False)
#smpl.to_csv('sampled_fluxes.csv')
smpl = pd.read_csv('csv/sampled_fluxes.csv', index_col='Unnamed: 0')
#%%
# compute pairwise matrices
pw_JD = jaccard_DM(rxn)
pw_GK = gKernel_DM(graphlist)
pw_flx = flux_DM(smpl)
pw_GK.columns = pw_GK.index = pw_flx.columns = pw_flx.index =  pw_JD.index

## Heatmaps and PCA plots
names = ['Jd','Gk','Flx']

for i , m in enumerate([pw_JD, pw_GK, pw_flx]):
    sns.clustermap(m)
    plt.savefig( 'figures/savefig/SKM_heatmap_'+names[i]+'.png',bbox_inches='tight')
    plt.show()
    e = embed(m,'KernelPCA')
    fig = plt.figure(figsize=(10,7))
    sns.scatterplot(x=e.iloc[:, 0], y=e.iloc[:, 1], hue= label_Tr, s=90)
    plt.legend(loc='center right', bbox_to_anchor=(1.25, 0.94), ncol=1)
    plt.savefig('figures/savefig//SKM_KPCA_'+names[i]+'.png',bbox_inches='tight')
    plt.show()

mantels_rez = mantel_test([pw_JD, pw_GK, pw_flx],'SKM')

#%%
# Trajectory plot
embedding = embed(pw_JD, 'KernelPCA')
plot_trajectory(embedding, label_Tr, label_num,'Training_status', plot_arrows=True, plot_patient_nr=True)

# Consensus Clustering 
consensus_label = consensus_clustering([pw_JD, pw_GK, pw_flx])
print(consensus_label)
plot_trajectory(embedding, consensus_label, label_num, 'clustering', plot_arrows=False, plot_patient_nr=True)
plt.show()
#%%
flx = smpl
consensus_label = [0,0,0,1,1,1,0,0,0,1,0,0,0,0,0,1,1,1,1,1,1,1,1,0]
training_label = [1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0]
z = zscores(flx)
rxns_clust , not_tested1 = discr_rxns_analysis(z,consensus_label)
rxns_clust.to_csv('csv/discr_rxn_consensus_clust.csv', sep =',')

rxns_tr , not_tested2 = discr_rxns_analysis(z,training_label)
rxns_tr.to_csv('csv/discr_rxn_consensus_train.csv', sep =',')

# %%
smpl = pd.read_csv('csv/sampled_fluxes.csv', index_col='Unnamed: 0')

rez_clust  = pd.read_csv('csv/discr_rxn_consensus_clust.csv' , index_col='ID').drop('Unnamed: 0',1)
rez_clust = rez_clust[rez_clust['p-value']<0.051]
rez_clust['Pathway'].value_counts(ascending=True).plot.barh(figsize =(15,12))
plt.xlabel('Number of significant reactions')
plt.ylabel('Pathway')
plt.savefig('figures/savefig/diff_rxn_clust.png',bbox_inches='tight')
plt.show()

rez_train  = pd.read_csv('csv/discr_rxn_consensus_train.csv' , index_col='ID').drop('Unnamed: 0',1)
rez_train = rez_train[rez_train['p-value']<0.051]
rez_train['Pathway'].value_counts(ascending=True).plot.barh(figsize =(15,12))
plt.xlabel('Number of significant reactions')
plt.ylabel('Pathway')
plt.savefig('figures/savefig/diff_rxn_train.png',bbox_inches='tight')
plt.show()

# %%
consensus_label = np.array(consensus_label)
plot_zscores(z,['ELAIDCPT1','C50CPT1'], 1-consensus_label)
plot_zscores(z,['FAOXC101C102m','KYN'],training_label)

# %%
