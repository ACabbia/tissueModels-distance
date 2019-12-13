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


path_lib = '/home/acabbia/Documents/Muscle_Model/models/merged_100_2class/'
path_ref = '/home/acabbia/Documents/Muscle_Model/models/HMRdatabase.xml'

# get tissue label 
label = [filename.split('_')[0] for filename in sorted(os.listdir(path_lib))]

#load models 
ref_model = cobra.io.read_sbml_model(path_ref)
rxn, met, gen, graphlist, flx = load_library(path_lib, ref_model,sampling = True, FBA=False)

flx.to_csv('smpl_mat_PDGSM.tsv', sep = '\t')
#load flux matrix to save time
#flx = pd.read_csv('smpl_mat_PDGSM.tsv')

flx = flx.loc[(abs(flx)> 1e-9).any(1)].dropna(how='all')
flx.fillna(value=0,inplace=True)

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
