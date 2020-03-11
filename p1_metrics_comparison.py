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

def comparison(rxn, graphlist, flx): 

    dm_JD = jaccard_DM(rxn)
    dm_GK = gKernel_DM(graphlist)
    dm_FLX = flux_DM(flx)

    for m in [dm_JD, dm_GK, dm_FLX]:
        #heatmap
        sns.clustermap(m)
        plt.show()
   
        #PCA
        e = embed(m,'KernelPCA')
        ax = sns.scatterplot(x=e.iloc[:, 0], y=e.iloc[:, 1], s=70)
        plt.show()

    values = []
    pval = []

    for pair in product([dm_JD, dm_GK, dm_FLX],repeat=2):
    
        sol = (mantel(pair[0],pair[1],'pearson'))
        values.append(np.round(sol[0], 3))
        pval.append(sol[1])

    rez = pd.DataFrame(np.array(values).reshape((3,3)),
                   index = ['Jaccard','GraphKernel','fluxCorr'],
                   columns = ['Jaccard','GraphKernel','fluxCorr']) 
        
    return rez

# paper part 1:  compare the three metrics
# heatmaps 
# PCA plots 
# mantels test 

######## AGORA
path_lib = '/home/acabbia/Documents/Muscle_Model/models/merged_100_2class/'
path_ref = '/home/acabbia/Documents/Muscle_Model/models/HMRdatabase.xml'

# load library  
ref_model = cobra.io.read_sbml_model(path_ref)
rxn, met, gen, graphlist, flx = load_library(path_lib, ref_model, sampling = True, FBA=False)
flx = pd.read_csv('flx_mat_PDGSM.csv', index_col='Unnamed: 0')

rez = comparison(rxn,graphlist,flx)


'''
path_lib = '/home/acabbia/Documents/Muscle_Model/models/AGORA_1.03/'
path_ref = '/home/acabbia/Documents/Muscle_Model/models/AGORA_universe.xml'


'''