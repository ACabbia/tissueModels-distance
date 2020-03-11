# print statistics and plots for a library of GSMM
#%%
import cobra
import pandas as pd
import seaborn as sns
from functions import load_library
from matplotlib.pyplot import show , figure
from numpy import round

import matplotlib
matplotlib.rcParams.update({'font.size':16})

def split_classes(df):
    #splits bin matrices in class specific df's
    df1 = pd.DataFrame()
    df2 = pd.DataFrame()
    
    for c in df.columns:
        if c.split('_')[2] == 'Trained':
            df1[c] = df[c]
        elif c.split('_')[2] == 'Untrained':
            df2[c] = df[c]
    return df1 , df2

def model_stats_plot(rxn,met,gen):

    R_df = pd.DataFrame()
    M_df = pd.DataFrame()
    G_df = pd.DataFrame()
    table_df = pd.DataFrame(index = ['Median nr. Reactions',
                                     'Median nr. Metabolites',
                                     'Median nr. Genes'])

    R_trained, R_untrained = split_classes(rxn)
    M_trained, M_untrained = split_classes(met)
    G_trained, G_untrained = split_classes(gen)

    # PRINT STATS
    print('Median number of reactions in "Trained" models: ',round(R_trained.sum().median(),2))
    print('Median number of reactions in "Trained" models: ',round(R_untrained.sum().median(),2))
    print('Median number of metabolites in "Trained" models: ',round(M_trained.sum().median(),2))
    print('Median number of metabolites in "Trained" models: ',round(M_untrained.sum().median(),2))
    print('Median number of genes in "Trained" models: ',round(G_trained.sum().median(),2))
    print('Median number of genes in "Trained" models: ',round(G_untrained.sum().median(),2))

    table_df['Untrained'] = [round(R_untrained.sum().median(),2),
                             round(M_untrained.sum().median(),2),
                             round(G_untrained.sum().median(),2)]
    table_df['Trained'] = [round(R_trained.sum().median(),2),
                           round(M_trained.sum().median(),2),
                           round(G_trained.sum().median(),2)]

    # Plot model content (R/M/G) distributions (Violin plots)
    R_df["Untrained"] = R_untrained.sum().values
    R_df["Trained"] = R_trained.sum().values
    R_df = R_df.melt(var_name = "Condition", value_name = 'Number of reactions')
    figure(figsize=(10,7))
    sns.violinplot(x="Condition", y="Number of reactions", data=R_df).set_title('Reactions content')
    show()

    M_df["Untrained"] = M_untrained.sum().values
    M_df["Trained"] = M_trained.sum().values
    M_df = M_df.melt(var_name = "Condition", value_name = 'Number of metabolites')
    figure(figsize=(10,7))
    sns.violinplot(x="Condition", y="Number of metabolites", data=M_df).set_title('Metabolites content')
    show()

    G_df["Untrained"] = G_untrained.sum().values
    G_df["Trained"] = G_trained.sum().values
    G_df = G_df.melt(var_name = "Condition", value_name = 'Number of genes')
    figure(figsize=(10,7))
    sns.violinplot(x="Condition", y="Number of genes", data=G_df).set_title('Genes content')
    show()

    return table_df


path_lib = '/home/acabbia/Documents/Muscle_Model/models/muscle_old_sedentary_trained/'
path_ref = '/home/acabbia/Documents/Muscle_Model/models/recon2.2.xml'
ref_model = cobra.io.read_sbml_model(path_ref)

rxn, met, gen, graphlist, flx = load_library(path_lib, ref_model, sampling = False, FBA=False)



#%%

table = model_stats_plot(rxn,met,gen)