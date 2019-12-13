import os
import cobra
import pandas as pd
import grakel as gk
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import pdist, jaccard, cosine, squareform
from sklearn.decomposition import KernelPCA
from sklearn.manifold import MDS, TSNE
from adjustText import adjust_text
from tqdm import tqdm


def binary(model, ref_model):

    # init
    rxns = []
    mets = []
    genes = []

    for r in ref_model.reactions:
        if r in model.reactions:
            rxns.append(1)
        else:
            rxns.append(0)

    for m in ref_model.metabolites:
        if m in model.metabolites:
            mets.append(1)
        else:
            mets.append(0)

    for g in ref_model.genes:
        if g in model.genes:
            genes.append(1)
        else:
            genes.append(0)

    return rxns, mets, genes


def load_library(path, ref_model, sampling=False, FBA=False):
    '''
    loads models from library folder and prepares data structures for further analysis
    returns:

        - Binary matrices (rxn,met,genes) --> EDA and Jaccard
        - Graphlist --> Graph Kernels
        - Flux matrix --> cosine similarity

    '''
    # Init

    reactions_matrix = pd.DataFrame(index=[r.id for r in ref_model.reactions])
    metabolite_matrix = pd.DataFrame(
        index=[m.id for m in ref_model.metabolites])
    gene_matrix = pd.DataFrame(index=[g.id for g in ref_model.genes])
    flx_df = pd.DataFrame(index=[r.id for r in ref_model.reactions])
    graphlist = []

    for filename in sorted(os.listdir(path)):
        model = cobra.io.read_sbml_model(path+filename)
        label = str(filename).split('.')[0]
        print('===================================================================')
        print('Loaded model', label)
        # 1: make binary matrices
        rxns, mets, genes = binary(model, ref_model)
        reactions_matrix[label] = rxns
        metabolite_matrix[label] = mets
        gene_matrix[label] = genes

        # 2: make graphlist

        graphlist.append(modelNet(model, remove_hub_metabolites=True))

        # 3: FBA/sampling
        if sampling:
            try:
                smp = cobra.flux_analysis.sample(model, 1000, processes=4)
            except:
                pass
                
            # find mean from sampled fluxes and append to sampled_fluxes df
                flx_df[label] = smp.mean()

        if FBA:
            print('FBA...')
            bm = [r.id for r in model.reactions if 'biomass'in r.id]

            try:
                bm.remove('EX_biomass(e)')

            except ValueError:
                pass
            
            try:
                model.objective = 'ATPS4m'
            except:
                model.objective = bm[0]

            try:
                model =  get_bounds_from_file(model, 'fluxes.tsv')
            except:
                    
                for e in model.exchanges:
                    e.bounds = -1000, 1000

            sol = model.optimize()

            flx_df[label] = sol.fluxes

    return reactions_matrix, metabolite_matrix, gene_matrix, graphlist, flx_df


def jaccard_DM(df):
    # returns square pairwise (jaccard) distance matrix between elements of df

    DM = pd.DataFrame(squareform(pdist(df.T, metric=jaccard)),
                      index=df.columns, columns=df.columns)

    return DM


def req_list(ref_model, bounds_file):
    # makes required exchange reaction lists from flux input file

    ref_exch = set([r.id for r in ref_model.exchanges])
    requirements = []

    with open(bounds_file, 'r') as f:

        for line in f:

            ex = line.split()[0]

            if ex in ref_exch:
                requirements.append(str(ex))

    return requirements


def remove_currency_metabolites(model):
    # removes the 10 more connected metabolites
    # before the topology analysis (GK)

    ids = []
    num = []

    for met in model.metabolites:
        ids.append(met.id)
        num.append(len(met.reactions))

    currency_met_df = pd.Series(data=num, index=ids)
    currency_met_df.sort_values(ascending=False, inplace=True)
    currency_met_list = list(currency_met_df.head(10).index)

    for m in currency_met_list:
        model.metabolites.get_by_id(m).remove_from_model()

    return model


def modelNet(model, remove_hub_metabolites=True):

    if remove_hub_metabolites:
        model = remove_currency_metabolites(model)

    # Returns a grakel.Graph object from a cobra.model object

    edges_in = []
    edges_out = []
    edges = []

    for r in model.reactions:
        # enumerate 'substrates -> reactions' edges
        substrates = [s.id for s in r.reactants]
        edges_in.extend([(s, r.id) for s in substrates])
        # enumerate 'reactions -> products' edges
        products = [p.id for p in r.products]
        edges_out.extend([(p, r.id) for p in products])

    # Join lists
    edges.extend(edges_in)
    edges.extend(edges_out)

    # labels
    label_m = {m.id: m.name for m in model.metabolites}
    label_r = {r.id: r.name for r in model.reactions}
    label_nodes = label_m
    label_nodes.update(label_r)
    label_edges = {p: p for p in edges}

    g = gk.Graph(edges, node_labels=label_nodes, edge_labels=label_edges)

    return g


def KPCA(DM, label):
    # Kernel-PCA 2-D scatterplot
    kpca = KernelPCA(kernel="precomputed", n_components=2, n_jobs=-1)
    X_kpca = kpca.fit_transform(1-DM)

    g = sns.scatterplot(
        x=X_kpca[:, 0], y=X_kpca[:, 1], hue=label, legend='brief')
    g.tick_params(axis='x', labelsize=14)
    g.tick_params(axis='y', labelsize=14)

    box = g.get_position()  # get position of figure
    g.set_position([box.x0, box.y0, box.width, box.height])  # resize position
    # Put a legend to the right side
    plt.legend(loc='center right', bbox_to_anchor=(1.5, 0.5), ncol=1)

    plt.show(g)


def get_bounds_from_file(model, file):

    # close all lower bounds
    for r in model.exchanges:
        r.bounds = 0, 1000

    # get new lower bounds from file
    with open(file, 'r') as f:

        for line in f:
            rxn, bound = line.split()

            try:
                model.exchanges.get_by_id(rxn).bounds = -np.float(bound), 1000

            except:
                print('Reaction '+rxn+' not in the model')
                pass

    return model


def plot_trajectory(embedding_df, label_Tr, patient_nr_label, plot_arrows=True, plot_patient_nr=True):

    # plot training trajectories

    data = pd.concat([embedding_df, patient_nr_label], axis=1)
    data.columns = ['x', 'y', 'patient_nr']

    # plot arrows between untr --> tr for same subject
    ax = sns.scatterplot(
        x=embedding_df.iloc[:, 0], y=embedding_df.iloc[:, 1], hue=label_Tr, s=70)

    for x in set(data['patient_nr'].values):

        df = data[data['patient_nr'] == x]

        A = [df.iloc[1, 0], df.iloc[1, 1]]
        B = [df.iloc[0, 0], df.iloc[0, 1]]

        if plot_arrows:
            ax.arrow(A[0], A[1], B[0]-A[0], B[1]-A[1],
                     length_includes_head=True, color='black', alpha=0.5)
        else:
            continue

        if plot_patient_nr:

            plt.text(A[0] * (1 + 0.1), A[1] * (1 + 0.1),
                     int(df.iloc[0, 2]), fontsize=9)
            plt.text(B[0] * (1 + 0.1), B[1] * (1 + 0.1),
                     int(df.iloc[0, 2]), fontsize=9)
        else:
            continue

    plt.suptitle('Training trajectories in older adults ')
    plt.xlabel('PC 1')
    plt.ylabel('PC 2')
    plt.show()


def plot_trajectory_adj(embedding_df, label_Tr, patient_nr_label, plot_arrows=True, plot_patient_nr=True):

    # plot training trajectories

    data = pd.concat([embedding_df, patient_nr_label], axis=1)
    data.columns = ['x', 'y', 'patient_nr']

    # plot arrows between untr --> tr for same subject
    ax = sns.scatterplot(
        x=embedding_df.iloc[:, 0], y=embedding_df.iloc[:, 1], hue=label_Tr, s=80)

    for x in set(data['patient_nr'].values):

        df = data[data['patient_nr'] == x]

        A = [df.iloc[1, 0], df.iloc[1, 1]]
        B = [df.iloc[0, 0], df.iloc[0, 1]]

        if plot_arrows:

            arr = []
            a = ax.arrow(A[0], A[1], B[0]-A[0], B[1]-A[1],
                         length_includes_head=True, color='black', alpha=0.5)
            arr.append(a)

        else:
            continue

        if plot_patient_nr:

            texts = [plt.text(A[0], A[1], str(int(df.iloc[0, 2])))]
            texts.append(plt.text(B[0], B[1], str(int(df.iloc[0, 2]))))

            adjust_text(texts, add_objects=arr, expand_text=(
                1.5, 1.5), expand_points=(1.5, 1.5))

        else:
            continue

    plt.suptitle('Training trajectories in older adults ')
    plt.xlabel('PC 1')
    plt.ylabel('PC 2')
    plt.show()

def Mdist(rxn, graphlist, flux_df):

    pw_DM = jaccard_DM(rxn)
    pw_GK = gKernel_DM(graphlist)
    pw_flx = flux_DM(flux_df)

    M_dist = (pw_DM + pw_GK + pw_flx) / 3

    return M_dist

def flux_DM(df):
    # returns square pairwise (cosine) distance matrix between elements of sampled flux df
    df.fillna(inplace=True, value=0)
    # take absolute value to ensure positivity == metric
    DM = df.corr().abs()
    return 1-DM


def gKernel_DM(graphList):
    # returns 1 - kernel similarity matrix (i.e. distance)
    gkernel = gk.WeisfeilerLehman(
        base_kernel=gk.VertexHistogram, normalize=True)
    K = pd.DataFrame(gkernel.fit_transform(graphList))
    return 1-K

def embed(pw_matrix, embed):

    # different embeddings (MDS, KPCA, TSNE)

    if embed == 'KernelPCA':
        embedding = pd.DataFrame(KernelPCA(
            kernel="precomputed", n_components=2).fit_transform(1-pw_matrix))
    elif embed == 'TSNE':
        embedding = pd.DataFrame(TSNE(
            metric="precomputed", random_state=42, perplexity=5).fit_transform(pw_matrix))
    elif embed == 'MDS':
        embedding = pd.DataFrame(MDS(
            n_components=2, dissimilarity='precomputed', random_state=42).fit_transform(pw_matrix))
    else:
        print('Embedding type unknown. Possible values: KernelPCA, TSNE, MDS')

    return embedding


def select(df, to_keep = []):

    # selects rows and columns in a df
    # to_keep : list of strings or substrings of the indexes of the cols to keep

    ids = []
    for i in df.columns:
        for n in to_keep:
            if n in i:
                ids.append(i)

    new_dm = df.loc[:,ids]

    return new_dm

def select_DM(pw_DM, to_keep = []):

    # selects rows and columns in a square (symmetric) distance matrix
    # to_keep : list of strings or substrings of the indexes of the rows/cols to keep

    ids = []
    for i in pw_DM.index:
        for n in to_keep:
            if n in i:
                ids.append(i)

    new_dm = pw_DM.loc[ids,ids]

    return new_dm


def RQ(model):
    model.objective = 'ATPS4m'
    sol = model.optimize()
    
    rq = np.nan

    try:
        rq = abs(sol['EX_co2(e)']) / abs(sol['EX_o2(e)'])
    except:
        print('something happened')

    return rq


def RQ_test(path):
    'Functional Metabolic tests'
    carbon_sources = ['EX_glc(e)','EX_fru(e)','EX_octa(e)','EX_lnlnca(e)','EX_hdcea(e)','EX_lnlc(e)',
                      'EX_ttdca(e)','EX_hdca(e)','EX_ocdca(e)','EX_arach(e)','EX_lgnc(e)']
    rq_df = pd.DataFrame(index = carbon_sources)

    for filename in sorted(os.listdir(path)):
        model = cobra.io.read_sbml_model(path+filename)
        print(filename)

        rq_list = []
        for c in carbon_sources:

            #close all inflows , leave outflows open
      
            for r in model.exchanges:
                r.bounds = 0,0
            
            # make sure sink reactions are irreversible
            for r in model.reactions:
                if 'sink' in r.id:
                    r.bounds = 0,1000

            try:
                #allow oxygen and water free exchange, co2 outflow 
                model.reactions.get_by_id('EX_o2(e)').bounds= -1000,1000
                model.reactions.get_by_id('EX_h2o(e)').bounds= -1000,1000
                model.reactions.get_by_id('EX_co2(e)').bounds= 0,1000
                #model.reactions.get_by_id('EX_h(e)').bounds= 0,1000 
                        
                # allow inflow of selected carbon source 
                model.reactions.get_by_id(c).bounds = -1000, 1000 
            
            except KeyError:
                pass        
    
            # FBA and RQ computations
            rq = RQ(model)
            rq_list.append(rq)
            print(c)
            print('RQ = ',rq)
            #print(model.summary())
                    
        print("===============================================================")

        rq_df[filename] = rq_list

    return rq_df     

    ## glc and fru don't work yet