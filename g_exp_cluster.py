#%%

import GEOparse
import pandas as pd
from sklearn.linear_model import Lasso
from tissuespecific.reconstruction import Builder
from scipy.spatial.distance import pdist, jaccard, cosine, squareform
from sklearn.decomposition import PCA , KernelPCA
import seaborn as sns
import matplotlib.pyplot as plt 

def plot_pca(pca_df,label):
       
    g = sns.scatterplot(pca_df.iloc[:, 0],pca_df.iloc[:, 1],hue=label,palette='colorblind')

    g.tick_params(axis='x', labelsize=14)
    g.tick_params(axis='y', labelsize=14)

    # Put a legend to the right side
    #plt.legend(loc='center right', bbox_to_anchor=(1.1, 0.5), ncol=1)

    plt.show(g)

#==============================================================================================

serie = GEOparse.get_GEO('GSE28422')

cluster = [1,1,1,0,0,0,1,1,1,0,1,1,1,1,1,0,0,0,0,0,0,0,0,1]

translator = Builder.affyprobe_translator(serie.gsms['GSM702363'].table, 'hgnc_id')

# aggregate gene expression vectors of old trained /untrained individuals in a DataFrame:
expression_matrix = pd.DataFrame(index=serie.gsms['GSM702363'].table['ID_REF'].values)

for gsm in serie.gsms:

    name = serie.gsms[gsm].metadata['title'][0]

    if 'Old' in name and ('T1' in name or 'T3' in name):

        expression_matrix[name] = serie.gsms[gsm].table['VALUE'].values

#%%

# use Lasso Logistic regression to select features (reactions)
reg = Lasso(alpha = 0.7, tol=0.001).fit(expression_matrix.T,cluster)

coeff = pd.DataFrame(reg.coef_, index = expression_matrix.index)
coeff = coeff[coeff!=0].dropna().sort_values(by=0,ascending = False)

idx = []

for i in list(coeff.index):
    try:
        idx.append(translator[i])
    except:
        idx.append(i)
        continue
coeff.index = idx
coeff

#Does plotting the raw expresson data give the same insights?
#%%

dm = squareform(pdist(expression_matrix.T, metric = 'cosine'))
kpca = KernelPCA(n_components=2,kernel='precomputed').fit_transform(1- dm )
kpca_df = pd.DataFrame(kpca, columns = ['PC1','PC2'])
plot_pca(kpca_df,cluster)

#%%
