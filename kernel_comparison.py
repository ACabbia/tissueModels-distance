import cobra
import grakel as gk
import os
from time import process_time as clock
from functions import modelNet
import pandas as pd
from functions import embed
import matplotlib.pyplot as plt
import seaborn as sns
import logging
logging.getLogger("cobra.io.sbml").setLevel(logging.CRITICAL)
        
model_folder = 'models/muscle_old_sedentary_trained/'

graph_list = []
filenames = []

for filename in sorted(list(os.listdir(model_folder))):
    print('loading model ', filename)
    filenames.append(filename.split('.')[0])
    model = cobra.io.read_sbml_model(model_folder+filename)
    graph_list.append(modelNet(model, remove_hub_metabolites=True))

kernel_list = [gk.RandomWalk(method_type='baseline'),
              gk.RandomWalk(method_type='fast'),
              gk.WeisfeilerLehman(base_kernel=gk.VertexHistogram, normalize=True)]

names=['RW-Gartner03','RW-Vishwwnathan10''WLS']
label_tr = [name.split('_')[2] for name in filenames ]
times = []

for i, kernel in enumerate(kernel_list):
    
    print('Starting', names[i], 'kernel')

    start = clock()
    km = kernel.fit_transform(graph_list)
    end = clock()
    K = pd.DataFrame(1-km)
    K.index = K.columns = filenames

    sns.clustermap(K)
    plt.savefig('figures/savefig/graph_kernels/heatmap_'+names[i]+'.png',bbox_inches='tight')
 
    e = embed(K,'KernelPCA')
    fig = plt.figure(figsize=(10,7))
    sns.scatterplot(x=e.iloc[:, 0], y=e.iloc[:, 1], hue=label_tr, s=90)
    plt.savefig('figures/savefig/graph_kernels/scatterplot_'+names[i]+'.png',bbox_inches='tight')

    time = end-start
    times.append(time)

times_df = pd.DataFrame(index=names)
times_df['Time (s)'] = times
times_df.plot.bar(logy=True)
plt.savefig('figures/savefig/graph_kernels/times.png',bbox_inches = 'tight')
times_df.to_csv('csv/gkernel_times.csv',sep=',')

