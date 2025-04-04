import pandas as pd
import sklearn
from sklearn.cluster import MeanShift, KMeans, DBSCAN, HDBSCAN, OPTICS

import pathlib as pl

import numpy as np
import os
from visualization import *
from clustering import *
from stepmix.stepmix import StepMix

import warnings
warnings.filterwarnings('ignore')

if __name__ == '__main__':
    method_name = 'stepmix'
    folder_name = 'csv_linear_multiclass_reseg_only_valid'



    assert method_name != 'template', "Method name has not been updated"
    volume_data = pd.read_csv(f'/mnt/nas6/data/Target/task_524-504_PARSED_METS_mrct1000_nobatch/{folder_name}/volumes.csv', index_col=None)
    rano_data = pd.read_csv(f'/mnt/nas6/data/Target/task_524-504_PARSED_METS_mrct1000_nobatch/{folder_name}/rano.csv', index_col=None)
    renamer = {elem: 'rano-'+elem for elem in rano_data.columns}
    rano_data = rano_data.rename(columns=renamer)

    complete_data = pd.concat([volume_data, rano_data], axis=1)
    
    output =  pl.Path(f'/home/lorenz/BMDataAnalysis/output/{folder_name}/{method_name}_')

    k = 19#range(2, 42)
    
    ## load volume data
    data_cols = ["60", "120", "180", "240", "300", "360"]
    rano_cols = ['rano-'+elem for elem in data_cols]
    print(f'clustering {len(complete_data)} metastases')
    
    # Normalize by t0 Volume
    complete_data[data_cols] = complete_data[data_cols].div(complete_data["0"], axis=0)

    ##### ------------------ MAKE CHANGES HERE -------------------------------------
    ## Do the clustering
    if not isinstance(k, int):
        best_cluster = None
        best_aic = np.inf
        best_bic = np.inf
        best_k = 1
        for c_k in k:
            cluster = StepMix(n_components=c_k, measurement='continuous', structural='continuous', random_state=42)
            cluster.fit(complete_data[data_cols])

            aic = cluster.aic(complete_data[data_cols])
            bic = cluster.bic(complete_data[data_cols])
            print(f'Output achieved an AIC of {aic} and a BIC of {bic}')

            if best_aic > aic and best_bic > bic:
                best_bic = bic
                best_aic = aic
                best_cluster = cluster
                best_k = c_k
                print('updated best k', best_k)
    else:
        best_cluster = StepMix(n_components=k, measurement='continuous', structural='continuous', random_state=42)
        best_cluster.fit(complete_data[data_cols])
        best_k = k
        best_aic = best_cluster.aic(complete_data[data_cols])
        best_bic = best_cluster.bic(complete_data[data_cols])
    ##### ------------------ MAKE CHANGES HERE -------------------------------------


    labels = best_cluster.predict(complete_data[data_cols])

    complete_data['cluster'] = labels
    print(f'Best clustering achieved an AIC of {best_aic} and a BIC of {best_bic} with {best_k} clusters')
    
    complete_data = filter_small_clusters(complete_data, 'cluster', 15)

    DB_score = sklearn.metrics.davies_bouldin_score(complete_data[data_cols], complete_data['cluster'])
    print(f'Clustering achieved a Davies-Bouldin score of {DB_score}')

    output = output.parent/(output.name+str(best_k))

    ## Make plots for cluster
    # plot_cluster_centers(complete_data, output, data_cols, rano_cols, label_col='cluster', init_col='0')

    ## load rano data
    plot_sankey(complete_data[rano_cols], output)

    # ## Make plots for dimesnionality reduction
    # plot_umap(complete_data, data_cols, output)

    # ## Make plot for cluster trajectories
    # plot_combined_trajectories(complete_data, data_cols, 'cluster', output)

    # ## Make plot for recur prob
    # plot_recur_probs(complete_data, rano_cols, 'cluster', output)




