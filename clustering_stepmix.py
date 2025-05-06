import pandas as pd
import sklearn
from sklearn.cluster import MeanShift, KMeans, DBSCAN, HDBSCAN, OPTICS

import pathlib as pl
from data import get_derivatives
import numpy as np
import os
from visualization import *
from clustering import *
from stepmix.stepmix import StepMix

import warnings
warnings.filterwarnings('ignore')

if __name__ == '__main__':
    method_name = 'stepmix'
    folder_name = 'csv_nn'
    basedir = '/mnt/nas6/data/Target/BMPipeline_full_rerun/PARSED_METS_task_502'
    use_derivatives = False

    complete_data = pd.read_csv(f'{basedir}/{folder_name}/features.csv', index_col=None)
    #complete_data = complete_data[complete_data['RT_matched']]
    
    output =  pl.Path(f'/home/lorenz/BMDataAnalysis/output/{folder_name}/{method_name}_')

    k = range(2, 42)
    
    ## load volume data
    data_tps = ["t1", "t2", "t3", "t4", "t5", "t6"]
    rano_cols = [elem+'_rano' for elem in data_tps]
    data_cols = [elem+'_volume' for elem in data_tps]
    print(f'clustering {len(complete_data)} metastases')

    
    # Normalize by t0 Volume
    complete_data[data_cols] = complete_data[data_cols].div(complete_data["t0_volume"], axis=0)

    print(complete_data[data_cols])
    print(complete_data[data_cols].describe())
    complete_data[data_cols].info()

    if use_derivatives:
        derivatives = get_derivatives(complete_data[["t0_volume"]+data_cols], 'sobel', 'constant', 'relative')
        derivatives.index=complete_data.index
        complete_data = pd.concat([complete_data, derivatives], axis=1)
        print('AFTER DATA MANIPULATION')
        print(complete_data[data_cols])
        print(complete_data[data_cols].describe())
        complete_data[data_cols].info()

        data_cols = list(derivatives.columns)+data_cols



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
    
    filtered_data, invalid_labels = filter_small_clusters(complete_data, 'cluster', 15)

    DB_score = sklearn.metrics.davies_bouldin_score(filtered_data[data_cols], filtered_data['cluster'])
    print(f'Clustering achieved a Davies-Bouldin score of {DB_score}')

    if use_derivatives:
        output = output.parent.parent/(output.parent.name+'_deriv')/output.name
        data_cols = ["t1_volume", "t2_volume", "t3_volume", "t4_volume", "t5_volume", "t6_volume"] # overwrite the columns so it plots just the trajectory not the derivatives

    output = output.parent/(output.name+str(best_k))

    # ## Make plots for cluster
    plot_cluster_centers(filtered_data, output, data_cols, rano_cols, label_col='cluster', init_col='t0_volume')

    # ## load rano data
    plot_sankey(filtered_data[rano_cols], output)

    # ## Make plots for dimesnionality reduction
    mapping = {l: (-1 if l in invalid_labels else l) for l in np.unique(labels)}
    complete_data['cluster'] = complete_data['cluster'].map(mapping)
    plot_umap(complete_data, data_cols, output)

    # ## Make plot for cluster trajectories
    plot_combined_trajectories(filtered_data, data_cols, 'cluster', output)

    ## Make plot for recur prob
    plot_recur_probs(filtered_data, rano_cols, 'cluster', output)

    ## make plot for recur probs but allow any count, not just from t1
    plot_recur_probs_noncausal(filtered_data, rano_cols, output)




