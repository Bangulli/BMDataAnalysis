import pandas as pd
import sklearn
from sklearn.cluster import MeanShift, KMeans, DBSCAN, HDBSCAN, OPTICS
from sktime.datatypes import convert_to
import pathlib as pl
from sktime.clustering.k_means import TimeSeriesKMeans
import numpy as np
import os
from data import get_derivatives
import copy
from visualization import *
from clustering import *
from stepmix.stepmix import StepMix

import warnings
warnings.filterwarnings('ignore')

if __name__ == '__main__':
    use_valid = True
    method_name = f'''tsxmeans_{'valid' if use_valid else 'all'}'''
    folder_name = 'csv_nn'
    use_derivatives = False
    complete_data = pd.read_csv(f'/mnt/nas6/data/Target/BMPipeline_full_rerun/PARSED_METS_task_502/{folder_name}/features.csv', index_col=None)
    if use_valid: complete_data = complete_data[complete_data['RT_matched']]
    output =  pl.Path(f'/home/lorenz/BMDataAnalysis/output/{folder_name}/{method_name}_')

    k = 42
    
    ## load volume data
    data_tps = ["t1", "t2", "t3", "t4", "t5", "t6"]
    rano_cols = [elem+'_rano' for elem in data_tps]
    data_cols = [elem+'_volume' for elem in data_tps]
    print(f'clustering {len(complete_data)} metastases')
    
    # Normalize by t0 Volume
    complete_data[data_cols] = complete_data[data_cols].div(complete_data["t0_volume"], axis=0)

    if use_derivatives:
        derivatives = get_derivatives(complete_data[["t0_volume"]+data_cols], 'sobel', 'constant', 'relative')

        complete_data = pd.concat([complete_data, derivatives], axis=1)

        derivative_cols = list(derivatives.columns)

    ##### ------------------ MAKE CHANGES HERE -------------------------------------
    fmt = wide_df_to_3d_np(copy.deepcopy(complete_data[data_cols]))

    if use_derivatives:
        drv = wide_df_to_3d_np(copy.deepcopy(derivatives))

        fmt = np.concatenate((fmt, drv), axis=-1)


    best_cluster = TimeSeriesXMeans(k_max=k, metric='dtw', random_seed=42)
    best_cluster, best_k = best_cluster.fit(fmt)

    best_bic, best_aic = good_approx_bic_aic(complete_data[data_cols], best_cluster.labels_, best_k)
    
    ##### ------------------ MAKE CHANGES HERE -------------------------------------


    labels = best_cluster.labels_

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
    # ## Make plots for dimesnionality reduction
    plot_umap(complete_data, data_cols, output)

    # ## Make plot for cluster trajectories
    plot_combined_trajectories(filtered_data, data_cols, 'cluster', output)

    ## Make plot for recur prob
    plot_recur_probs(filtered_data, rano_cols, 'cluster', output)

    ## make plot for recur probs but allow any count, not just from t1
    plot_recur_probs_noncausal(filtered_data, rano_cols, output)




