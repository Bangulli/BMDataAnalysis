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
    method_name = 'tsxmeans'
    folder_name = 'csv_linear_multiclass_reseg_only_valid'
    use_derivatives = False
    volume_data = pd.read_csv(f'/mnt/nas6/data/Target/task_524-504_PARSED_METS_mrct1000_nobatch/{folder_name}/volumes.csv', index_col=None)
    rano_data = pd.read_csv(f'/mnt/nas6/data/Target/task_524-504_PARSED_METS_mrct1000_nobatch/{folder_name}/rano.csv', index_col=None)
    renamer = {elem: 'rano-'+elem for elem in rano_data.columns}
    rano_data = rano_data.rename(columns=renamer)

    complete_data = pd.concat([volume_data, rano_data], axis=1)
    
    output =  pl.Path(f'/home/lorenz/BMDataAnalysis/output/{folder_name}/{method_name}_')

    k = 42
    
    ## load volume data
    data_cols = ["60", "120", "180", "240", "300", "360"]
    rano_cols = ['rano-'+elem for elem in data_cols]
    print(f'clustering {len(complete_data)} metastases')
    
    # Normalize by t0 Volume
    complete_data[data_cols] = complete_data[data_cols].div(complete_data["0"], axis=0)

    if use_derivatives:
        derivatives = get_derivatives(complete_data[["0"]+data_cols], 'sobel', 'constant', 'relative')

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
        data_cols = ["60", "120", "180", "240", "300", "360"] # overwrite the columns so it plots just the trajectory not the derivatives

    output = output.parent/(output.name+str(best_k))

    # ## Make plots for cluster
    plot_cluster_centers(filtered_data, output, data_cols, rano_cols, label_col='cluster', init_col='0')

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




