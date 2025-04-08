import pandas as pd
import sklearn
from sklearn.cluster import MeanShift, KMeans, DBSCAN, HDBSCAN, OPTICS
from sktime.datatypes import convert_to
import pathlib as pl
from sktime.clustering.k_shapes import TimeSeriesKShapes
import numpy as np
import os
import copy
from visualization import *
from clustering import *
from stepmix.stepmix import StepMix

import warnings
warnings.filterwarnings('ignore')

if __name__ == '__main__':
    method_name = 'tsxshapes'
    folder_name = 'csv_linear_multiclass_reseg_only_valid'
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

    ##### ------------------ MAKE CHANGES HERE -------------------------------------
    fmt = wide_df_to_3d_np(copy.deepcopy(complete_data[data_cols]))
    best_cluster = TimeSeriesXShapes(k_max=k)
    best_cluster, best_k = best_cluster.fit(fmt)

    best_bic, best_aic = good_approx_bic_aic(complete_data[data_cols], best_cluster.labels_, best_k)
    
    ##### ------------------ MAKE CHANGES HERE -------------------------------------


    labels = best_cluster.labels_

    complete_data['cluster'] = labels
    print(f'Best clustering achieved an AIC of {best_aic} and a BIC of {best_bic} with {best_k} clusters')
    
    complete_data = filter_small_clusters(complete_data, 'cluster', 15)

    DB_score = sklearn.metrics.davies_bouldin_score(complete_data[data_cols], complete_data['cluster'])
    print(f'Clustering achieved a Davies-Bouldin score of {DB_score}')

    output = output.parent/(output.name+str(best_k))

    # ## Make plots for cluster
    plot_cluster_centers(complete_data, output, data_cols, rano_cols, label_col='cluster', init_col='0')

    # ## load rano data
    plot_sankey(complete_data[rano_cols], output)

    # ## Make plots for dimesnionality reduction
    plot_umap(complete_data, data_cols, output)

    # ## Make plot for cluster trajectories
    plot_combined_trajectories(complete_data, data_cols, 'cluster', output)

    ## Make plot for recur prob
    plot_recur_probs(complete_data, rano_cols, 'cluster', output)

    ## make plot for recur probs but allow any count, not just from t1
    plot_recur_probs_noncausal(complete_data, rano_cols, output)




