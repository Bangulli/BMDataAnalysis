import pandas as pd
import sklearn
from sklearn.cluster import MeanShift, KMeans, DBSCAN, HDBSCAN, OPTICS

import pathlib as pl
from data import get_derivatives, load_prepro_data
import numpy as np
import os
from visualization import *
from clustering import *
from stepmix.stepmix import StepMix

import warnings
warnings.filterwarnings('ignore')

if __name__ == '__main__':
    method_name = 'quantiles'
    folder_name = 'clusters'
    basedir = '/mnt/nas6/data/Target/BMPipeline_full_rerun/PARSED_METS_task_502'
    use_derivatives = False
    

    train, test = load_prepro_data(pl.Path(f'/mnt/nas6/data/Target/BMPipeline_full_rerun/229_patients faulty/PARSED_METS_task_502/csv_nn/features.csv'),
                                    used_features=['volume'],
                                    test_size=0.2,
                                    drop_suffix=None,
                                    prefixes=["t0", "t1", "t2", "t3", "t4", "t5", "t6"],
                                    target_suffix='rano',
                                    normalize_suffix=None,
                                    rano_encoding=None,
                                    time_required=False,
                                    interpolate_CR_swing_length=1,
                                    drop_CR_swing_length=2,
                                    normalize_volume=None,
                                    add_index_as_col = True,
                                    save_processed=None)#pl.Path(f'/mnt/nas6/data/Target/BMPipeline_full_rerun/229_patients faulty/PARSED_METS_task_502/csv_nn/clustering_prerpocessed.csv'))
    
    all_data = pd.concat((train, test), axis=0)
    

    quantiles = all_data['t0_volume'].quantile([0.25, 0.5, 0.75, 1.0])
    quantiles = quantiles.values  # Convert to array for index-based access
    subsets = {}
    for i, q in enumerate(quantiles):
        if i == 0:
            cur = all_data.loc[all_data['t0_volume'] <= q, 'cluster'] = i
        elif i == len(quantiles) - 1:
            cur = all_data.loc[all_data['t0_volume'] > quantiles[i-1], 'cluster'] = i
        else:
            lower = quantiles[i - 1]
            upper = q
            cur = all_data.loc[(all_data['t0_volume'] > lower) & (all_data['t0_volume'] <= upper), 'cluster'] = i

    output =  pl.Path(f'/home/lorenz/BMDataAnalysis/output/{folder_name}/{method_name} n_clusters')
    complete_data = all_data

    
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

    
    filtered_data, invalid_labels = filter_small_clusters(complete_data, 'cluster', 2)

    DB_score = sklearn.metrics.davies_bouldin_score(filtered_data[data_cols], filtered_data['cluster'])
    print(f'Clustering achieved a Davies-Bouldin score of {DB_score}')

    if use_derivatives:
        output = output.parent.parent/(output.parent.name+'_deriv')/output.name
        data_cols = ["t1_volume", "t2_volume", "t3_volume", "t4_volume", "t5_volume", "t6_volume"] # overwrite the columns so it plots just the trajectory not the derivatives

    output = output.parent/(output.name+str(len(quantiles)))

    # ## Make plots for cluster
    plot_cluster_centers(filtered_data, output, ["t0_volume", "t1_volume", "t2_volume", "t3_volume", "t4_volume", "t5_volume", "t6_volume"], rano_cols, label_col='cluster', init_col='t0_volume')

    # ## load rano data
    plot_sankey(filtered_data[rano_cols], output)

    # ## Make plots for dimesnionality reduction
    mapping = {l: (-1 if l in invalid_labels else l) for l in np.unique(complete_data["cluster"])}
    complete_data['cluster'] = complete_data['cluster'].map(mapping)
    plot_umap(complete_data, ["t0_volume", "t1_volume", "t2_volume", "t3_volume", "t4_volume", "t5_volume", "t6_volume"], output)

    # ## Make plot for cluster trajectories
    plot_combined_trajectories(filtered_data, ["t1_volume", "t2_volume", "t3_volume", "t4_volume", "t5_volume", "t6_volume"], 'cluster', output)

    ## Make plot for recur prob
    plot_recur_probs(filtered_data, rano_cols, 'cluster', output)

    ## make plot for recur probs but allow any count, not just from t1
    plot_recur_probs_noncausal(filtered_data, rano_cols, output)




