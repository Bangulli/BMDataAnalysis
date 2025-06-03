import sys
from pathlib import Path

# Add parent directory to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))
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
import copy
import warnings
warnings.filterwarnings('ignore')



if __name__ == '__main__':
    method_name = 'tsxshapes'
    folder_name = 'clusters'
    use_derivatives = False
    split_by_vol = False
    all_data, test = load_prepro_data(pl.Path(f'/mnt/nas6/data/Target/BMPipeline_full_rerun/PARSED_METS_task_502/final_extraction/all_features_nn.csv'),
                                used_features=['volume'],
                                test_size=None,
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
                                save_processed=None)#pl.Path(f'/mnt/nas6/data/Target/BMPipeline_full_rerun/PARSED_METS_task_502/csv_nn/clustering_prerpocessed.csv'))
    
    output =  pl.Path(f'/home/lorenz/BMDataAnalysis/final_output/{folder_name}/{method_name}_')

    k = 42
    
    if not split_by_vol: subsets={f'all n_samples{len(all_data)}':all_data} # use all data
    elif split_by_vol=='meanshift':
        sub_clusterer = MeanShift()
        sub_clusterer.fit(all_data[['t0_volume']])
        means = sub_clusterer.cluster_centers_
        labels = sub_clusterer.labels_
        subsets = {}
        for i, mean in enumerate(means):
            binary = labels==i
            subsets[f"{i}th subset mean{mean} n_samples{sum(binary)}"] = all_data.iloc[binary]
    
    ## split into subsets according to 
    elif split_by_vol == 'quantile':
        quantiles = all_data['t0_volume'].quantile([0.25, 0.5, 0.75, 1.0])
        quantiles = quantiles.values  # Convert to array for index-based access
        subsets = {}
        for i, q in enumerate(quantiles):
            if i == 0:
                cur = all_data[all_data['t0_volume'] <= q]
                subsets[f"{i+1}st quantile {q:.2f} n_samples{len(cur)}"] = cur 
            elif i == len(quantiles) - 1:
                cur = all_data[all_data['t0_volume'] > quantiles[i-1]]
                subsets[f"{i+1}th quantile {quantiles[i-1]:.2f} n_samples{len(cur)}"] = cur
            else:
                lower = quantiles[i - 1]
                upper = q
                cur = all_data[(all_data['t0_volume'] > lower) & (all_data['t0_volume'] <= upper)]
                subsets[f"{i+1}th quantile ({lower:.2f} - {upper:.2f}) n_samples{len(cur)}"] = cur

    else: subsets= {f'all n_samples{len(all_data)}':all_data} # use all data

    for tag, complete_data in subsets.items():
        output =  pl.Path(f'/home/lorenz/BMDataAnalysis/output/{folder_name}/{method_name}_{tag} n_clusters')

        k = 42
        
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

            derivative_cols = list(derivatives.columns)

        ##### ------------------ MAKE CHANGES HERE -------------------------------------
        fmt = wide_df_to_3d_np(copy.deepcopy(complete_data[data_cols]))

        if use_derivatives:
            
            drv = wide_df_to_3d_np(copy.deepcopy(derivatives))

            fmt = np.concatenate((fmt, drv), axis=-1)

        best_cluster = TimeSeriesXShapes(k_max=k)
        best_cluster, best_k = best_cluster.fit(fmt)

        best_bic, best_aic = good_approx_bic_aic(complete_data[data_cols], best_cluster.labels_, best_k)
        
        ##### ------------------ MAKE CHANGES HERE -------------------------------------


        labels = best_cluster.labels_

        complete_data['cluster'] = labels
        print(f'Best clustering achieved an AIC of {best_aic} and a BIC of {best_bic} with {best_k} clusters')
        
        filtered_data, invalid_labels = filter_small_clusters(complete_data, 'cluster', 15)

        DB_score = sklearn.metrics.davies_bouldin_score(complete_data[data_cols], complete_data['cluster'])
        print(f'Clustering achieved a Davies-Bouldin score of {DB_score}')

        if use_derivatives:
            output = output.parent.parent/(output.parent.name+'_deriv')/output.name
            data_cols = ["t1_volume", "t2_volume", "t3_volume", "t4_volume", "t5_volume", "t6_volume"] # overwrite the columns so it plots just the trajectory not the derivatives

        output = output.parent/(output.name+str(best_k))

        # ## Make plots for cluster
        plot_cluster_centers(filtered_data, output, ["t0_volume", "t1_volume", "t2_volume", "t3_volume", "t4_volume", "t5_volume", "t6_volume"], rano_cols, label_col='cluster', init_col='t0_volume')

        # ## load rano data
        plot_sankey(filtered_data[rano_cols], output)

        # ## Make plots for dimesnionality reduction
        mapping = {l: (-1 if l in invalid_labels else l) for l in np.unique(labels)}
        complete_data['cluster'] = complete_data['cluster'].map(mapping)
        plot_umap(complete_data, ["t0_volume", "t1_volume", "t2_volume", "t3_volume", "t4_volume", "t5_volume", "t6_volume"], output)

        # ## Make plot for cluster trajectories
        plot_combined_trajectories(filtered_data, ["t1_volume", "t2_volume", "t3_volume", "t4_volume", "t5_volume", "t6_volume"], 'cluster', output)

        ## Make plot for recur prob
        plot_recur_probs(filtered_data, rano_cols, 'cluster', output)

        ## make plot for recur probs but allow any count, not just from t1
        plot_recur_probs_noncausal(filtered_data, rano_cols, output)




