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

import warnings
warnings.filterwarnings('ignore')

if __name__ == '__main__':
    method_name = 'stepmix_exp'
    folder_name = 'clusters'
    use_derivatives = False
    split_by_vol = False # can be false for no splitting, 'quantile' for quantile splitting, 'meanshift' for using a meanshift to split
    discard = ['sub-PAT0122:1', 
           'sub-PAT0167:0', 
           'sub-PAT0182:2', 
           'sub-PAT0342:0', 
           'sub-PAT0411:0', 
           'sub-PAT0434:6', 
           'sub-PAT0434:9', 
           'sub-PAT0434:10', 
           'sub-PAT0434:11', 
           'sub-PAT0480:20', 
           'sub-PAT0484:4', 
           'sub-PAT0490:0', 
           'sub-PAT0612:2', 
           'sub-PAT0666:0', 
           'sub-PAT0756:0', 
           'sub-PAT1028:3',
           'sub-PAT0045:6',
           'sub-PAT0105:0',
           'sub-PAT0441:0', 
           'sub-PAT0686:1',
           'sub-PAT0807:3',
           ]
    train, test = load_prepro_data(pl.Path(f'/mnt/nas6/data/Target/BMPipeline_full_rerun/PARSED_METS_task_502/final_extraction/all_features_nn.csv'),
                                    used_features=['volume'],
                                    discard=discard,
                                    test_size=None,
                                    drop_suffix=None,
                                    prefixes=["t0", "t1", "t2", "t3", "t4", "t5", "t6"],
                                    target_suffix='rano',
                                    categorical=[],
                                    normalize_suffix=None,
                                    rano_encoding=None,
                                    time_required=False,
                                    normalize_volume=None,
                                    add_index_as_col = True,
                                    save_processed=None)#pl.Path(f'/mnt/nas6/data/Target/BMPipeline_full_rerun/PARSED_METS_task_502/csv_nn/clustering_prerpocessed.csv'))
    
    all_data = pd.concat((train, test), axis=0)
    
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
        output =  pl.Path(f'/home/lorenz/BMDataAnalysis/MICCAI_submission/{folder_name}/{method_name}_{tag} n_clusters')
        
        k = 5#range(2,42)
        
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
        
        filtered_data, invalid_labels = filter_small_clusters(complete_data, 'cluster', 15)#filter_small_clusters_by_samplestats(complete_data, 'cluster')#

        DB_score = sklearn.metrics.davies_bouldin_score(filtered_data[data_cols], filtered_data['cluster'])
        print(f'Clustering achieved a Davies-Bouldin score of {DB_score}')

        if use_derivatives:
            output = output.parent.parent/(output.parent.name+'_deriv')/output.name
            data_cols = ["t1_volume", "t2_volume", "t3_volume", "t4_volume", "t5_volume", "t6_volume"] # overwrite the columns so it plots just the trajectory not the derivatives

        output = output.parent/(output.name+str(best_k))
        to_save = complete_data[['Lesion ID', 'cluster']]
        
        # ## Make plots for cluster
        plot_cluster_centers(filtered_data, output, ["t0_volume", "t1_volume", "t2_volume", "t3_volume", "t4_volume", "t5_volume", "t6_volume"], rano_cols, label_col='cluster', init_col='t0_volume')
        to_save.to_csv(output/'assignments.csv')
        # ## load rano data
        plot_sankey(filtered_data[rano_cols], output)

        # ## Make plots for dimesnionality reduction
        # mapping = {l: (-1 if l in invalid_labels else l) for l in np.unique(labels)}
        # complete_data['cluster'] = complete_data['cluster'].map(mapping)
        # plot_umap(complete_data, ["t0_volume", "t1_volume", "t2_volume", "t3_volume", "t4_volume", "t5_volume", "t6_volume"], output)

        # ## Make plot for cluster trajectories
        plot_combined_trajectories(filtered_data, ["t1_volume", "t2_volume", "t3_volume", "t4_volume", "t5_volume", "t6_volume"], 'cluster', output)

        # ## Make plot for recur prob
        # plot_recur_probs(filtered_data, rano_cols, 'cluster', output)

        # ## make plot for recur probs but allow any count, not just from t1
        # plot_recur_probs_noncausal(filtered_data, rano_cols, output)




