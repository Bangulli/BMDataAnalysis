import pandas as pd
from scipy.stats import zscore
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from .feature_eliminators import FeatureCorrelationEliminator, LASSOFeatureEliminator, ModelFeatureEliminator
from sklearn.model_selection import train_test_split

def load_prepro_data(path, used_features, test_size=0.2, drop_suffix=None, col_normalization=zscore, prefixes=["t0", "t1", "t2", "t3", "t4", "t5", "t6"], target_suffix='rano', fill=0, normalize_suffix=['radiomics'], rano_encoding={'CR':0, 'PR':1, 'SD':2, 'PD':3}, time_required=False, normalize_volume=True):
    df = pd.read_csv(path, index_col='Lesion ID') # load
    df.fillna(fill, inplace=True) # fill

    # static data preprocessing
    if normalize_volume:
        volume_cols = [c for c in df.columns if c.endswith('_volume')]
        df[volume_cols[1:]] = df[volume_cols[1:]].div(df[volume_cols[0]], axis=0) # normalize follow up volume
        df[volume_cols[0]] = col_normalization(df[volume_cols[0]]) # normalize init volume

    ## nromalize radiomics 
    for sfx in normalize_suffix:
        for tp in prefixes:
            for col in df.columns:
                if col.startswith(f"{tp}_{sfx}"):
                    df[col]=col_normalization(pd.to_numeric(df[col], errors='coerce')) # try to parse every value to floats
    
    ## drop unused features
    to_keep = []
    used_features = [*used_features, 'timedelta_days'] if time_required else used_features
    for col in df.columns:
        for tp in prefixes:
            for feature in used_features:
                if (col.startswith(f"{tp}_{feature}") or col.endswith(target_suffix)) and col not in to_keep:
                    to_keep.append(col)
    ([df.drop(columns=c, inplace=True, axis=0) for c in df.columns if c not in to_keep])

    ## dataset splitting
    labels = [d[f'{prefixes[-1]}_{target_suffix}'] for i, d in df.iterrows()]
    train, test = train_test_split(df, test_size=test_size, random_state=42, stratify=labels)

    to_drop = [] # to avoid variable not assigned error
    if isinstance(drop_suffix, list):
        # drop ignored cols
        for drp in drop_suffix:
            to_drop = [c for c in df.columns if c.endswith(drp)]
    elif drop_suffix == 'infer':
        print('Automatically selecting features using lasso')
        drop_suffix = feature_selection(train, target_suffix, prefixes, LASSOFeatureEliminator(), rano_encoding, time_required)
        if drop_suffix:
            for drp in drop_suffix:
                to_drop = [c for c in df.columns if c.endswith(drp)]
            print(f'Removed feature suffixes {drop_suffix} from data')
        else:
            print('Automatic selection did not yield any features to drop.')
    elif callable(drop_suffix):
        print('Automatically selecting features using passed callable function')
        drop_suffix = feature_selection(train, target_suffix, prefixes, drop_suffix, rano_encoding, time_required)
        if drop_suffix:
            for drp in drop_suffix:
                to_drop = [c for c in df.columns if c.endswith(drp)]
            print(f'Removed feature suffixes {drop_suffix} from data')
        else:
            print('Automatic selection did not yield any features to drop.')
    else:
        print("No features are ignored")

    train = train.drop(columns=to_drop)
    test = test.drop(columns=to_drop)
    
    
    return train, test

def feature_selection(data, target_suffix, prefixes, eliminator=FeatureCorrelationEliminator(0.9), rano_encoding = {'CR':0, 'PR':1, 'SD':2, 'PD':3}, time_required=True):
    ### organize the dataframe such that it ignores timepoints
    ### this is done so that we select features on a timepoint level not globally
    tp_df = None
    for i, tp in enumerate(prefixes):
        if i == len(prefixes)-1: # skip the last timepoint, cause this is the one we want to predict
            continue
        tp_cols = [c for c in data.columns if c.startswith(tp)]

        # overwrite the rano col with the rano of the next timepoint so the target is always in the future
        for t_c in range(len(tp_cols)):
            if tp_cols[t_c] == f"{tp}_rano":
                tp_cols[t_c] == f"{prefixes[i+1]}_rano"

        ## unify column names and concatenate dataframe slices to unified tp dataframe
        if tp_df is None:
            tp_df = data[tp_cols]
            renamer = {c:c[3:] for c in tp_df.columns}
            tp_df.rename(columns=renamer, inplace=True)
        else:
            tp_df = pd.concat(
                (tp_df, 
                 data[tp_cols].rename(
                     columns={c:c[3:] for c in data[tp_cols].columns}
                     )
                 ), 
                ignore_index=True, 
                axis=0
                )

    ### slice into target and data dfs
    tp_data = tp_df[[c for c in tp_df.columns if c != target_suffix]]

    if time_required: tp_data.drop(columns='timedelta_days', inplace=True) ## keep the time. its necessary for the parsing to graph so we cant drop it. only used in GNN 
    
    for c in tp_data.columns:
        tp_data[c] = pd.to_numeric(tp_data[c], errors='coerce')

    tp_target = tp_df[target_suffix].map(rano_encoding)

    if isinstance(eliminator, list):
        to_drop = []
        for elim in eliminator:
            to_drop += elim(tp_data.drop(to_drop), tp_target)
    else: to_drop = eliminator(tp_data, tp_target)

    return to_drop




if __name__ == '__main__':
    d = load_prepro_data('/mnt/nas6/data/Target/BMPipeline_DEVELOPMENT_runs/task_502_PARSED_METS_mrct1000_nobatch/csv_nn_only_valid/features.csv',
                         drop_suffix='infer')