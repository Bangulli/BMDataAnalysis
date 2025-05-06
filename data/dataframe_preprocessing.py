import pandas as pd
from scipy.stats import zscore
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from .feature_eliminators import FeatureCorrelationEliminator, LASSOFeatureEliminator, ModelFeatureEliminator
from sklearn.model_selection import train_test_split

"""
## normalize all volumes by the cubic root of the fraction
            if normalize_volume == 'frac+3root':
                print('Normalizing time series volume as the cubic rood for the fraction TX/T0')

                ## compute cubic root of the fraction for follow ups
                df[volume_cols[1:]] = df[volume_cols[1:]].div(df[volume_cols[0]], axis=0)**(1/3) # normalize follow up volume
                df['ignored_vol_normalizer'] = ["np.exp({})*"+str(f) for f in df[volume_cols[0]]] # "np.exp({})*factor"

                ## norm t0 by standardizing the cubic root
                df[volume_cols[0]] = df[volume_cols[0]]**(1/3)
                std = df[volume_cols[0]].std()
                mean = df[volume_cols[0]].mean()   
                df[volume_cols[0]] = (df[volume_cols[0]]-mean).div(std) # normalize init volume

                # on testset
                test[volume_cols[1:]] = test[volume_cols[1:]].div(test[volume_cols[0]], axis=0)**(1/3) # normalize follow up volume
                test['ignored_vol_normalizer'] = ["np.exp({})*"+str(f) for f in test[volume_cols[0]]] # "np.exp({})*factor"
                test[volume_cols[0]] = (test[volume_cols[0]]**(1/3)-mean).div(std) # normalize init volume

            elif normalize_volume == '3root+std':
                print('Normalizing time series volume by standardizing the cubic root of the absolute volume')
                df[volume_cols] = df[volume_cols] ** (1/3)
                std = df[volume_cols].values.std()
                mean = df[volume_cols].values.mean()
                df[volume_cols] = (df[volume_cols]-mean).div(std)
                df['ignored_vol_normalizer'] = ["({}*"+str(std)+"+"+str(mean)+")**3"]*len(df)

                # on testset
                test[volume_cols] = (test[volume_cols]-mean).div(std)
                test['ignored_vol_normalizer'] = ["({}*"+str(std)+"+"+str(mean)+")**3"]*len(test)
                    


                """

def load_prepro_data(path, 
                     used_features, # tells the method which feature classes to include in the output, for example ['radiomics', 'volume'] will produce a dataframe where each time point has radiomics features and volume
                     test_size=0.2, # controls the test set size for splitting
                     drop_suffix=None, # a list of strings, for each timepoint, the column that has a suffix listed here are dropped, for example ['_radiomics_featrue_1'] will drop t0_radiomics_feature_1, t1..... and so on
                     prefixes=["t0", "t1", "t2", "t3", "t4", "t5", "t6"], # the list of timepoint prefixes to include, only timepoints listed here are included
                     target_suffix='rano', # each timepoint has a target rano class, that is included in columns with this suffix
                     fill=0, # value to fill NAs with
                     normalize_suffix=['radiomics'], # feature classes that need are normalized with the col_normalization fucntion
                     rano_encoding={'CR':0, 'PR':1, 'SD':2, 'PD':3},  # encoding for rano classes from string to categorical
                     time_required=False, # if true will include timedelta in the output, used for graphs. regular prediction may not need this
                     normalize_volume=True): # wheter or not to normalize the volume and if so what kind of normalization to use: True=norm values by t0, False=No norm, '3root' use cubic rood
    
    df = pd.read_csv(path, index_col='Lesion ID') # load
    df.fillna(fill, inplace=True) # fill

    ## encode rano classes
    for col in [c for c in df.columns if c.endswith(target_suffix)]:
        df[col] = df[col].map(rano_encoding)

    ## dataset splitting
    labels = [d[f'{prefixes[-1]}_{target_suffix}'] for i, d in df.iterrows()]
    df, test = train_test_split(df, test_size=test_size, random_state=42, stratify=labels)

    # normalize volume feature
    if normalize_volume:
        volume_cols = [c for c in df.columns if c.endswith('_volume')]
        for step, mode in enumerate(normalize_volume.split('->')):
            print(f"working on step {step}: {mode}")
            ## standardize
            if mode == 'std':
                std = df[volume_cols].values.std()
                mean = df[volume_cols].values.mean()
                df[volume_cols] = (df[volume_cols]-mean).div(std)
                fmt = "({}*"+str(std)+"+"+str(mean)+")"

                # on testset
                test[volume_cols] = (test[volume_cols]-mean).div(std)

            ## normalize all volumes by the same factor
            elif mode == 'max':
                factor = df[volume_cols].max().max()
                print(f'Normalizing time series volume by factor {factor}')
                df[volume_cols] = df[volume_cols].div(factor)# normalize follow up volume
                fmt="({}*"+str(factor)+")"

                # on testset
                test[volume_cols] = test[volume_cols].div(factor)# normalize follow up volume
                
            ## normalize asll volumes by the whole brain volume of the respective series
            elif mode == 'log':
                print('normalizing time series volume its log')
                df[volume_cols] = np.log1p(df[volume_cols])
                fmt="np.exp({})"

                # on testset
                test[volume_cols] =  np.log1p(test[volume_cols])

            
            ## normalize time series volume by the baseline volume
            elif mode == 'frac':
                print('Normalizing time series volume by baseline volume for each series')
                train_fmt = ["({}*"+str(f)+")" for f in df[volume_cols[0]].tolist()]
                df[volume_cols] = df[volume_cols].div(df[volume_cols[0]], axis=0) # normalize follow up volume
                # on testset
                fmt = ["({}*"+str(f)+")" for f in test[volume_cols[0]].tolist()] # generate reverter first before norming
                test[volume_cols] = test[volume_cols].div(test[volume_cols[0]], axis=0) # normalize follow up volume
                
            
            elif mode == '3root':
                print('Normalizing time series volume by its cubic root')
                df[volume_cols] = df[volume_cols]**(1/3) # normalize follow up volume
                fmt = "({}**3)"

                # on testset
                test[volume_cols] = test[volume_cols]**(1/3) # normalize follow up volume

        
            else: raise ValueError(f"Unrecognized volume normalization mode {normalize_volume}")

            ## accumulate decode functions
            if step == 0:
                if isinstance(fmt, list):
                    test['ignored_vol_normalizer'] = fmt
                    df['ignored_vol_normalizer'] = train_fmt

                else:
                    test['ignored_vol_normalizer'] = [fmt]*len(test)
                    df['ignored_vol_normalizer'] = [fmt]*len(df)
         
            else:
                
                if isinstance(fmt, list):
            
                    for j, id in enumerate(test.index):
                        test.loc[id, 'ignored_vol_normalizer'] = test.loc[id, 'ignored_vol_normalizer'].format(fmt[j])
                 
  
                    for j, id in enumerate(df.index):
                        df.loc[id, 'ignored_vol_normalizer'] = df.loc[id, 'ignored_vol_normalizer'].format(train_fmt[j])
               
                else:
            
                    for j, id in enumerate(test.index):
                        test.loc[id, 'ignored_vol_normalizer'] = test.loc[id, 'ignored_vol_normalizer'].format(fmt)
                  
              
                    for j, id in enumerate(df.index):
                        df.loc[id, 'ignored_vol_normalizer'] = df.loc[id, 'ignored_vol_normalizer'].format(fmt)
                    


        

    ## nromalize radiomics 
    if normalize_suffix is not None:
        for sfx in normalize_suffix:
            for tp in prefixes:
                for col in df.columns:
                    if col.startswith(f"{tp}_{sfx}"):
                        df[col] = pd.to_numeric(df[col], errors='coerce')
                        test[col] = pd.to_numeric(test[col], errors='coerce')
                        std = df[col].std()
                        mean = df[col].mean()   
                        df[col]=(df[col]-mean).div(std) # try to parse every value to floats
                        test[col]=(test[col]-mean).div(std)
    
    ## drop unused features
    to_keep = []
    used_features = [*used_features, 'timedelta_days'] if time_required else used_features
    for col in df.columns:
        for tp in prefixes:
            for feature in used_features:
                if (col.startswith('ignored')):
                    to_keep.append(col)
                if (col.startswith(f"{tp}_{feature}") or col.endswith(target_suffix)) and col not in to_keep:
                    to_keep.append(col)
    ([df.drop(columns=c, inplace=True, axis=0) for c in df.columns if c not in to_keep])

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

    train = df.drop(columns=to_drop)
    test = test.drop(columns=to_drop)

    print('Training Set')
    print(train.describe())
    train.info()

    print('Testing Set')
    print(test.describe())
    test.info()
    
    
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
    d, _ = load_prepro_data('/mnt/nas6/data/Target/BMPipeline_DEVELOPMENT_runs/task_502_PARSED_METS_mrct1000_nobatch/csv_nn_only_valid/features.csv',
                         drop_suffix=None, used_features=['volume'], normalize_volume='factor')
    print(d)
    d.info()
    print(d.describe())

    for i, row in d.iterrows():
        print(row.name)