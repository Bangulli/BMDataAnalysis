import sys
from pathlib import Path

# Add parent directory to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))
import pandas as pd
from scipy.stats import zscore
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from data.feature_eliminators import FeatureCorrelationEliminator, LASSOFeatureEliminator, ModelFeatureEliminator
from sklearn.model_selection import train_test_split
from visualization.clustering import plot_sankey 
import pathlib as pl
import copy

def load_prepro_data(path, 
                     used_features, # tells the method which feature classes to include in the output, for example ['radiomics', 'volume'] will produce a dataframe where each time point has radiomics features and volume
                     discard=None,
                     categorical=[], # a list of columns with categorical features. Encoding is infered at runtime
                     test_size=0.2, # controls the test set size for splitting
                     drop_suffix=None, # a list of strings, for each timepoint, the column that has a suffix listed here are dropped, for example ['_radiomics_featrue_1'] will drop t0_radiomics_feature_1, t1..... and so on
                     prefixes=["t0", "t1", "t2", "t3", "t4", "t5", "t6"], # the list of timepoint prefixes to include, only timepoints listed here are included
                     target_suffix='rano', # each timepoint has a target rano class, that is included in columns with this suffix
                     fill=0, # value to fill NAs with if None will be skipped
                     normalize_suffix=['radiomics'], # feature classes that need are normalized with the col_normalization fucntion
                     rano_encoding={'CR':0, 'PR':1, 'SD':2, 'PD':3},  # encoding for rano classes from string to categorical
                     drop_CR_swing_length = 2, # if there is a swing to CR in the rano assignment of at least this lenght the row will be discarded
                     interpolate_CR_swing_length = 1, # if there is a swing to CR in the rano assignment of at most this length, the values will be interpolated
                     time_required=False, # if true will include timedelta in the output, used for graphs. regular prediction may not need this
                     normalize_volume=None, # wheter or not to normalize the volume and if so what kind of normalization to use: True=norm values by t0, False=No norm, '3root' use cubic rood
                     save_processed = None, # saves preprocessed data to here if exists will load and return
                     outlier_detection_factor = 5, # if a lesion shrinks or grows by more than this factor in between two neighboring timepoints during its observation it is regarded as an outlier and dropped
                     add_index_as_col=False): 
    """
    Main data loading and preprocessing function
    It has multiple stages:
    load data, preprocessed if the path provided in save_preprocessed is already an existing file
    fill missing if fill is not None, uses the variable fill as the value to fill, typically 0
    store init volume in case the normalization destroys the value for later usage if 'init_volume' is in the used features list
    outlier rejection and interpolation based on swings to CR and dips/spikes in volume, CR swings of len >= drop_CR_swing_length are dropped, CR swings of len < interpolate_CR_swing_length are interpolated, spikes to cur/prev > outlier_detection_factor or post/cur < 1/outlier_detection_factor are dropped 
    the volume is encoded based on the value passed in normalize_volume, can be None = no normalization, or any combination of options listed in the big if elif else block combined by 'step1->step2->...->stepN' will add a column with a string called ignored_vol_normalizer. if you call eval(str.format(pred)) will give you the real world volume by reversing the normalization encoding


    """
    if save_processed is not None:
        if (save_processed.parent/('train_'+save_processed.name)).is_file(): 
            print('found preprocessed dataframes, loading tem instead of doing all the shabang again')
            if test_size is not None: 
                tr = pd.read_csv(save_processed.parent/('train_'+save_processed.name), index_col='Lesion ID')
                tr['Lesion ID'] = tr.index
                tr.drop(columns='Lesion ID.1', inplace=True, errors='ignore')
                return tr, pd.read_csv(save_processed.parent/('test_'+save_processed.name))
            else: 
                tr = pd.read_csv(save_processed.parent/('train_'+save_processed.name), index_col='Lesion ID')
                tr['Lesion ID'] = tr.index
                tr.drop(columns='Lesion ID.1', inplace=True, errors='ignore')
                return tr, None

    df = pd.read_csv(path, index_col='Lesion ID') # load
    if discard is not None: 
        df.drop(index=discard, inplace=True, axis=1)
        print("dropping indices:", discard)
    if fill is not None: df.fillna(fill, inplace=True) # fill
    if 'init_volume' in used_features: df['init_volume'] = df['t0_volume'].copy(deep=True)



    ## detect outliers and interpolate missing data
    # im sorry this is horrible but it works
    # df, rano_cols = sample_rejection(df, interpolate_CR_swing_length, outlier_detection_factor, prefixes, drop_CR_swing_length)
    # plot_sankey(df[rano_cols], pl.Path(''))

    ## encode rano classes
    if rano_encoding is not None:
        for col in [c for c in df.columns if c.endswith(target_suffix)]:
            df[col] = df[col].map(rano_encoding)

    ## encode categorical features:
    if categorical:
        df = encode_categorical(df, categorical)

    ## dataset splitting
    labels = [d[f'{prefixes[-1]}_{target_suffix}'] for i, d in df.iterrows()]
    if test_size is not None: df, test = train_test_split(df, test_size=test_size, random_state=42, stratify=labels)
    else: test=None

    ## normalize volume feature
    if normalize_volume:
        df, test = apply_volume_normalization(df, test, prefixes, normalize_volume, test_size)
                           

    ## nromalize radiomics 
    df, test = normalize_other(df, test, normalize_suffix, prefixes, categorical, test_size, fill)
    
    ## drop unused feature types
    to_keep = []
    used_features = [*used_features, 'timedelta_days'] if time_required else used_features
    for col in df.columns:
        for tp in prefixes:
            for feature in used_features:
                if (col.startswith('ignored')):
                    to_keep.append(col)
                if (col.startswith(f"{tp}_{feature}") or col.endswith(target_suffix)) and col not in to_keep:
                    to_keep.append(col)
                if col in used_features:
                    to_keep.append(col)
                if col.startswith(feature):
                    to_keep.append(col)
    to_rm = [c for c in df.columns if c not in to_keep]
    df.drop(columns=to_rm, inplace=True, axis=0)
    if test_size is not None: test.drop(columns=to_rm, inplace=True, axis=0)

    ## drop rejected features
    to_drop = [] # to avoid variable not assigned error
    if isinstance(drop_suffix, list):
        # drop ignored cols
        to_drop = []
        for drp in drop_suffix:
            to_drop += [c for c in df.columns if c.endswith(drp)]
    elif drop_suffix == 'infer':
        print('Automatically selecting features using lasso')
        drop_suffix = feature_selection(df, target_suffix, prefixes, LASSOFeatureEliminator(), rano_encoding, time_required)
        to_drop = []
        if drop_suffix:
            for drp in drop_suffix:
                to_drop += [f"{c}_{drp}" for c in prefixes]
        else:
            print('Automatic selection did not yield any features to drop.')
    elif callable(drop_suffix):
        print('Automatically selecting features using passed callable function')
        drop_suffix = feature_selection(df, target_suffix, prefixes, drop_suffix, rano_encoding, time_required)
        to_drop = []
        if drop_suffix:
            for drp in drop_suffix:
                to_drop += [f"{c}_{drp}" for c in prefixes]
        else:
            print('Automatic selection did not yield any features to drop.')
    else:
        print("No features are ignored")

    if time_required: to_drop = [c for c in to_drop if not c.endswith('timedelta_days')]

    print(f"Had {len(df.columns)} in the beginning")
    df.drop(columns=to_drop, inplace=True, errors='ignore')
    if test_size is not None: test.drop(columns=to_drop, inplace=True, errors='ignore')
    print(f"left with {len(df.columns)} features (not accounted for gt and misc cols in the data)")

    print('Training Set')
    print(df.describe())
    df.info()

    if test_size is not None:
        print('Testing Set')
        print(test.describe())
        test.info()

    if add_index_as_col:
        df['Lesion ID'] = df.index
        if test_size is not None: test['Lesion ID'] = test.index

    if save_processed is not None:
        df.to_csv(save_processed.parent/('train_'+save_processed.name), index=True)
        if test_size is not None: test.to_csv(save_processed.parent/('test_'+save_processed.name), index=True)
    
    
    if test_size is not None: return df, test
    else: return df, None

def normalize_other(df, test, normalize_suffix, prefixes, categorical, test_size, fill):
    processed_cols =[]
    if normalize_suffix is not None:
        for sfx in normalize_suffix:
            for tp in prefixes:
                for col in df.columns:
                    if col in categorical:
                        continue
                    if col.startswith(f"{tp}_{sfx}"):
                        df[col] = pd.to_numeric(df[col], errors='coerce')
                        if test_size is not None:test[col] = pd.to_numeric(test[col], errors='coerce')
                        std = max(df[col].std(), 1e-6) # avoid div by 0
                        mean = df[col].mean()   
                        print(f'standardizing {col} with mean {mean} and std {std}')
                        df[col]=(df[col]-mean).div(std) # try to parse every value to floats
                        if test_size is not None: test[col]=(test[col]-mean).div(std)
                    elif col in normalize_suffix and col not in processed_cols:
                        processed_cols.append(col)
                        df[col] = pd.to_numeric(df[col], errors='coerce')
                        if test_size is not None:test[col] = pd.to_numeric(test[col], errors='coerce')
                        std = max(df[col].std(), 1e-6) # avoid div by 0
                        mean = df[col].mean()   
                        print(f'standardizing {col} with mean {mean} and std {std}')
                        df[col]=(df[col]-mean).div(std) # try to parse every value to floats
                        if test_size is not None: test[col]=(test[col]-mean).div(std)

        if fill is not None: df.fillna(fill, inplace=True)
        if fill is not None and test_size is not None: test.fillna(fill, inplace=True)
                    
    else: print('No feature normalization is applied aside from volume')
    return df, test

def apply_volume_normalization(df, test, prefixes, normalize_volume, test_size):
    volume_cols = [c for c in df.columns if c.endswith('_volume') and c.split('_')[0] in prefixes]
    for step, mode in enumerate(normalize_volume.split('->')):
        print(f"working on step {step}: {mode}")
        ## standardize
        if mode == 'std':
            std = df[volume_cols].values.std()
            mean = df[volume_cols].values.mean()
            df[volume_cols] = (df[volume_cols]-mean).div(std)
            fmt = "({}*"+str(std)+"+"+str(mean)+")"

            # on testset
            if test_size is not None: test[volume_cols] = (test[volume_cols]-mean).div(std)

        ## normalize all volumes by the same factor
        elif mode == 'max':
            factor = df[volume_cols].max().max()
            print(f'Normalizing time series volume by factor {factor}')
            df[volume_cols] = df[volume_cols].div(factor)# normalize follow up volume
            fmt="({}*"+str(factor)+")"

            # on testset
            if test_size is not None: test[volume_cols] = test[volume_cols].div(factor)# normalize follow up volume
            
        ## normalize asll volumes by the whole brain volume of the respective series
        elif mode == 'log':
            print('normalizing time series volume its log')
            df[volume_cols] = np.log1p(df[volume_cols])
            fmt="(np.expm1({}))"

            # on testset
            if test_size is not None: test[volume_cols] =  np.log1p(test[volume_cols])

        
        ## normalize time series volume by the baseline volume
        elif mode == 'frac':
            print('Normalizing time series volume by baseline volume for each series')
            train_fmt = ["({}*"+str(f)+")" for f in df[volume_cols[0]].tolist()]
            df[volume_cols] = df[volume_cols].div(df[volume_cols[0]], axis=0) # normalize follow up volume
            # on testset
            fmt = ["({}*"+str(f)+")" for f in test[volume_cols[0]].tolist()] # generate reverter first before norming
            if test_size is not None: test[volume_cols] = test[volume_cols].div(test[volume_cols[0]], axis=0) # normalize follow up volume
            
        
        elif mode == '3root':
            print('Normalizing time series volume by its cubic root')
            df[volume_cols] = df[volume_cols]**(1/3) # normalize follow up volume
            fmt = "({}**3)"

            # on testset
            if test_size is not None: test[volume_cols] = test[volume_cols]**(1/3) # normalize follow up volume

    
        else: raise ValueError(f"Unrecognized volume normalization mode {normalize_volume}")

        ## accumulate decode functions
        if step == 0:
            if isinstance(fmt, list):
                if test_size is not None: test['ignored_vol_normalizer'] = fmt
                df['ignored_vol_normalizer'] = train_fmt

            else:
                if test_size is not None: test['ignored_vol_normalizer'] = [fmt]*len(test)
                df['ignored_vol_normalizer'] = [fmt]*len(df)
        
        else:
            
            if isinstance(fmt, list):
        
                for j, id in enumerate(test.index):
                    if test_size is not None: test.loc[id, 'ignored_vol_normalizer'] = test.loc[id, 'ignored_vol_normalizer'].format(fmt[j])
                

                for j, id in enumerate(df.index):
                    df.loc[id, 'ignored_vol_normalizer'] = df.loc[id, 'ignored_vol_normalizer'].format(train_fmt[j])
            
            else:
        
                for j, id in enumerate(test.index):
                    if test_size is not None: test.loc[id, 'ignored_vol_normalizer'] = test.loc[id, 'ignored_vol_normalizer'].format(fmt)
                
            
                for j, id in enumerate(df.index):
                    df.loc[id, 'ignored_vol_normalizer'] = df.loc[id, 'ignored_vol_normalizer'].format(fmt)
    return df, test

def sample_rejection(df, interpolate_CR_swing_length, outlier_detection_factor, prefixes, drop_CR_swing_length):
    print('Detecting and adressing outliers')
    drop_rows = []
    drop_row_for_dips_and_spikes =[]
    interpolate_rows = []
    for id, row in df.iterrows():
        ps = prefixes
        rano_cols = [f"{prfx}_rano" for prfx in ps if prfx != 't0']
        volume_cols = [f"{prfx}_volume" for prfx in ps]
        targets=row[rano_cols] 
        values = targets.values
        columns = targets.index
        volumes=row[volume_cols]
        drop_rows_dipsike = []
        compressed_rano = []
        for i, j in enumerate(values):

            ## experimental CR swing rejector
            j = 'non-CR' if j != 'CR' else j
            if i == 0: compressed_rano.append([1,j, [columns[i]]])
            else:
                if compressed_rano[len(compressed_rano)-1][1] == j:
                    compressed_rano[len(compressed_rano)-1][0] += 1
                    compressed_rano[len(compressed_rano)-1][2] = compressed_rano[len(compressed_rano)-1][2] + [columns[i]]
                else:
                    compressed_rano.append([1,j, [columns[i]]])

            ## experimental spike/dip outliere rejector
            if outlier_detection_factor is not None and id not in drop_rows_dipsike and i<len(values)-1 and id not in drop_rows and id not in interpolate_rows:
                vol_prev = max(volumes[i], 1e-6)
                vol_cur = max( volumes[i+1], 1e-6)
                vol_post = max(volumes[i+2], 1e-6)
                if vol_cur/vol_prev > outlier_detection_factor and vol_post/vol_cur < 1/outlier_detection_factor:# check if it spikes upwards
                    drop_rows_dipsike.append(id)
                elif vol_cur/vol_prev < 1/outlier_detection_factor and vol_post/vol_cur > outlier_detection_factor:# check if it dips downwards
                    drop_rows_dipsike.append(id)

        if [swing for swing in compressed_rano[:-1] if swing[0]>=drop_CR_swing_length and swing[1]=='CR'] and id not in drop_rows:
            print(f"Metastasis {id} is dropped and has rano compression: {compressed_rano}")
            drop_rows.append(id)
        
        if [swing for swing in compressed_rano[:-1] if swing[0]<=interpolate_CR_swing_length and swing[1]=='CR'] and id not in drop_rows:
            print(f"Metastasis {id} is interpolated and has rano compression: {compressed_rano}")
            interpolate_rows.append((id, compressed_rano))
        
        if id in drop_rows_dipsike and id not in drop_rows and id not in [tag[0] for tag in interpolate_rows]:
            print(f"Metastasis {id} is dropped for a suspicious spike")
            drop_rows.append(id)
            drop_row_for_dips_and_spikes.append(id)

    print(f"{len(drop_rows)} are dropped because of long CR swings")
    print(f"{len(drop_row_for_dips_and_spikes)} of those are because of dips and spikes")

    df = df.drop(drop_rows, axis='index')
    df = apply_interpolation_for_target_CR(df, interpolate_rows, interpolate_CR_swing_length, prefixes)
    return df, rano_cols

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

    tp_target = copy.deepcopy(tp_df[target_suffix])#.map(rano_encoding)
    tp_df.drop(columns=target_suffix, inplace=True)
    print(len(tp_data), len(tp_target))
    print(tp_target)

    if isinstance(eliminator, list):
        to_drop = []
        for elim in eliminator:
            to_drop += elim(tp_data.drop(to_drop), tp_target)
    else: to_drop = eliminator(tp_data, tp_target)

    return to_drop

def apply_interpolation_for_target_CR(df, targets, threshold, prefixes):
    vol_cols = [f"{tp}_volume" for tp in prefixes]
    for target in targets:
        id = target[0]
        compression = target[1]
        for i,(count, tag, cols) in enumerate(compression):
            if i == len(compression)-1: # ignore the last entry because we dont know if its a sing or not.
                break
            if tag == 'CR': # if it is a CR tag process it
                if count <= threshold: # if it is a swing of target length process it
                    if i == 0: # previous col in the 0th entry is the t0 volume
                        prev = 't0_volume'
                    else:
                        prev = compression[i-1][2][-1] # if not it is the last entry in the previous swings columns
                    post = compression[i+1][2][0] # posterior is always the first column in the next swings columns. No safety check needed because we break on compression[-1]
                    prev_time = f"{prev.split('_')[0]}_timedelta_days" # get the time delta from t0 to interpolate with time weighted linear interpolation
                    post_time = f"{post.split('_')[0]}_timedelta_days"

                    for current in cols: # iterate over all cols in the target cols and interpolate the volume
                        current_time = f"{current.split('_')[0]}_timedelta_days"
         
                        current_time = int(df.loc[id, current_time])
       
                        v1 =  int(df.loc[id, f"{prev.split('_')[0]}_volume"])

                        prev_time =  int(df.loc[id, prev_time])
                 
                        dt1 = prev_time-current_time
                        assert dt1<0
                        v2 =  int(df.loc[id, f"{post.split('_')[0]}_volume"])
                       
                        post_time =  int(df.loc[id, post_time])
                        
                        dt2 = post_time-current_time
           
                        assert dt2>0
                        # compute the stuff
                        a = (v2-v1)/(dt2-dt1) # slope
                        b = v1-a*dt1 # intercept
                        df.loc[id, f"{current.split('_')[0]}_volume"] = b # assign interpolated volume
                        data = df.loc[id][vol_cols[:vol_cols.index(f"{current.split('_')[0]}_volume")+1]]
                        df.loc[id, current] = get_rano(data)
    return df            

def get_rano(data):
    data = data.to_list()
    baseline = data[0]
    nadir = min(data)
    nadir = max(nadir, 1e-6) # avoid div by 0
    current = data[-1]
    if baseline<nadir:
        print("Values for autoread RANO incorrect: baseline < nadir")

    if current == 0:
        return 'CR'
        
    ratio_baseline =current/baseline
    ratio_nadir = current/nadir
    th1 = 0.343
    th2 = 1.728

    if ratio_baseline<=th1:
        response='PR'
    elif ratio_nadir<th2:
        response='SD'
    else:
        response='PD'
    return response

def encode_categorical(df, categorical):
    """
    One-hot encode categorical columns in a DataFrame.
    
    Parameters:
    - df: input DataFrame
    - categorical: list of column names to be treated as categorical
    """
    for col in categorical:
        # # Only encode if more than one unique non-null value
        # unique_vals = df[col].dropna().unique()
        # if len(unique_vals) <= 2:
        #     # Binary encoding: map to 0/1
        #     val_set = [v for v in unique_vals if v != '' and v is not None]
        #     if len(val_set) == 2:
        #         mapping = {v: i for i, v in enumerate(sorted(val_set))}
        #     else:
        #         mapping = {val_set[0]: 1}  # Single unique value (or mix of missing/empty)
        #     df[col] = df[col].map(mapping)
        # else:
        #     # One-hot encoding
        dummies = pd.get_dummies(df[col], prefix=col, dummy_na=False)
        df.drop(columns=col, inplace=True)
        df = pd.concat([df, dummies], axis=1)
    #print(df.columns)
    #raise RuntimeError("stop")
    return df



if __name__ == '__main__':
    data = pl.Path(f'/mnt/nas6/data/Target/BMPipeline_full_rerun/PARSED_METS_task_502/csv_nn/features.csv')
    prediction_type = 'binary'
    feature_selection = 'LASSO'
    method = 'LogisticRegression'
    output_path = pl.Path(f'/home/lorenz/BMDataAnalysis/output/final_baseline_complete')
    used_features = ['volume', 'total_lesion_count', 'total_lesion_volume', 'Sex',	'Age@Onset', 'Weight', 'Height', 'Primary_loc_1', 'lesion_location', 'radiomics_original']#, 'Primary_loc_2', 'Primary_hist_1', 'Primary_hist_2']#, 'radiomics_original']
    categorical =  ['Sex',	'Primary_loc_1', 'lesion_location']#, 'Primary_loc_2', 'Primary_hist_1', 'Primary_hist_2']
    if prediction_type == 'binary':
        rano_encoding={'CR':0, 'PR':0, 'SD':1, 'PD':1}
    elif prediction_type == '1v3':
        rano_encoding={'CR':0, 'PR':1, 'SD':1, 'PD':1}
    else:
        rano_encoding={'CR':0, 'PR':1, 'SD':2, 'PD':3}


    data_prefixes = ["t0", "t1", "t2", "t3", "t4", "t5", "t6"] # used in the training method to select the features for each step of the sweep
    volume_cols = [c+'_volume' for c in data_prefixes] # used to normalize the volumes
    rano_cols = [elem+'_rano' for elem in data_prefixes] # used in the training method to select the current targets
    train_data, test_data = load_prepro_data(data,
                                        categorical=categorical,
                                        fill=0,
                                        used_features=used_features,
                                        test_size=0.2,
                                        drop_suffix=None,
                                        prefixes=data_prefixes,
                                        target_suffix='rano',
                                        normalize_suffix=[f for f in used_features if f!='volume' and f!='total_lesion_count'],
                                        rano_encoding=rano_encoding,
                                        time_required=False,
                                        interpolate_CR_swing_length=1,
                                        drop_CR_swing_length=2,
                                        normalize_volume='std',
                                        save_processed=None)