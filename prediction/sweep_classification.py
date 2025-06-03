import sklearn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import copy
from .evaluation import classification_evaluation
import torch
import pathlib as pl
from sklearn.metrics import make_scorer, balanced_accuracy_score, f1_score
from sklearn.datasets import make_regression
from sklearn.model_selection import GridSearchCV, train_test_split, StratifiedKFold, cross_validate
import os

def train_classification_model_sweep(model, df_train, df_test, data_prefixes, prediction_targets, verbose=True, rano_encoding={'CR':0, 'PR':1, 'SD':2, 'PD':3}, working_dir=pl.Path('./'), extra_data=[]):
    """
    Trains a sweep set of models to provided data.
    sweep == sweeps over the set and trains multiple possible configurations by training different feature configs to predict the next in time and 1 year response
        Example: t0 -> t1 & t0 -> t6; t0,t1 -> t2 & t0,t1 -> t6; ...
    model: a machine learning model, needs a fit and a predict fucntion, like sklearn. Trained models will be copies of this model
    df_train: a pandas dataframe with the training data
    df_test: a pandas dataframe with the test data, result metrics will be computed using this
    verbose: verbosity level, to control reporting while running, default True = print to console, False = dont 
    """
    if verbose: print(f'Encoding RANO response strings using encoding {rano_encoding}')
    rano_cols = [c for c in df_train.columns if c in prediction_targets]
    df_train[rano_cols] = df_train[rano_cols].applymap(lambda x: rano_encoding.get(x, x))
    df_test[rano_cols] = df_test[rano_cols].applymap(lambda x: rano_encoding.get(x, x))
    models = {}
    quant_results = {}
    for i in range(1, len(prediction_targets)):

        features = [d for d in df_train.columns if d not in prediction_targets and d.split('_')[0] in data_prefixes[:i]]
        features += extra_data

        key_next = f"nxt_{data_prefixes[:i]}->{prediction_targets[i]}"
        key_year = f"1yr_{data_prefixes[:i]}->{prediction_targets[-1]}"
        if verbose: print(f'Training configuration {i}: {key_next} & {key_year}')
        if verbose: print(f"using nontimepoint features {extra_data}")
        if verbose: [print(f"used feature: {c}") for c in features]
        if verbose: print(f"target variable are: {prediction_targets[-1]} for one year and {prediction_targets[i]} for next value")

        models[key_next] = copy.deepcopy(model)
        models[key_year] = copy.deepcopy(model)
        if verbose: print("= Initialized models")

        X_train = df_train[features]
        y_next = df_train[prediction_targets[i]].astype(int)
        y_year = df_train[prediction_targets[-1]].astype(int)

        X_test = df_test[features]
        gt_next = df_test[prediction_targets[i]]
        gt_year = df_test[prediction_targets[-1]]

        # if verbose: print("== Training next value predictor")
        # models[key_next].fit(X_train, y_next)
        # if verbose: print("== Saving next value predictor")
        # torch.save(models[key_next], working_dir/(key_next+'_model.pkl'))

        # if verbose: print("=== evaluating...")
        # pd_next = models[key_next].predict(X_test)
        # quant_results[key_next] = classification_evaluation(gt_next, pd_next, rano_encoding)
        # if verbose: print(f"=== Next Value Model {key_next} achieved results {quant_results[key_next]} for quantity")

        if verbose: print("== Training year predictor")
        models[key_year].fit(X_train, y_year)
        if verbose: print("== Saving next value predictor")
        torch.save(models[key_year], working_dir/(key_year+'_model.pkl'))

        if verbose: print("=== evaluating...")
        pd_year = models[key_year].predict(X_test)
        quant_results[key_year] = classification_evaluation(gt_year, pd_year, rano_encoding)
        if verbose: print(f"=== One Year Model {key_year} achieved results {quant_results[key_year]} for quantity")

    return models, quant_results

def train_classification_model_sweep_cv(model, df, data_prefixes, prediction_targets, verbose=True, rano_encoding={'CR':0, 'PR':1, 'SD':2, 'PD':3}, working_dir=pl.Path('./'), extra_data=[]):
    """
    Trains a sweep set of models to provided data.
    sweep == sweeps over the set and trains multiple possible configurations by training different feature configs to predict the next in time and 1 year response
        Example: t0 -> t1 & t0 -> t6; t0,t1 -> t2 & t0,t1 -> t6; ...
    model: a machine learning model, needs a fit and a predict fucntion, like sklearn. Trained models will be copies of this model
    df_train: a pandas dataframe with the training data
    df_test: a pandas dataframe with the test data, result metrics will be computed using this
    verbose: verbosity level, to control reporting while running, default True = print to console, False = dont 
    """
    if verbose: print(f'Encoding RANO response strings using encoding {rano_encoding}')
    rano_cols = [c for c in df.columns if c in prediction_targets]
    df[rano_cols] = df[rano_cols].applymap(lambda x: rano_encoding.get(x, x))
    models = {}
    quant_results = {}
    for i in range(1, len(prediction_targets)):

        features = [d for d in df.columns if d not in prediction_targets and d.split('_')[0] in data_prefixes[:i]]
        features += extra_data

        key_year = f"1yr_{data_prefixes[:i]}->{prediction_targets[-1]}"
        if verbose: print(f'Training configuration {i}: redacted & {key_year}')
        if verbose: print(f"using nontimepoint features {extra_data}")
        if verbose: [print(f"used feature: {c}") for c in features]
        if verbose: print(f"target variable are: {prediction_targets[-1]} for one year and {prediction_targets[i]} for next value")

        
        if verbose: print("= Initialized models")

        # Define scoring dictionary
        # scoring = {
        #     'balanced_accuracy': make_scorer(balanced_accuracy_score),
        #     'f1': make_scorer(f1_score, average='binary'),  # use 'macro' or 'weighted' for multiclass
        #     'precision': make_scorer(sklearn.metrics.precision_score),
        #     'recall': make_scorer(sklearn.metrics.recall_score)
        # }

        # cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        # results = cross_validate(models[key_year], df[features], df[prediction_targets[-1]], cv=cv, scoring=scoring)
        # results = {k.replace('test_',''):np.mean(v) for k, v in results.items()}
        # print(results)

        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        all_folds = []
        for fold, (train_idx, val_idx) in enumerate(skf.split(df, df[prediction_targets[-1]])):
            print(f"Fold {fold+1}")
            wdir = working_dir/f"Fold {fold+1}"/key_year
            os.makedirs(wdir, exist_ok=True)
            train_df = df.iloc[train_idx].reset_index(drop=True)
            test_df = df.iloc[val_idx].reset_index(drop=True)
            fold_model = copy.deepcopy(model)
            fold_model.fit(train_df[features], train_df[prediction_targets[-1]])

            gt = test_df[prediction_targets[-1]]
            ids = test_df['Lesion ID']
            pred = fold_model.predict(test_df[features])
            prob = fold_model.predict_proba(test_df[features])[:, 1]

            all_folds.append(classification_evaluation(gt, pred, ids, rano_encoding, prob, wdir))

            torch.save(fold_model, wdir/'model.pkl')
            

        interim = {}
        for elem in all_folds:
            for k, v in elem.items():
                if k == 'confusion_matrix' or k=='classification_report':continue
                if not k in interim.keys():
                    interim[k] = [v]
                else:
                    interim[k].append(v)
        #print(interim)
        interim_std = {}
        for k in interim.keys():
            interim_std[k] = np.std(interim[k])
            interim[k] = np.mean(interim[k])
        quant_results[key_year] = interim
        #all_stds[key_year] = interim_std
        #quant_results[key_year] = all_folds
        if verbose: print(f"=== One Year Model {key_year} achieved results {quant_results[key_year]} for quantity")
        

    return models, quant_results