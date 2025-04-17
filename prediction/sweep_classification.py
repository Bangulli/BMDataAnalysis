import sklearn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import copy

def train_classification_model_sweep(model, df_train, df_test, data_cols, rano_cols, prediction_targets, verbose=True, rano_encoding={'CR':0, 'PR':1, 'SD':2, 'PD':3}):
    """
    Trains a sweep set of models to provided data.
    sweep == sweeps over the set and trains multiple possible configurations by training different feature configs to predict the next in time and 1 year response
        Example: t0 -> t1 & t0 -> t7; t0,t1 -> t2 & t0,t1 -> t7; ...
    model: a machine learning model, needs a fit and a predict fucntion, like sklearn. Trained models will be copies of this model
    df_train: a pandas dataframe with the training data
    df_test: a pandas dataframe with the test data, result metrics will be computed using this
    verbose: verbosity level, to control reporting while running, default True = print to console, False = dont 
    """
    if verbose: print(f'Encoding RANO response strings using encoding {rano_encoding}')
    df_train[rano_cols] = df_train[rano_cols].applymap(lambda x: rano_encoding.get(x, x))
    df_test[rano_cols] = df_test[rano_cols].applymap(lambda x: rano_encoding.get(x, x))
    models = {}
    quant_results = {}
    for i in range(1, len(prediction_targets)):

        features = [d for d in data_cols if d not in prediction_targets[i-1:]]

        key_next = f"nxt_{features}->{prediction_targets[i]}"
        key_year = f"1yr_{features}->{prediction_targets[-1]}"
        if verbose: print(f'Training configuration {i}: {key_next} & {key_year}')

        models[key_next] = copy.deepcopy(model)
        models[key_year] = copy.deepcopy(model)
        if verbose: print("= Initialized models")

        X_train = df_train[features]
        y_next = df_train[rano_cols[i]].astype(int)
        y_year = df_train[rano_cols[-1]].astype(int)

        X_test = df_test[features]
        gt_next = df_test[rano_cols[i]]
        gt_year = df_test[rano_cols[-1]]

        if verbose: print("== Training next value predictor")
        models[key_next].fit(X_train, y_next)

        if verbose: print("=== evaluating...")
        pd_next = models[key_next].predict(X_test)
        quant_results[key_next] = quantity_assessment(gt_next, pd_next)
        if verbose: print(f"=== Next Value Model {key_next} achieved results {quant_results[key_next]} for quantity")

        if verbose: print("== Training year predictor")
        models[key_year].fit(X_train, y_year)

        if verbose: print("=== evaluating...")
        pd_year = models[key_year].predict(X_test)
        quant_results[key_year] = quantity_assessment(gt_year, pd_year)
        if verbose: print(f"=== One Year Model {key_year} achieved results {quant_results[key_year]} for quantity")

    return models, quant_results

def quantity_assessment(rano_gt, rano_pd):
    res = {}

    # compute class weights
    weights = {u: len(rano_gt)/np.sum(rano_gt==u) for u in np.unique(rano_gt)}
    #print(weights)
    # make weight vector
    weights = [weights[l] for l in rano_gt]
    # is equivalent to sklearn.utils.class_weight.compute_sample_weight
    
    res['balanced_accuracy'] = sklearn.metrics.accuracy_score(rano_gt, rano_pd, sample_weight=weights)
    res['accuracy'] = sklearn.metrics.accuracy_score(rano_gt, rano_pd)
    #res['roc_auc'] = sklearn.metrics.roc_auc_score(rano_gt, rano_pd, multi_class='ovr')
    res['f1'] = sklearn.metrics.f1_score(rano_gt, rano_pd, average='weighted')
    res['precision'] = sklearn.metrics.precision_score(rano_gt, rano_pd, average='weighted')
    res['recall'] = sklearn.metrics.recall_score(rano_gt, rano_pd, average='weighted')
    return res