import sklearn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import copy

def train_model_sweep(model, df_train, df_test, verbose=True, data_cols=None, rano_cols=None):
    """
    Trains a sweep set of models to provided data.
    sweep == sweeps over the set and trains multiple possible configurations by training different feature configs to predict the next in time and 1 year response
        Example: t0 -> t1 & t0 -> t7; t0,t1 -> t2 & t0,t1 -> t7; ...
    model: a machine learning model, needs a fit and a predict fucntion, like sklearn. Trained models will be copies of this model
    df_train: a pandas dataframe with the training data
    df_test: a pandas dataframe with the test data, result metrics will be computed using this
    verbose: verbosity level, to control reporting while running, default True = print to console, False = dont 
    """
    if data_cols is None:
        cols = list(df_train.columns)
    else:
        cols = data_cols
        
    models = {}
    qual_results = {}
    quant_results = {}
    for i in range(1, len(cols)):
        key_next = f"nxt_{cols[:i]}->{cols[i]}"
        key_year = f"1yr_{cols[:i]}->{cols[-1]}"
        if verbose: print(f'Training configuration {i}: {key_next} & {key_year}')

        models[key_next] = copy.deepcopy(model)
        models[key_year] = copy.deepcopy(model)
        if verbose: print("= Initialized models")

        X_train = df_train[cols[:i]]
        X_test = df_test[cols[:i]]

        y_next = df_train[cols[i]]
        y_year = df_train[cols[-1]]

        if verbose: print("== Training next value predictor")
        models[key_next].fit(X_train, y_next)

        if verbose: print("=== evaluating...")
        gt_next = df_test[cols[i]]
        pd_next = models[key_next].predict(X_test)
        qual_results[key_next] = quality_assessment(gt_next, pd_next)
        quant_results[key_next] = quantity_assessment(gt_next, pd_next, X_test)
        if verbose: print(f"=== Next Value Model {key_next} achieved results {qual_results[key_next]} for quality and {quant_results[key_next]} for quantity")

        if verbose: print("== Training year predictor")
        models[key_year].fit(X_train, y_year)

        if verbose: print("=== evaluating...")
        gt_year = df_test[cols[-1]]
        pd_year = models[key_year].predict(X_test)
        qual_results[key_year] = quality_assessment(gt_year, pd_year)
        quant_results[key_year] = quantity_assessment(gt_year, pd_year, X_test)
        if verbose: print(f"=== One Year Model {key_year} achieved results {qual_results[key_year]} for quality and {quant_results[key_year]} for quantity")

    return models, qual_results, quant_results

def quality_assessment(gt, pd):
    res = {}
    res['rmse'] = sklearn.metrics.root_mean_squared_error(gt, pd)
    res['r2'] = sklearn.metrics.r2_score(gt, pd)
    res['mae'] = sklearn.metrics.mean_absolute_error(gt, pd) 
    return res

def quantity_assessment(gt, pd, input):
    res = {}
    rano_gt, rano_pd = assign_rano(input, gt, pd)

    res['accuracy'] = sklearn.metrics.accuracy_score(rano_gt, rano_pd)
    #res['roc_auc'] = sklearn.metrics.roc_auc_score(rano_gt, rano_pd, multi_class='ovr')
    res['f1'] = sklearn.metrics.f1_score(rano_gt, rano_pd, average='weighted')
    res['precision'] = sklearn.metrics.precision_score(rano_gt, rano_pd, average='weighted')
    res['recall'] = sklearn.metrics.recall_score(rano_gt, rano_pd, average='weighted')
    return res

def assign_rano(input, gt, pd, mode='3d'):
    gt=list(gt)
    pd=list(pd)
    gts = []
    pds = []
    for i in range(len(input)):
        vector = np.asarray(input.iloc[i,:])

        baseline = vector[0]
        nadir = np.min(vector)
        nadir = max(nadir, 1e-6) # avoid division by zero error
        nadir = min(nadir, baseline) # overrule nadir with baseline if it is smaller

        gts.append(rano(gt[i], baseline, nadir, mode))
        pds.append(rano(pd[i], baseline, nadir, mode))

    return gts, pds

def rano(lesion_volume, baseline, nadir, mode='3d'):
    """
    Returns the RANO-BM classification for the Metastasis, given the basline and nadir values from the series
    """
    if baseline<nadir:
        print("Values for autoread RANO incorrect: baseline < nadir")

    if lesion_volume == 0:
        return 0
        
    ratio_baseline = lesion_volume/baseline
    ratio_nadir = lesion_volume/nadir
    if mode == '1d':
        th1 = 0.7
        th2 = 1.2
    elif mode == '3d':
        th1 = 0.343
        th2 = 1.728

    if ratio_baseline<=th1:
        response=1
    elif ratio_nadir<th2:
        response=2
    else:
        response=3
    return response