import sklearn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import copy
from .evaluation import classification_evaluation

def train_classification_model_sweep(model, df_train, df_test, data_prefixes, prediction_targets, verbose=True, rano_encoding={'CR':0, 'PR':1, 'SD':2, 'PD':3}):
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

        key_next = f"nxt_{data_prefixes[:i]}->{prediction_targets[i]}"
        key_year = f"1yr_{data_prefixes[:i]}->{prediction_targets[-1]}"
        if verbose: print(f'Training configuration {i}: {key_next} & {key_year}')
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

        if verbose: print("== Training next value predictor")
        models[key_next].fit(X_train, y_next)

        if verbose: print("=== evaluating...")
        pd_next = models[key_next].predict(X_test)
        quant_results[key_next] = classification_evaluation(gt_next, pd_next, rano_encoding)
        if verbose: print(f"=== Next Value Model {key_next} achieved results {quant_results[key_next]} for quantity")

        if verbose: print("== Training year predictor")
        models[key_year].fit(X_train, y_year)

        if verbose: print("=== evaluating...")
        pd_year = models[key_year].predict(X_test)
        quant_results[key_year] = classification_evaluation(gt_year, pd_year, rano_encoding)
        if verbose: print(f"=== One Year Model {key_year} achieved results {quant_results[key_year]} for quantity")

    return models, quant_results

