import sys
from pathlib import Path

# Add parent directory to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))
import sklearn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import copy
from .evaluation import regression_evaluation

def train_regression_model_sweep(model, df_train, df_test, verbose=True, data_cols=None, rano_cols=None, extra_data=None):
    """
    Trains a sweep set of models to provided data.
    sweep == sweeps over the set and trains multiple possible configurations by training different feature configs to predict the next in time and 1 year response
        Example: t0 -> t1 & t0 -> t7; t0,t1 -> t2 & t0,t1 -> t7; ...
    model: a machine learning model, needs a fit and a predict fucntion, like sklearn. Trained models will be copies of this model
    df_train: a pandas dataframe with the training data
    df_test: a pandas dataframe with the test data, result metrics will be computed using this
    verbose: verbosity level, to control reporting while running, default True = print to console, False = dont 
    """
    cols = data_cols
    if extra_data is None: extra_data=[]
        
    models = {}
    qual_results = {}
    quant_results = {}
    for i in range(1, len(cols)):
        key_next = f"nxt_{cols[:i]+extra_data}->{cols[i]}"
        key_year = f"1yr_{cols[:i]+extra_data}->{cols[-1]}"
        if verbose: print(f'Training configuration {i}: {key_next} & {key_year}')

        models[key_next] = copy.deepcopy(model)
        models[key_year] = copy.deepcopy(model)
        if verbose: print("= Initialized models")

        X_train = df_train[cols[:i]+extra_data]
        X_test = df_test[cols[:i]+extra_data]

        y_next = df_train[cols[i]]
        y_year = df_train[cols[-1]]

        if verbose: print("== Training next value predictor")
        models[key_next].fit(X_train, y_next)

        if verbose: print("=== evaluating...")
        gt_next = df_test[cols[i]]
        pd_next = models[key_next].predict(X_test)
        qual_results[key_next] = regression_evaluation(gt_next, pd_next, df_test['ignored_vol_normalizer'])
        if verbose: print(f"=== Next Value Model {key_next} achieved results {qual_results[key_next]} for quality")

        if verbose: print("== Training year predictor")
        models[key_year].fit(X_train, y_year)

        if verbose: print("=== evaluating...")
        gt_year = df_test[cols[-1]]
        pd_year = models[key_year].predict(X_test)
        qual_results[key_year] = regression_evaluation(gt_year, pd_year, df_test['ignored_vol_normalizer'])
        if verbose: print(f"=== One Year Model {key_year} achieved results {qual_results[key_year]} for quality")

    return models, qual_results, None

