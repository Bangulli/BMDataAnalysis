import sys
from pathlib import Path

# Add parent directory to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from prediction import regression_evaluation, train_regression_model_sweep
import pandas as pd
import pathlib as pl
from sklearn.svm import SVR, SVC
from sklearn.linear_model import LinearRegression, GammaRegressor, TweedieRegressor
from sklearn.model_selection import train_test_split
from visualization import plot_regression_metrics, plot_regression_metrics_sweep
from sklearn.model_selection import GridSearchCV, train_test_split
import data as d
from lightgbm import LGBMRegressor
import numpy as np
import csv
import os
from sklearn.base import BaseEstimator, RegressorMixin
import copy


if __name__ == '__main__':
    for normalization in ['frac->log']:#['3root->std', 'std', 'log', 'frac->3root', 'max', 'frac', None]: # 
        folder_name = 'init_vol'
        method_name = 'reg_sweep_linear'
        #normalization = '3root+std'
        model = LinearRegression() #NonZeroRegressor(regressor=SVR(), classifier=SVC())
        data = pl.Path(f'/mnt/nas6/data/Target/BMPipeline_full_rerun/229_patients faulty/PARSED_METS_task_502/csv_nn/features.csv')
        valid_char_norm = normalization.replace('->', '+') if normalization is not None else normalization
        output = pl.Path(f'''/home/lorenz/BMDataAnalysis/output/{folder_name}/regression_{method_name}/prepro_{valid_char_norm}''')
        os.makedirs(output, exist_ok=True)
        
        data_cols = ["t0", "t1", "t2", "t3", "t4", "t5", "t6"]
        rano_cols = [f"{elem}_rano" for elem in data_cols]


        train, test = d.load_prepro_data(data,
                                        used_features=['volume', 'init_volume'],
                                        test_size=0.2,
                                        drop_suffix=None,
                                        prefixes=data_cols,
                                        target_suffix='rano',
                                        normalize_suffix=['init_volume'],
                                        rano_encoding={ 'CR': 0,'PR': 1,'SD': 2,'PD': 3 },
                                        time_required=False,
                                        interpolate_CR_swing_length=1,
                                        drop_CR_swing_length=2,
                                        normalize_volume=normalization,
                                        save_processed=None)


        _, res, _ = train_regression_model_sweep(model, train, test, verbose=True, data_cols=[f"{elem}_volume" for elem in data_cols], rano_cols=rano_cols, extra_data=['init_volume'])
        plot_regression_metrics_sweep(res, output)


        