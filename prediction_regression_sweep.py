from prediction import *
import pandas as pd
import pathlib as pl
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from visualization import *

from sklearn.datasets import make_regression
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from scipy.stats import zscore

from lightgbm import LGBMRegressor

if __name__ == '__main__':
    # Define pipeline with scaling and SVR
    # pipeline = Pipeline([
    #     ('scaler', StandardScaler()),
    #     ('svr', SVR())
    # ])

    # # Define a large parameter grid
    # param_grid = {
    #     'svr__kernel': ['linear', 'rbf', 'poly'],
    #     'svr__C': [0.1, 1, 10, 100, 1000],
    #     'svr__epsilon': [0.01, 0.1, 0.2, 0.5, 1],
    #     'svr__gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1],
    #     'svr__shrinking': [True, False]
    # }

    # # Initialize GridSearchCV
    # grid_search = GridSearchCV(
    #     estimator=pipeline,
    #     param_grid=param_grid,
    #     scoring='neg_mean_squared_error',
    #     cv=5,
    #     verbose=False,
    #     n_jobs=-1
    # )

    folder_name = 'csv_linear_multiclass_reseg_only_valid'
    method_name = 'relative_LGBM'
    model = LGBMRegressor()
    data =  pd.read_csv(f'/mnt/nas6/data/Target/task_524-504_PARSED_METS_mrct1000_nobatch/{folder_name}/volumes.csv', index_col=None)
    output = pl.Path(f'/home/lorenz/BMDataAnalysis/output/{folder_name}/regression_{method_name}')
    
    data_cols = ["0", "60", "120", "180", "240", "300", "360"]

    ### static data preprocessing
    data[data_cols[1:]] = data[data_cols[1:]].div(data["0"], axis=0) # normalize
    data[data_cols[0]]=zscore(data[data_cols[0]])

    train, test = train_test_split(data[data_cols], test_size=0.2)
    _, res_qual, res_quant = train_regression_model_sweep(model, train, test, data_cols=data_cols)
    plot_prediction_metrics(res_qual, output)
    #plot_prediction_metrics(res_quant, output)