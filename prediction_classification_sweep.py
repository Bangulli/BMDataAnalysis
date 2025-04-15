from prediction import *
import pandas as pd
import pathlib as pl
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from visualization import *

from lightgbm import LGBMClassifier
from xgboost import XGBClassifier

from sklearn.datasets import make_regression
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from scipy.stats import zscore

if __name__ == '__main__':
    # # Define pipeline with scaling and SVR
    # pipeline = Pipeline([
    #     ('scaler', StandardScaler()),
    #     ('svc', SVC())
    # ])

    # # Define a large parameter grid
    # param_grid = {
    #     'svc__kernel': ['linear', 'rbf', 'poly'],
    #     'svc__C': [0.1, 1, 10, 100, 1000],
    #     'svc__gamma': ['scale', 'auto'],
    #     'svc__class_weight': [None, 'balanced']
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
    prediction_type = 'multi'
    method = 'LGBM'
    model = LGBMClassifier(class_weight='balanced')

    if prediction_type == 'binary':
        rano_encoding={'CR':0, 'PR':0, 'SD':1, 'PD':1}
    else:
        rano_encoding={'CR':0, 'PR':1, 'SD':2, 'PD':3}

    data_cols = ["0", "60", "120", "180", "240", "300", "360"]
    rano_cols = ['rano-'+elem for elem in data_cols]

    data = pd.read_csv(f'/mnt/nas6/data/Target/task_524-504_PARSED_METS_mrct1000_nobatch/{folder_name}/volumes.csv', index_col=None)
    ### static data preprocessing
    data[data_cols[1:]] = data[data_cols[1:]].div(data["0"], axis=0) # normalize
    data[data_cols[0]]=zscore(data[data_cols[0]])
    
    rano = pd.read_csv(f'/mnt/nas6/data/Target/task_524-504_PARSED_METS_mrct1000_nobatch/{folder_name}/rano.csv', index_col=None)
    rano = rano.rename(columns = {c:r for c, r in zip(data_cols, rano_cols)})
    data = pd.concat((data, rano), axis=1)
    output = pl.Path(f'/home/lorenz/BMDataAnalysis/output/{folder_name}/classification_{prediction_type}_{method}')
    
    

    #data[data_cols[1:]] = data[data_cols[1:]].div(data["0"], axis=0) # normalize

    train, test = train_test_split(data, test_size=0.2, random_state=42)
    _, res_quant = train_classification_model_sweep(model, train, test, data_cols=data_cols, rano_cols=rano_cols, rano_encoding=rano_encoding)
    plot_prediction_metrics(res_quant, output)