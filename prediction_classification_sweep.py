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

    data_source = pl.Path('/mnt/nas6/data/Target/BMPipeline_DEVELOPMENT_runs/task_502_PARSED_METS_mrct1000_nobatch/csv_nn_only_valid/features.csv')
    prediction_type = 'multi'
    method = 'LGBM'
    model = LGBMClassifier(class_weight='balanced')
    output_path = pl.Path(f'/home/lorenz/BMDataAnalysis/output/{data_source.parent.name}')
    used_features = ['radiomics']

    if prediction_type == 'binary':
        rano_encoding={'CR':0, 'PR':0, 'SD':1, 'PD':1}
    else:
        rano_encoding={'CR':0, 'PR':1, 'SD':2, 'PD':3}

    data_prefixes = ["t0", "t1", "t2", "t3", "t4", "t5", "t6"] # used in the training method to select the features for each step of the sweep
    volume_cols = [c+'_volume' for c in data_prefixes] # used to normalize the volumes
    rano_cols = [elem+'_rano' for elem in data_prefixes] # used in the training method to select the current targets

    data = pd.read_csv(data_source, index_col=None)
    data.fillna(0)
        

    # static data preprocessing
    data[volume_cols[1:]] = data[volume_cols[1:]].div(data[volume_cols[0]], axis=0) # normalize follow up volume
    data[volume_cols[0]]=zscore(data[volume_cols[0]]) # normalize init volume
    ## normalize delta times by one year
    times = [c+'_timedelta_days' for c in data_prefixes]
    data[times]=data[times].div(365.25) 
    ## nromalize radiomics 
    for tp in data_prefixes:
        for col in data.columns:
            if col.startswith(f"{tp}_radiomics"):
                data[col]=zscore(pd.to_numeric(data[col], errors='coerce')) # try to parse every value to floats
    ## drop unused features
    to_keep = []
    for col in data.columns:
        for tp in data_prefixes:
            for feature in used_features:
                if col.startswith(f"{tp}_{feature}") or col in rano_cols and col not in to_keep:
                    to_keep.append(col)

    print(to_keep)
    ([data.drop(columns=c, inplace=True, axis=0) for c in data.columns if c not in to_keep])
    print(f'Running with features {data.columns}')
  
    output = output_path/f'classification_{prediction_type}_{method}_features={used_features}'

    ## dataset splitting
    labels = [d[rano_cols[-1]] for i, d in data.iterrows()]
    train, test = train_test_split(data, test_size=0.2, random_state=42, stratify=labels)


    _, res_quant = train_classification_model_sweep(model, train, test, data_prefixes=data_prefixes, rano_encoding=rano_encoding, prediction_targets=rano_cols)
    plot_prediction_metrics_sweep(res_quant, output)