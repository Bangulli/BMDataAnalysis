from prediction import *
import pandas as pd
import pathlib as pl
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from visualization import *
import data as d
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
    feature_selection = 'LASSO'
    method = 'LGBM'
    model = LGBMClassifier(class_weight='balanced')
    output_path = pl.Path(f'/home/lorenz/BMDataAnalysis/output/{data_source.parent.name}')
    used_features = ['radiomics', 'volume']

    if prediction_type == 'binary':
        rano_encoding={'CR':0, 'PR':0, 'SD':1, 'PD':1}
    else:
        rano_encoding={'CR':0, 'PR':1, 'SD':2, 'PD':3}

    if feature_selection == 'LASSO':
        eliminator = d.LASSOFeatureEliminator()
    if feature_selection == 'correlation':
        eliminator = d.FeatureCorrelationEliminator()
    if feature_selection == 'model':
        eliminator = d.ModelFeatureEliminator()
    else:
        eliminator = None

    data_prefixes = ["t0", "t1", "t2", "t3", "t4", "t5", "t6"] # used in the training method to select the features for each step of the sweep
    volume_cols = [c+'_volume' for c in data_prefixes] # used to normalize the volumes
    rano_cols = [elem+'_rano' for elem in data_prefixes] # used in the training method to select the current targets

    train_data, test_data = d.load_prepro_data(path=data_source, drop_suffix=eliminator, rano_encoding=rano_encoding, used_features=used_features)
  
    output = output_path/f'classification_{prediction_type}_{method}_featuretypes={used_features}_selection={feature_selection}'
    os.makedirs(output, exist_ok=True)
    with open(output/'used_feature_names.txt', 'w') as file:
        file.write("Used feature names left in the dataframe:\n")
        for c in train.columns:
            file.write(f"   - {c}\n")
        file.write("NOTE: rano columns are used as targets not as prediction")




    _, res_quant = train_classification_model_sweep(model, train_data, test_data, data_prefixes=data_prefixes, rano_encoding=rano_encoding, prediction_targets=rano_cols)
    plot_prediction_metrics_sweep(res_quant, output)