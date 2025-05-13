import sys
from pathlib import Path

# Add parent directory to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from prediction import *
import pandas as pd
import pathlib as pl
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression, RidgeClassifier, LogisticRegression
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
from collections import Counter
from scipy.stats import zscore

if __name__ == '__main__':
    data = pl.Path(f'/mnt/nas6/data/Target/BMPipeline_full_rerun/229_patients faulty/PARSED_METS_task_502/csv_nn/features.csv')
    prediction_type = 'multiclass'
    feature_selection = 'LASSO'
    method = 'LGBM'
    model = LGBMClassifier(class_weight='balanced')
    output_path = pl.Path(f'/home/lorenz/BMDataAnalysis/output/baseline')
    used_features = ['volume', 'radiomics']

    if prediction_type == 'binary':
        rano_encoding={'CR':0, 'PR':0, 'SD':1, 'PD':1}
    else:
        rano_encoding={'CR':0, 'PR':1, 'SD':2, 'PD':3}

    if feature_selection == 'LASSO':
        eliminator = d.LASSOFeatureEliminator()
    elif feature_selection == 'correlation':
        eliminator = d.FeatureCorrelationEliminator()
    elif feature_selection == 'model':
        eliminator = d.ModelFeatureEliminator()
    else:
        eliminator = None

    data_prefixes = ["t0", "t1", "t2", "t3", "t4", "t5", "t6"] # used in the training method to select the features for each step of the sweep
    volume_cols = [c+'_volume' for c in data_prefixes] # used to normalize the volumes
    rano_cols = [elem+'_rano' for elem in data_prefixes] # used in the training method to select the current targets

    output = output_path/f'classification/{prediction_type}/featuretypes={used_features}_selection={feature_selection}/{method}'
    os.makedirs(output, exist_ok=True)

    train_data, test_data = d.load_prepro_data(data,
                                        used_features=used_features,
                                        test_size=0.2,
                                        fill=0,
                                        drop_suffix=eliminator,
                                        prefixes=data_prefixes,
                                        target_suffix='rano',
                                        normalize_suffix=[f for f in used_features if f!='volume'],
                                        rano_encoding=rano_encoding,
                                        time_required=False,
                                        interpolate_CR_swing_length=1,
                                        drop_CR_swing_length=2,
                                        normalize_volume='std',
                                        save_processed=output.parent/'used_data.csv')
    
    dist = Counter(test_data['t6_rano'])
    inv_enc = {v:k for k,v in rano_encoding.items()}
    dist = {inv_enc[k]:v for k,v in dist.items()}
  
    os.makedirs(output, exist_ok=True)
    with open(output/'used_feature_names.txt', 'w') as file:
        file.write("Used feature names left in the dataframe:\n")
        for c in train_data.columns:
            file.write(f"   - {c}\n")
        file.write("NOTE: rano columns are used as targets not as prediction")




    _, res_quant = train_classification_model_sweep(model, train_data, test_data, data_prefixes=data_prefixes, rano_encoding=rano_encoding, prediction_targets=rano_cols)
    plot_prediction_metrics_sweep(res_quant, output, classes=list(rano_encoding.keys()), distribution=dist)