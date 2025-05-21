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
    config = {'LogisticRegression': LogisticRegression, 'LGBM': LGBMClassifier, 'SVC':SVC}
    for method, model in config.items():
        data = pl.Path(f'/mnt/nas6/data/Target/BMPipeline_full_rerun/PARSED_METS_task_502/csv_nn/features.csv')
        prediction_type = '1v3'
        feature_selection = None#'LASSO'
        #method = 'LogisticRegression'
        model = model(class_weight='balanced')
        output_path = pl.Path(f'/home/lorenz/BMDataAnalysis/output/baseline_complete')
        used_features = ['volume']#, 'total_lesion_count', 'total_lesion_volume', 'Sex',	'Age@Onset', 'Weight', 'Height', 'Primary_loc_1', 'Primary_hist_1', 'lesion_location', 'radiomics_original']# 'Primary_loc_2', 'Primary_hist_1', 'Primary_hist_2']#]
        categorical =  []#['Sex',	'Primary_loc_1', 'lesion_location', 'Primary_hist_1']#, 'Primary_loc_2', 'Primary_hist_1', 'Primary_hist_2']
        if prediction_type == 'binary':
            rano_encoding={'CR':0, 'PR':0, 'SD':1, 'PD':1}
        elif prediction_type == '1v3':
            rano_encoding={'CR':0, 'PR':1, 'SD':1, 'PD':1}
        else:
            rano_encoding={'CR':0, 'PR':1, 'SD':2, 'PD':3}

        if feature_selection == 'LASSO':
            eliminator = d.LASSOFeatureEliminator(alpha=0.1)
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
                                            categorical=categorical,
                                            fill=0,
                                            used_features=used_features,
                                            test_size=0.2,
                                            drop_suffix=eliminator,
                                            prefixes=data_prefixes,
                                            target_suffix='rano',
                                            normalize_suffix=[f for f in used_features if f!='volume' and f!='total_lesion_count'],
                                            rano_encoding=rano_encoding,
                                            time_required=False,
                                            interpolate_CR_swing_length=1,
                                            drop_CR_swing_length=2,
                                            normalize_volume='std',
                                            save_processed=output.parent/'encoding_test_used_data.csv')
        
        dist = Counter(test_data['t6_rano'])
        inv_enc = {v:k for k,v in {'CR':0, 'non-CR':1}.items()}
        dist = {inv_enc[k]:v for k,v in dist.items()}
    
        os.makedirs(output, exist_ok=True)
        with open(output/'used_feature_names.txt', 'w') as file:
            file.write("Used feature names left in the dataframe:\n")
            for c in train_data.columns:
                file.write(f"   - {c}\n")
            file.write("NOTE: rano columns are used as targets not as prediction")
        extra_data = [c for c in train_data.columns if not (c.startswith('ignored') or c.split('_')[0] in data_prefixes)]
        print("using extra data cols", extra_data)



        _, res_quant = train_classification_model_sweep(model, train_data, test_data, data_prefixes=data_prefixes, rano_encoding={'CR':0, 'non-CR':1}, prediction_targets=rano_cols, working_dir=output, extra_data=extra_data)
        plot_prediction_metrics_sweep(res_quant, output, classes=['CR', 'non-CR'], distribution=dist)