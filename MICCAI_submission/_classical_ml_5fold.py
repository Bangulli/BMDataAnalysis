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
    predictor = {'LGBM': LGBMClassifier(class_weight='balanced')}#{'LogisticRegression': LogisticRegression(class_weight='balanced'), 'LGBM': LGBMClassifier(class_weight='balanced'), 'SVC':SVC(class_weight='balanced', probability=True)}
    target = ['1v3', 'binary']#, 'sota']
    patient_features = ['Age@Onset', 'Weight', 'Height']
    tp_features = ['volume', 'radiomics']
    categorical_features =  ['Sex',	'Primary_loc_1', 'lesion_location', 'Primary_hist_1']
    for prediction_type in target:
            if prediction_type == 'binary':
                rano_encoding={'CR':0, 'PR':0, 'SD':1, 'PD':1}
                classes = ['resp', 'non-resp']
            elif prediction_type == '1v3':
                rano_encoding={'CR':0, 'PR':1, 'SD':1, 'PD':1}
                classes = ['CR', 'non-CR']
            elif prediction_type == 'sota':
                rano_encoding={'CR':0, 'PR':0, 'SD':0, 'PD':1}
                classes = ['non-PD', 'PD']
            else:
                rano_encoding={'CR':0, 'PR':1, 'SD':2, 'PD':3}
                classes = list(rano_encoding.keys())

            data_prefixes = ["t0", "t1", "t2", "t3", "t4", "t5", "t6"] # used in the training method to select the features for each step of the sweep
            volume_cols = [c+'_volume' for c in data_prefixes] # used to normalize the volumes
            rano_cols = [elem+'_rano' for elem in data_prefixes] # used in the training method to select the current targets

            
            discard = ['sub-PAT0122:1', 
                'sub-PAT0167:0', 
                'sub-PAT0182:2', 
                'sub-PAT0342:0', 
                'sub-PAT0411:0', 
                'sub-PAT0434:6', 
                'sub-PAT0434:9', 
                'sub-PAT0434:10', 
                'sub-PAT0434:11', 
                'sub-PAT0480:20', 
                'sub-PAT0484:4', 
                'sub-PAT0490:0', 
                'sub-PAT0612:2', 
                'sub-PAT0666:0', 
                'sub-PAT0756:0', 
                'sub-PAT1028:3',
                'sub-PAT0045:6',
                'sub-PAT0105:0',
                'sub-PAT0441:0', 
                'sub-PAT0686:1',
                'sub-PAT0807:3',
                ]
            
            output_path = pl.Path(f'/home/lorenz/BMDataAnalysis/MICCAI_submission/Classical')
            output = output_path/f'classification/{prediction_type}/featuretypes={patient_features} - {categorical_features} - {tp_features}'
            os.makedirs(output, exist_ok=True)
            

            data = pl.Path(f'/mnt/nas6/data/Target/BMPipeline_full_rerun/PARSED_METS_task_502/final_extraction/all_features_nn.csv')
            data = pd.read_csv(data, index_col="Lesion ID")
            
            keep_col = ['t6_rano'] + categorical_features + patient_features
            for c in data.columns:
                for p in data_prefixes:
                    for f in tp_features:
                        if c.startswith(f"{p}_{f}"):
                            keep_col.append(c)
            
            data.drop(index=discard, inplace=True)
            data.drop(columns=[c for c in data.columns if c not in keep_col], inplace=True)
            data['t6_rano'] = data['t6_rano'].map(rano_encoding).astype(int)
            #print([c for c in data.columns if not c.startswith('t')]
            extra_features = patient_features.copy()
            for col in categorical_features:
                dummies = pd.get_dummies(data[col], prefix=col, dummy_na=False)
                extra_features += list(dummies.columns)
                data.drop(columns=col, inplace=True)
                data = pd.concat([data, dummies], axis=1)
            data.fillna(0, inplace=True)

            dist = Counter(data['t6_rano'])
            inv_enc = {v:k for v,k in enumerate(classes)}
            dist = {inv_enc[k]:v for k,v in dist.items()}
            print("Class distribution is:", dist)
        
            os.makedirs(output, exist_ok=True)
            with open(output/'used_feature_names.txt', 'w') as file:
                file.write("Used feature names left in the dataframe:\n")
                for c in data.columns:
                    file.write(f"   - {c}\n")
                file.write("NOTE: rano columns are used as targets not as prediction")
            extra_data = [c for c in data.columns if not (c.startswith('ignored') or c.split('_')[0] in data_prefixes or 'Lesion ID' in c)]
            print("using extra data cols", patient_features)


            for method, model in predictor.items():
                wdir = output/method
                os.makedirs(wdir, exist_ok=True)
                _, res_quant = train_classification_model_sweep_cv(model, data, data_prefixes=data_prefixes, rano_encoding=rano_encoding, prediction_target='t6_rano', working_dir=wdir, extra_data=extra_features)
                plot_prediction_metrics_sweep(res_quant, wdir, classes=classes, distribution=dist)