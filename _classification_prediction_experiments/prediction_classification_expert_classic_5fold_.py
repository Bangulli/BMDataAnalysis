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
    predictor = {'LogisticRegression': LogisticRegression(class_weight='balanced')}#{'LogisticRegression': LogisticRegression(class_weight='balanced'), 'LGBM': LGBMClassifier(class_weight='balanced'), 'SVC':SVC(class_weight='balanced', probability=True)}
    target = ['binary']#, 'binary', 'sota']
    feature_selection = 'LASSO'
    features = [
        #['volume'],
        #['volume', 'total_lesion_count', 'total_lesion_volume', 'Sex',	'Age@Onset', 'Weight', 'Height', 'Primary_loc_1', 'Primary_hist_1', 'lesion_location'],
        #['volume', 'total_lesion_count', 'total_lesion_volume', 'Sex',	'Age@Onset', 'Weight', 'Height', 'Primary_loc_1', 'Primary_hist_1', 'lesion_location', 'deep'],
        # ['volume', 'total_lesion_count', 'total_lesion_volume', 'Sex',	'Age@Onset', 'Weight', 'Height', 'Primary_loc_1', 'Primary_hist_1', 'lesion_location', 'radiomics_original', 'border_radiomics'],
        ['volume', 'total_lesion_count', 'total_lesion_volume', 'Sex',	'Age@Onset', 'Weight', 'Height', 'Primary_loc_1', 'Primary_hist_1', 'lesion_location', 'radiomics_original', 'border_radiomics', 'deep'],
    ]
    for prediction_type in target:
        for used_features in features:
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

            if feature_selection == 'LASSO':
                eliminator = d.LASSOFeatureEliminator(alpha=0.1)
            elif feature_selection == 'correlation':
                eliminator = d.FeatureCorrelationEliminator(0.9)
            elif feature_selection == 'model':
                eliminator = d.ModelFeatureEliminator()
            else:
                eliminator = None

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
            output_path = pl.Path(f'/home/lorenz/BMDataAnalysis/final_output/classic_experts_5fold_assignments_bugfix')
            output = output_path/f'classification/{prediction_type}/featuretypes={used_features}_selection={feature_selection}'
            os.makedirs(output, exist_ok=True)
            categorical =  ['Sex',	'Primary_loc_1', 'lesion_location', 'Primary_hist_1']#, 'Primary_loc_2', 'Primary_hist_1', 'Primary_hist_2']
            data = pl.Path(f'/mnt/nas6/data/Target/BMPipeline_full_rerun/PARSED_METS_task_502/final_extraction/all_features_nn.csv')
            data, _ = d.load_prepro_data(data,
                                            discard=discard,
                                                    categorical=categorical,
                                                    fill=0,
                                                    used_features=used_features,
                                                    test_size=None,
                                                    drop_suffix=eliminator,
                                                    prefixes=data_prefixes,
                                                    target_suffix='rano',
                                                    normalize_suffix=[f for f in used_features if f!='volume'],
                                                    rano_encoding=rano_encoding,
                                                    time_required=False,
                                                    interpolate_CR_swing_length=1,
                                                    drop_CR_swing_length=2,
                                                    normalize_volume='std',
                                                    add_index_as_col=True,
                                                    save_processed=output/'encoding_test_used_data.csv')
            dist = Counter(data['t6_rano'])
            inv_enc = {v:k for v,k in enumerate(classes)}
            dist = {inv_enc[k]:v for k,v in dist.items()}
        
            os.makedirs(output, exist_ok=True)
            with open(output/'used_feature_names.txt', 'w') as file:
                file.write("Used feature names left in the dataframe:\n")
                for c in data.columns:
                    file.write(f"   - {c}\n")
                file.write("NOTE: rano columns are used as targets not as prediction")
            extra_data = [c for c in data.columns if not (c.startswith('ignored') or c.split('_')[0] in data_prefixes or 'Lesion ID' in c)]
            print("using extra data cols", extra_data)


            for method, model in predictor.items():
                wdir = output/method
                os.makedirs(wdir)
                _, res_quant = train_classification_model_sweep_cv(model, data, data_prefixes=data_prefixes, rano_encoding=inv_enc, prediction_targets=rano_cols, working_dir=wdir, extra_data=extra_data)
                plot_prediction_metrics_sweep(res_quant, wdir, classes=classes, distribution=dist)