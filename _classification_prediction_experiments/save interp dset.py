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
        data = pl.Path(f'/mnt/nas6/data/Target/BMPipeline_full_rerun/PARSED_METS_task_502/csv_linear_clean/rerun.csv')
        prediction_type = '1v3'
        feature_selection = None
        #method = 'LogisticRegression'
        model = model(class_weight='balanced')
        output_path = pl.Path(f'/home/lorenz/BMDataAnalysis/output/5fold_promising')
        used_features = ['volume', 'total_lesion_count', 'total_lesion_volume', 'Sex',	'Age@Onset', 'Weight', 'Height', 'Primary_loc_1', 'Primary_hist_1', 'lesion_location']#, 'radiomics_original', 'deep']# 'Primary_loc_2', 'Primary_hist_1', 'Primary_hist_2']#]
        categorical =  ['Sex',	'Primary_loc_1', 'lesion_location', 'Primary_hist_1']#, 'Primary_loc_2', 'Primary_hist_1', 'Primary_hist_2']
        if prediction_type == 'binary':
            rano_encoding={'CR':0, 'PR':0, 'SD':1, 'PD':1}
        elif prediction_type == '1v3':
            rano_encoding={'CR':0, 'PR':1, 'SD':1, 'PD':1}
        else:
            rano_encoding={'CR':0, 'PR':1, 'SD':2, 'PD':3} #{'CR':'CR', 'PR':'PR', 'SD':'SD', 'PD':'PD'}

        if feature_selection == 'LASSO':
            eliminator = d.LASSOFeatureEliminator(alpha=0.1)
        elif feature_selection == 'correlation':
            eliminator = d.FeatureCorrelationEliminator()
        elif feature_selection == 'model':
            eliminator = d.ModelFeatureEliminator()
        else:
            eliminator = None

        data_prefixes = [f"t{i}" for i in range(7)] # used in the training method to select the features for each step of the sweep
        volume_cols = [c+'_volume' for c in data_prefixes] # used to normalize the volumes
        rano_cols = [elem+'_rano' for elem in data_prefixes] # used in the training method to select the current targets

        output = output_path/f'classification/{prediction_type}/featuretypes={used_features}_selection={feature_selection}/{method}'
        os.makedirs(output, exist_ok=True)

        data, _ = d.load_prepro_data(pl.Path('/mnt/nas6/data/Target/BMPipeline_full_rerun/PARSED_METS_task_502/csv_sanitycheck/nn.csv'),
                                            categorical=[],
                                            fill=0,
                                            used_features=used_features,
                                            test_size=None,
                                            drop_suffix=eliminator,
                                            prefixes=data_prefixes,
                                            target_suffix='rano',
                                            normalize_suffix=[f for f in used_features if f!='volume' and f!='total_lesion_count'],
                                            rano_encoding={'CR':'CR', 'PR':'PR', 'SD':'SD', 'PD':'PD'},
                                            time_required=False,
                                            interpolate_CR_swing_length=1,
                                            drop_CR_swing_length=2,
                                            normalize_volume=None,
                                            save_processed=None)#pl.Path('/mnt/nas6/data/Target/BMPipeline_full_rerun/PARSED_METS_task_502/csv_sanitycheck/noswing_linear.csv'))#pl.Path('/mnt/nas6/data/Target/BMPipeline_full_rerun/PARSED_METS_task_502/csv_linear_clean/noswing_rerun.csv'))
        
        # data_prefixes = [f"t{i}" for i in range(38)]
        # data, _ = d.load_prepro_noisy_data(pl.Path('/mnt/nas6/data/Target/BMPipeline_full_rerun/PARSED_METS_task_502/csv_sanitycheck/noninter.csv'),
        #                                     categorical=[],
        #                                     fill=0,
        #                                     used_features=used_features,
        #                                     test_size=None,
        #                                     drop_suffix=eliminator,
        #                                     prefixes=data_prefixes,
        #                                     target_suffix='rano',
        #                                     normalize_suffix=[f for f in used_features if f!='volume' and f!='total_lesion_count'],
        #                                     rano_encoding={'CR':'CR', 'PR':'PR', 'SD':'SD', 'PD':'PD'},
        #                                     time_required=False,
        #                                     interpolate_CR_swing_length=1,
        #                                     drop_CR_swing_length=2,
        #                                     normalize_volume=None,
        #                                     target_time=360,
        #                                     save_processed=pl.Path('/mnt/nas6/data/Target/BMPipeline_full_rerun/PARSED_METS_task_502/csv_sanitycheck/noswing_noninter.csv'))
        # #plot_sankey(data[rano_cols], pl.Path('/mnt/nas6/data/Target/BMPipeline_full_rerun/PARSED_METS_task_502/csv_linear_clean'), tag='noswing_')
        break