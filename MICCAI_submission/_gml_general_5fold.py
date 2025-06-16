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
import torch_geometric.transforms as T
class AddNoise(T.BaseTransform):
    def __init__(self, p=0.1):
        self.p = p
    def __call__(self, data):
        data.x = data.x + self.p * torch.randn_like(data.x)
        data.edge_weights = data.edge_weights + 0.1 * torch.randn_like(data.edge_weights)
        # data.edge_attr = data.edge_attr + 0.01 * torch.randn_like(data.edge_attr)
        return data
    
class FeatureDropout(T.BaseTransform):
    def __init__(self, p: float = 0.1):
        self.p = p

    def __call__(self, data):
        if data.x is not None:
            mask = torch.rand_like(data.x) > self.p
            data.x = data.x * mask
        return data
if __name__ == '__main__':
    target = ['1v3', 'binary']#, 'sota']
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
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
            
            output_path = pl.Path(f'/home/lorenz/BMDataAnalysis/MICCAI_submission/GML')
            output = output_path/f'classification/{prediction_type}/featuretypes={patient_features} - {categorical_features} - {tp_features}'
            os.makedirs(output, exist_ok=True)
            

            data = pl.Path(f'/mnt/nas6/data/Target/BMPipeline_full_rerun/PARSED_METS_task_502/final_extraction/all_features_nn.csv')
            data = pd.read_csv(data, index_col='Lesion ID')
            data['Lesion ID'] = data.index
            
            keep_col = ['t6_rano', 'Lesion ID'] + categorical_features + patient_features + [f"{t}_timedelta_days" for t in data_prefixes]
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
            print("using extra data cols", patient_features)


            

            skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            all_folds = []
            for fold, (train_idx, val_idx) in enumerate(skf.split(data, data['t6_rano'])):
                print(f"Fold {fold+1}")
                wdir = output/f"Fold {fold+1}"
                os.makedirs(wdir, exist_ok=True)
                train_df = data.iloc[train_idx].reset_index(drop=True)
                test_df   = data.iloc[val_idx].reset_index(drop=True)
                print('Standardizing values')
                with open(wdir/"standardization.csv", "w") as file:
                    writer = csv.DictWriter(file, fieldnames=['column', 'mean', 'std'])
                    writer.writeheader()
                    for col in train_df.columns:
                        if (col in patient_features) or (col == 't6_rano') or (col=='Lesion ID') or ('timedelta' in col):
                            continue
                        train_df[col] = pd.to_numeric(train_df[col], 'coerce')
                        test_df[col] = pd.to_numeric(test_df[col], 'coerce')
                        mean = train_df[col].mean()
                        std = train_df[col].std()
                        ref = {'column': col, 'mean': mean, 'std': std}
                        writer.writerow(ref)
                        train_df[col] = (train_df[col]-mean)/std
                        test_df[col] = (test_df[col]-mean)/std
                # make datasets
                dataset_train = d.BrainMetsGraphClassification(train_df,
                    used_timepoints = data_prefixes[:-1], 
                    ignored_suffixes = ('_timedelta_days', '_rano', 'Lesion ID'), 
                    rano_encoding = rano_encoding,
                    target_name = 't6_rano',
                    extra_features = patient_features,
                    fully_connected=True,
                    direction='past',
                    transforms = T.Compose([AddNoise(0.5), FeatureDropout(0.1)]),
                    random_period = True,
                    )
                
                loss = F.binary_cross_entropy_with_logits
                model = BigGAT(1, dataset_train.get_node_size()).to(device)
                optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=5e-4)
                scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                    optimizer,
                    T_0=50,       # First restart after 10 epochs
                    T_mult=2,     # Increase period between restarts by this factor
                    eta_min=1e-6  # Minimum LR
                )

                # run training
                best_model, best_loss = torch_engine.train(model, 
                                                dataset= dataset_train, 
                                                loss_function=loss,
                                                epochs=1000,
                                                optimizer=optimizer,
                                                scheduler=scheduler,
                                                working_dir=wdir,
                                                device=device,
                                                validation=0.25,
                                                batch_size=128,
                                                weighted_loss=True
                                                )
                print(f"Best model achieved loss {best_loss:4f}")
                all_results = {}
                for i in range(1, len(data_prefixes)):
                    key_year = f"1yr_{data_prefixes[:i]}->t6_rano"
                    dataset_test = d.BrainMetsGraphClassification(test_df,
                    used_timepoints = data_prefixes[:i], 
                    ignored_suffixes = ('_timedelta_days', '_rano', 'Lesion ID'), 
                    rano_encoding = rano_encoding,
                    target_name = 't6_rano',
                    extra_features = patient_features,
                    fully_connected=True,
                    transforms = None,
                    direction = 'past',
                    random_period=False,
                    )
                    experiment_dir = wdir/key_year
                    os.makedirs(experiment_dir, exist_ok=True)
                    # evaluate
                    best_res = torch_engine.test_classification(best_model, dataset_test, experiment_dir, 'cuda', rano_encoding, True)
                    print(f"""Best model achieved a class weight balanced accuracy {best_res['balanced_accuracy']:4f}""")
                    
                    all_results[key_year] = best_res

                    # plot
                    plot_prediction_metrics(best_res, experiment_dir)
                plot_prediction_metrics_sweep(all_results, wdir, classes=classes, distribution=dist)
                all_folds.append(all_results)
            
            plot_prediction_metrics_sweep_fold(all_folds, output, classes=classes, distribution=dist)