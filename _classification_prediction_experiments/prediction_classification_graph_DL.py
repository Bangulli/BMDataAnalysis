import sys
from pathlib import Path

# Add parent directory to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))
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
import data as d
import pandas as pd
import torch
import numpy as np
from collections import Counter
from torch_geometric.loader import DataLoader
from prediction import GCN, GAT, classification_evaluation
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from scipy.stats import zscore
import warnings
warnings.filterwarnings('ignore')

if __name__ == '__main__':
########## setup
    data_source = pl.Path('/mnt/nas6/data/Target/BMPipeline_full_rerun/229_patients faulty/PARSED_METS_task_502/csv_nn/features.csv')
    prediction_type = 'multi'
    feature_selection = 'none'
    method = 'simplest_GCN_batch'
    output_path = pl.Path(f'/home/lorenz/BMDataAnalysis/output/baselines/')
    used_features = ['volume', 'radiomics']

    if prediction_type == 'binary':
        rano_encoding={'CR':0, 'PR':0, 'SD':1, 'PD':1}
        num_out=2
    else:
        rano_encoding={'CR':0, 'PR':1, 'SD':2, 'PD':3}
        num_out=4
    
    if feature_selection == 'lasso':
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

    train_data, test_data = d.load_prepro_data(data_source,
                                        used_features=['volume'],
                                        test_size=0.2,
                                        drop_suffix=None,
                                        prefixes=data_prefixes,
                                        target_suffix='rano',
                                        normalize_suffix=None,
                                        rano_encoding={ 'CR': 0,'PR': 1,'SD': 2,'PD': 3 },
                                        time_required=True,
                                        interpolate_CR_swing_length=1,
                                        drop_CR_swing_length=2,
                                        normalize_volume='frac',
                                        save_processed=None)

    ## prepare output
    output = output_path/f'classification_{prediction_type}_{method}_featuretypes={used_features}_selection={feature_selection}'
    os.makedirs(output, exist_ok=True)
    with open(output/'used_feature_names.txt', 'w') as file:
        file.write("Used feature names left in the dataframe:\n")
        for c in train_data.columns:
            file.write(f"   - {c}\n")
        file.write("NOTE: rano columns are used as targets not as prediction")

    ## class weight definition for torch
    labels = [row['t6_rano'] for i, row in train_data.iterrows()]
    label_counts = Counter(labels)
    num_classes = len(label_counts)  # or len(label_counts)
    counts = torch.tensor([label_counts[i] for i in range(num_classes)], dtype=torch.float)
    torch_weights = 1.0 / (counts + 1e-6)  # Avoid divide by zero
    torch_weights = torch_weights / torch_weights.sum()  # Normalize (optional but nice)

    # make datasets
    dataset_train = d.BrainMetsGraphClassification(train_data,
        used_timepoints = ['t0', 't1', 't2', 't3', 't4', 't5'], 
        ignored_suffixes = ('_timedelta_days', '_rano', 'Lesion ID'), 
        rano_encoding = rano_encoding,
        target_name = 't6_rano',
        extra_features = None,
        fully_connected=False,
        direction=None,
        transforms = None,
        )
    dataset_test = d.BrainMetsGraphClassification(test_data,
        used_timepoints = ['t0', 't1', 't2', 't3', 't4', 't5'], 
        ignored_suffixes = ('_timedelta_days', '_rano', 'Lesion ID'), 
        rano_encoding = rano_encoding,
        target_name = 't6_rano',
        extra_features = None,
        fully_connected=False,
        transforms = None,
        direction = None
        )

    # init training variables
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SimplestGCN(num_out, dataset_train.get_node_size()).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=50,       # First restart after 10 epochs
        T_mult=2,     # Increase period between restarts by this factor
        eta_min=1e-6  # Minimum LR
    )

    # run training
    best_model, best_loss = torch_engine.train(model, 
                                    dataset= dataset_train, 
                                    loss_function=F.cross_entropy,
                                    epochs=500,
                                    optimizer=optimizer,
                                    scheduler=scheduler,
                                    working_dir=output,
                                    device=device,
                                    validation=0.25,
                                    batch_size=64
                                    )
    print(f"Best model achieved loss {best_loss:4f}")

    # evaluate
    best_res = torch_engine.test_classification(best_model, dataset_test, output, device)
    print(f"""Best model achieved a class weight balanced accuracy {best_res['balanced_accuracy']:4f}""")
    print(best_res['classification_report'])

    # plot
    plot_prediction_metrics(best_res, output)