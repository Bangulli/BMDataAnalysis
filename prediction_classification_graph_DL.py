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
from prediction import GraphClassificationModel, GCN, GAT, classification_evaluation
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from scipy.stats import zscore
import warnings
warnings.filterwarnings('ignore')

if __name__ == '__main__':
########## setup
    data_source = pl.Path('/mnt/nas6/data/Target/BMPipeline_DEVELOPMENT_runs/task_502_PARSED_METS_mrct1000_nobatch/csv_nn_only_valid/features.csv')
    prediction_type = 'multi'
    method = 'GAT'
    output_path = pl.Path(f'/home/lorenz/BMDataAnalysis/output/{data_source.parent.name}')
    used_features = ['volume', 'timedelta_days']

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

    ([data.drop(columns=c, inplace=True, axis=0) for c in data.columns if c not in to_keep])
  
    output = output_path/f'classification_{prediction_type}_{method}_features={used_features}'
    os.makedirs(output, exist_ok=True)

    ## dataset splitting
    labels = [d[rano_cols[-1]] for i, d in data.iterrows()]
    train, test = train_test_split(data, test_size=0.2, random_state=42, stratify=labels)

    ## class weight definition for torch
    labels = [row['t6_rano'] for i, row in train.iterrows()]
    label_counts = Counter(labels)
    num_classes = len(label_counts)  # or len(label_counts)
    counts = torch.tensor([label_counts[i] for i in range(num_classes)], dtype=torch.float)
    torch_weights = 1.0 / (counts + 1e-6)  # Avoid divide by zero
    torch_weights = torch_weights / torch_weights.sum()  # Normalize (optional but nice)


    dataset_train = d.BrainMetsGraphDataset(train,
        used_timepoints = ['t0', 't1', 't2', 't3', 't4', 't5'], 
        ignored_suffixes = ('_timedelta_days', '_rano', 'Lesion ID'), 
        rano_encoding = rano_encoding,
        target_name = 't6_rano',
        extra_features = None,
        fully_connected=False,
        transforms = None,
        )
    dataset_test = d.BrainMetsGraphDataset(test,
        used_timepoints = ['t0', 't1', 't2', 't3', 't4', 't5'], 
        ignored_suffixes = ('_timedelta_days', '_rano', 'Lesion ID'), 
        rano_encoding = rano_encoding,
        target_name = 't6_rano',
        extra_features = None,
        fully_connected=False,
        transforms = None,
        )

    train_loader = DataLoader(dataset_train, batch_size=64, shuffle=True)
    test_loader = DataLoader(dataset_test, batch_size=len(dataset_test))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = GAT(4, dataset_train.get_node_size()).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=10,       # First restart after 10 epochs
        T_mult=2,     # Increase period between restarts by this factor
        eta_min=1e-6  # Minimum LR
    )

    model.train()
    best_acc = 0
    best_res = None
    for epoch in range(200):
        print(f'== epoch {epoch}/200')
        # train step
        model.train()
        for batch in train_loader:
            batch.to(device)
            optimizer.zero_grad()
            out = model(batch)
            loss = F.cross_entropy(out, batch.y, weight=torch_weights.to(device))
            print(loss)
            #print(loss)
            loss.backward()
            optimizer.step()
            scheduler.step()
        # validation step
        model.eval()
        for batch in test_loader:
            batch.to(device)
            pred = model(batch).argmax(dim=1).cpu().numpy()
            gt = batch.y.cpu().numpy()
            res = classification_evaluation(gt, pred)
            acc = res['balanced_accuracy']
            print('current accuracy', acc)
            if acc > best_acc:
                best_acc = acc
                best_res = res
    print(f"Best model achieved accuracy {best_acc:4f}")
    print(best_res['classification_report'])

    plot_prediction_metrics(res, output)