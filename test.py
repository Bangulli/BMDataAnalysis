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
from typing import Any, List, Tuple, Union

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from lightgbm import LGBMClassifier
import torch_geometric.transforms as T
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
from torch_geometric.utils import to_networkx
import networkx as nx



if __name__ == '__main__':
        data_prefixes = ["t0", "t1", "t2", "t3", "t4", "t5", "t6"] # used in the training method to select the features for each step of the sweep
        volume_cols = [c+'_volume' for c in data_prefixes] # used to normalize the volumes
        rano_cols = [elem+'_rano' for elem in data_prefixes] # used in the training method to select the current targets

        train_data, test_data = d.load_prepro_data('/mnt/nas6/data/Target/BMPipeline_full_rerun/PARSED_METS_task_502/csv_nn/features.csv',
                                            categorical=[],
                                            fill=0,
                                            used_features=['volume'],
                                            test_size=0.2,
                                            drop_suffix=None,
                                            prefixes=data_prefixes,
                                            target_suffix='rano',
                                            normalize_suffix=None,
                                            rano_encoding={'CR':0, 'PR':1, 'SD':2, 'PD':3},
                                            time_required=True,
                                            interpolate_CR_swing_length=1,
                                            drop_CR_swing_length=2,
                                            normalize_volume='std',
                                            save_processed=None)
        extra_data = [c for c in train_data.columns if not (c.startswith('ignored') or c.split('_')[0] in data_prefixes)]
        print("using extra data cols", extra_data)
        dist = Counter(test_data['t6_rano'])

        ## class weight definition for torch
        labels = [row['t6_rano'] for i, row in train_data.iterrows()]
        label_counts = Counter(labels)
        num_classes = len(label_counts)  # or len(label_counts)
        counts = torch.tensor([label_counts[i] for i in range(num_classes)], dtype=torch.float)
        torch_weights = 1.0 / (counts + 1e-6)  # Avoid divide by zero
        torch_weights = torch_weights / torch_weights.sum()  # Normalize (optional but nice)


        # make datasets
        dataset_train = d.BrainMetsGraphClassification(train_data,
            used_timepoints = data_prefixes[:4], 
            ignored_suffixes = ('_timedelta_days', '_rano', 'Lesion ID'), 
            rano_encoding = {'CR':0, 'PR':1, 'SD':2, 'PD':3},
            target_name = 't6_rano',
            extra_features = extra_data,
            fully_connected=True,
            direction='future',
            transforms = None)
        
        G = to_networkx(dataset_train[0])
        plt.figure(figsize=(8, 6))
        nx.draw(G, with_labels=True, cmap='Set2')
        plt.savefig("/home/lorenz/BMDataAnalysis/output/graph_visualization_fully_future.png", dpi=300, bbox_inches='tight')
        plt.close()  # optional: closes the figure to avoid displaying it in Jupyter
