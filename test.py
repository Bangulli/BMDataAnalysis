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


nonin = pd.read_csv('/mnt/nas6/data/Target/BMPipeline_full_rerun/PARSED_METS_task_502/csv_test_torm/train_exp_filtered_series_nonin.csv')
inter = pd.read_csv('/mnt/nas6/data/Target/BMPipeline_full_rerun/PARSED_METS_task_502/csv_nn_filtered/noswings_filtered_series.csv')

print(len(nonin), len(inter))


print('without swings filter')
nonin = pd.read_csv('/mnt/nas6/data/Target/BMPipeline_full_rerun/PARSED_METS_task_502/csv_nn_filtered/filtered_series.csv')
inter = pd.read_csv('/mnt/nas6/data/Target/BMPipeline_full_rerun/PARSED_METS_task_502/csv_test_torm/filtered_series_nonin.csv')

print(len(nonin), len(inter), all(nonin['Lesion ID'] == inter['Lesion ID']))