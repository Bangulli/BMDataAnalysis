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
from PIL import Image
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
from scipy.ndimage import center_of_mass 
from core import *
import SimpleITK as sitk

def extract_axial_slice(image, com, size=64):
    # Convert physical COM to index space
    com_index = np.asarray(com, dtype=int)

    # Determine the bounding box (x and y) around the center index
    half_size = size // 2

    # Convert to numpy and squeeze to get a 2D image
    arr = sitk.GetArrayFromImage(image)
    slice_2d = arr[com_index[0]-half_size:com_index[0]+half_size, com_index[1]-half_size:com_index[1]+half_size, com_index[2]]

    return slice_2d

lesion_path = pl.Path('/mnt/nas6/data/Target/BMPipeline_full_rerun/PARSED_METS_task_502/sub-PAT0122/Metastasis 0')
output_path = pl.Path('/home/lorenz/BMDataAnalysis/logs')

ts = load_series(lesion_path)

for tp in range(min(len(ts), 7)):
    met, dt, k = ts[tp] #met is metastasis object, dt is datetime object, k is datetime string
    try:
        com = center_of_mass(met.image)
    except:
        pass
    slice_2d = extract_axial_slice(met.get_t1_image(), com, 64)


    slice_norm = slice_2d - np.min(slice_2d)
    if np.max(slice_norm) > 0:
        slice_norm = slice_norm / np.max(slice_norm)
    slice_uint8 = (slice_norm * 255).astype(np.uint8)

    # Convert to PIL Image and save
    img = Image.fromarray(slice_uint8)
    os.makedirs(output_path/lesion_path.parent.name, exist_ok = True)
    img.save(output_path/lesion_path.parent.name/f"t{tp}.png")

