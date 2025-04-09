import pandas as pd
import sklearn
from sklearn.cluster import MeanShift, KMeans, DBSCAN, HDBSCAN, OPTICS

import pathlib as pl
from data import *
import numpy as np
import os
from visualization import *
from clustering import *
from stepmix.stepmix import StepMix

import warnings
warnings.filterwarnings('ignore')

if __name__ == '__main__':
    method_name = 'stepmix'
    folder_name = 'csv_linear_multiclass_reseg_only_valid'
    volume_data = pd.read_csv(f'/mnt/nas6/data/Target/task_524-504_PARSED_METS_mrct1000_nobatch/{folder_name}/volumes.csv', index_col=None)
    rano_data = pd.read_csv(f'/mnt/nas6/data/Target/task_524-504_PARSED_METS_mrct1000_nobatch/{folder_name}/rano.csv', index_col=None)
    renamer = {elem: 'rano-'+elem for elem in rano_data.columns}
    rano_data = rano_data.rename(columns=renamer)

    complete_data = pd.concat([volume_data, rano_data], axis=1)
    
    output =  pl.Path(f'/home/lorenz/BMDataAnalysis/output/{folder_name}/{method_name}_')

    k = range(2, 42)
    
    ## load volume data
    data_cols = ["60", "120", "180", "240", "300", "360"]
    rano_cols = ['rano-'+elem for elem in data_cols]
    print(f'clustering {len(complete_data)} metastases')

    der = get_derivatives(complete_data[["0"]+data_cols])
    
    # Normalize by t0 Volume
    complete_data[data_cols] = complete_data[data_cols].div(complete_data["0"], axis=0)










