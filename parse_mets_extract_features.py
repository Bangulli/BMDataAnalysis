from core import Metastasis
from core import MetastasisTimeSeries
from core import Patient, load_patient, PatientMetCounter
import csv
import pathlib as pl
import os
from PrettyPrint import *
from scripts.compare_segs import *
import pandas as pd
import ast
from visualization import *


if __name__ == '__main__':
    dataset_path = pl.Path('/mnt/nas6/data/Target/BMPipeline_full_rerun/PROCESSED')
    met_path = pl.Path('/mnt/nas6/data/Target/BMPipeline_full_rerun/PARSED_METS_task_502')
    folder_name = 'csv_nn_experiments' # folder in which the output is stored in the met_path directory
    os.makedirs(met_path/folder_name, exist_ok=True)
    #match_report = pl.Path('/home/lorenz/BMDataAnalysis/logs/229_Patients/task_502/metrics.csv')
    
    os.makedirs(met_path, exist_ok=True)

    pats = [pat for pat in os.listdir(dataset_path) if pat.startswith('sub-PAT')]
    parsed = [pat for pat in os.listdir(met_path) if pat.startswith('sub-PAT')]
    pats = [pat for pat in pats if pat not in parsed]


    ## intially parses the metastases
    logger = Printer(log_type='txt')
    for pat in pats:
        print('== working on patient:', pat)
        p = Patient(dataset_path/pat, log=logger, met_dir_name='mets_task_502')
        p.validate()
        p.print()
        print('== saving patient:', pat)
        p.save(met_path)
        print('== resampling patient:', pat)
        p.resample_all_timeseries(360, 6, 'nearest')
        print('== extracting features for patient:', pat)

        v, keys = p.get_features('total_load')
        print(f"got features {v} with names {keys}")
        value_dicts += v



    with open(met_path/folder_name/'features.csv', 'w') as file:
        print(f'== extracted {len(keys)} features for {len(value_dicts)} metastases, writing to file...')
        header = keys
        writer = csv.DictWriter(file, fieldnames=header)
        writer.writeheader()
        for d in value_dicts:
            writer.writerow(d)
        print('== done')
        
    df = pd.read_csv(met_path/folder_name/'features.csv')
    ranos = [k for k in df.columns if k.endswith('_rano') and not k.startswith('t0')]
    print(ranos)
    plot_sankey(df[ranos], met_path/folder_name, tag='all_')
   

   ### patient 17 has no data, check that