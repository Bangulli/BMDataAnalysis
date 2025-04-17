## IDEAS
# patient wise loading and processing -> identify single lesions and store as csv
# 
# for each anat image in the patient segment
# identify correspondence -> maximum overlap https://www.nature.com/articles/s41467-024-52414-2
# 
# identify single lesions
# discretize each lesion into 6 timestamps by interpolating actual data into 2 month intervals -> METHOD? https://www.mdpi.com/2073-4441/9/10/796
#
# each lesion is a list of len(6) containing the volume at each time stamp
# for clustering each lesion is a list with len(6), t0 tumor volume and relative growth/shrinkage at each time stamp after t0

from core import Metastasis
from core import MetastasisTimeSeries
from core import Patient, load_patient, PatientMetCounter
import csv
import logging
import pathlib as pl
import os
import pandas as pd
from PrettyPrint import *
import ast
from visualization import *

if __name__ == '__main__':
    met_path = pl.Path('/mnt/nas6/data/Target/task_524-504_REPARSED_METS_mrct1000_nobatch') # location of preparsed metastases
    match_report = pl.Path('/home/lorenz/BMDataAnalysis/logs/504-524/metrics.csv') # location of matching report csv to filter out unmatched lesions
    match_report = pd.read_csv(match_report, sep=';', index_col=None) 
    folder_name = 'csv_nn_multiclass_reseg_only_valid' # folder in which the output is stored in the met_path directory
    os.makedirs(met_path/folder_name, exist_ok=True)


    parsed = [pat for pat in os.listdir(met_path) if pat.startswith('sub-PAT')]

    ## load mets from preparsed store
    value_dicts = []
    for pat in parsed:
        print('== loading patient:', pat)

        matched_mets = match_report.loc[match_report['patient_id']==pat, 'matched_mets'].to_list()
        if any(matched_mets[0]):
            matched_mets = ast.literal_eval(matched_mets[0])
            p = load_patient(met_path/pat)
            

            if p:
                p.discard_unmatched(list(matched_mets.values()))
                p.resample_all_timeseries(360, 6, 'nearest')

                v, keys = p.get_features('all')
            
                value_dicts += v

            else:
                print('== failed to load patient:', pat)
        else:
            print("== found no matching metastases for patient, skipped")


    with open(met_path/folder_name/'features.csv', 'w') as file:
        print(f'== extracted {len(keys)} features for {len(value_dicts)} metastases, writing to file...')
        header = keys
        writer = csv.DictWriter(file, fieldnames=header)
        writer.writeheader()
        for d in value_dicts:
            writer.writerow(d)
        print('== done')
        
    df = pd.read_csv(met_path/folder_name/'features.csv')
    ranos = [k for k in df.columns if k.startswith('rano_')]
    plot_sankey(df[ranos], met_path/folder_name)
   