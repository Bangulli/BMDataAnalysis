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
from scripts.compare_segs import *

if __name__ == '__main__':
    #processed_path = pl.Path('/mnt/nas6/data/Target/BMPipeline_full_rerun/PROCESSED_lenient_inclusion')
    met_path = pl.Path('/mnt/nas6/data/Target/BMPipeline_full_rerun/PARSED_METS_task_502') # location of preparsed metastases
    #match_report = pl.Path('/home/lorenz/BMDataAnalysis/logs/229_Patients/task_502/metrics.csv') # location of matching report csv to filter out unmatched lesions
    folder_name = 'csv_nn_blabla' # folder in which the output is stored in the met_path directory
    # if match_report.is_file():
    #     match_report = pd.read_csv(match_report, sep=';', index_col=None) 
    # else:
    #     compare_segs(match_report,processed_path, met_path)
    #     match_report = pd.read_csv(match_report, sep=';', index_col=None)
    
    os.makedirs(met_path/folder_name, exist_ok=True)


    parsed = [pat for pat in os.listdir(met_path) if pat.startswith('sub-PAT')]

    ## load mets from preparsed store
    value_dicts = []
    for pat in parsed:
        print('== loading patient:', pat)

        #matched_mets = match_report.loc[match_report['patient_id']==pat, 'matched_mets'].to_list()
        # if any(matched_mets[0]):
        #     matched_mets = ast.literal_eval(matched_mets[0])
        p = load_patient(met_path/pat)
        

        if p:
            p.resample_all_timeseries(360, 6, 'nearest')
            #p.tag_unmatched(list(matched_mets.values()))
            v, keys = p.get_features(['total_load'])
        
            value_dicts += v

        else:
            print('== failed to load patient:', pat)
        # else:
        #     print("== found no matching metastases for patient, skipped")
        raise RuntimeError("stop")


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
    matched = df.loc[df['RT_matched']==True, ranos]
    plot_sankey(matched[ranos], met_path/folder_name, tag=f'{len(matched)}_matched_')
    unmatched = df.loc[df['RT_matched']==False, ranos]
    plot_sankey(unmatched[ranos], met_path/folder_name, tag=f'{len(unmatched)}_unmatched_')
   