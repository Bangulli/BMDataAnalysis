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
from core import Patient, load_patient
import csv
import logging
import pathlib as pl
import os
import pandas as pd
from PrettyPrint import *
import ast
from visualization import *
from misc_scripts.compare_segs import *
import deep_features

if __name__ == '__main__':
    processed_path = pl.Path('/mnt/nas6/data/Target/BMPipeline_full_rerun/PROCESSED')
    met_path = pl.Path('/mnt/nas6/data/Target/BMPipeline_full_rerun/PARSED_METS_task_502') # location of preparsed metastases
    folder_name = 'final_extraction'
    file_name = 'none.csv'
    extractor = deep_features.get_vincent_encoder()
    os.makedirs(met_path/folder_name, exist_ok=True)
    parsed = [pat for pat in os.listdir(met_path) if pat.startswith('sub-PAT')]

    ## load mets from preparsed store
    value_dicts = []
    all_keys = []
    for pat in parsed:
        print('== loading patient:', pat)
        p = load_patient(met_path/pat)
        if not p: continue # load patient returns false if loading fails. it can happen for various reasons and definitely needs some improvements to robustness, starting from the saving function, since it sometimes leaves empty directories, which shouldnt happen
        print('== resampling patient:', pat)
        # p.discard_gaps(120, 360, False)
        # p.discard_swings(420, False)
        #p.resample_all_timeseries(360, 6, 'nearest')
        p.drop_short_timeseries(330)
        print('== extracting features for patient:', pat)
        v, keys = p.get_features(['all'], deep_extractor=extractor)
        if keys: all_keys += [k for k in keys if k not in all_keys] # this is going to cost a lot of time but is necessary for noninterpolated extraction, because the feature dicts will be of variable length so the csv dict writer needs to get all feature keys to make sure it works
        value_dicts += v

    
    ## write data to csv
    with open(met_path/folder_name/file_name, 'w') as file:
        print(f'== extracted {len(all_keys)} features for {len(value_dicts)} metastases, writing to file...')
        header = all_keys
        writer = csv.DictWriter(file, fieldnames=header)
        writer.writeheader()
        for d in value_dicts:
            writer.writerow(d)
        print('== done')
        
    df = pd.read_csv(met_path/folder_name/file_name)
    ranos = [k for k in df.columns if k.endswith('_rano') and not k.startswith('t0')]
    print(ranos)
    plot_sankey(df[ranos], met_path/folder_name, tag='none_')

   