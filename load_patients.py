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

if __name__ == '__main__':
    processed_path = pl.Path('/mnt/nas6/data/Target/BMPipeline_full_rerun/PROCESSED')
    met_path = pl.Path('/mnt/nas6/data/Target/BMPipeline_full_rerun/PARSED_METS_task_502') # location of preparsed metastases
    folder_name = 'csv_uninterpolated'
    os.makedirs(met_path/folder_name, exist_ok=True)
    parsed = [pat for pat in os.listdir(met_path) if pat.startswith('sub-PAT')]

    ## load mets from preparsed store
    value_dicts = []
    all_keys = []

    for pat in ['sub-PAT0049']:#, 'sub-PAT0049']:#, 'sub-PAT0020', 'sub-PAT0049']:
        print('== loading patient:', pat)
        p = load_patient(met_path/pat)
        if not p: continue # load patient returns false if loading fails. it can happen for various reasons and definitely needs some improvements to robustness, starting from the saving function, since it sometimes leaves empty directories, which shouldnt happen
        #p.drop_short_timeseries(300)
        p.discard_gaps(120, 360, False)
        p.discard_swings(420, True)
        p.resample_all_timeseries(360, 6, 'nearest')
        post_l, keys = p.get_features(['volume', 'rano', 'radiomics'], True, None)
        if keys: all_keys += [k for k in keys if k not in all_keys]

        ## write data to csv
        with open('/home/lorenz/BMDataAnalysis/without_swings', 'w') as file:
            header = all_keys
            writer = csv.DictWriter(file, fieldnames=header)
            writer.writeheader()
            for d in post_l:
                writer.writerow(d)
            print('== done')

        all_keys = []
        print('== loading patient:', pat)
        p = load_patient(met_path/pat)
        if not p: continue # load patient returns false if loading fails. it can happen for various reasons and definitely needs some improvements to robustness, starting from the saving function, since it sometimes leaves empty directories, which shouldnt happen
        #p.drop_short_timeseries(300)
        p.discard_gaps(120, 360, False)
        p.resample_all_timeseries(360, 6, 'nearest')
        pre_l, keys = p.get_features(['volume', 'rano', 'radiomics'], True, None)
        if keys: all_keys += [k for k in keys if k not in all_keys]

        ## write data to csv
        with open('/home/lorenz/BMDataAnalysis/with_swings', 'w') as file:
            header = all_keys
            writer = csv.DictWriter(file, fieldnames=header)
            writer.writeheader()
            for d in pre_l:
                writer.writerow(d)
            print('== done')





   