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
import pathlib as pl
import os
from PrettyPrint import *

if __name__ == '__main__':
    dataset_path = pl.Path('/mnt/nas6/data/Target/PROCESSED_mrct1000_nobatch')
    met_path = pl.Path('/mnt/nas6/data/Target/PARSED_METS_mrct1000_nobatch')
    
    os.makedirs(met_path, exist_ok=True)

    pats = [pat for pat in os.listdir(dataset_path) if pat.startswith('sub-PAT')]
    parsed = [pat for pat in os.listdir(met_path) if pat.startswith('sub-PAT')]
    pats = [pat for pat in pats if pat not in parsed]

    ## load mets from preparsed store
    met_dicts = []

    rano_dicts = []
    for pat in parsed:
        print('== loading patient:', pat)
        
        p = load_patient(met_path/pat)

        if p:
            p.resample_all_timeseries(360, 6, 'linear')
        
            rano_dicts += p.lesion_wise_rano()
            met_dicts += p.to_dicts()
        else:
            print('== failed to load patient:', pat)


    with open(met_path/'csv_linear_multiclass_reseg'/'volumes.csv', 'w') as file:
        header = ['Lesion ID', 'Brain Volume', 0, 60, 120, 180, 240, 300, 360]
        writer = csv.DictWriter(file, fieldnames=header)
        writer.writeheader()
        for d in met_dicts:
            print(d)
            writer.writerow(d)

    with open(met_path/'csv_linear_multiclass_reseg'/'rano.csv', 'w') as file:
        header = ['Lesion ID', 0, 60, 120, 180, 240, 300, 360]
        writer = csv.DictWriter(file, fieldnames=header)
        writer.writeheader()
        for d in rano_dicts:
            print(d)
            writer.writerow(d)
        
   