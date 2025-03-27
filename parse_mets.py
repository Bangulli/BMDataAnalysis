from core import Metastasis
from core import MetastasisTimeSeries
from core import Patient, load_patient, PatientMetCounter
import csv
import pathlib as pl
import os
from PrettyPrint import *

if __name__ == '__main__':
    dataset_path = pl.Path('/mnt/nas6/data/Target/PROCESSED_mrct1000_nobatch')
    met_path = pl.Path('/mnt/nas6/data/Target/PARSED_1C_METS_mrct1000_nobatch')
    
    os.makedirs(met_path, exist_ok=True)

    pats = [pat for pat in os.listdir(dataset_path) if pat.startswith('sub-PAT')]
    parsed = [pat for pat in os.listdir(met_path) if pat.startswith('sub-PAT')]
    pats = [pat for pat in pats if pat not in parsed]


    ## intially parses the metastases
    logger = Printer(log_type='txt')
    for pat in pats:
        print('== working on patient:', pat)
        p = Patient(dataset_path/pat, log=logger, met_dir_name='mets_task502')
        p.print()
        p.save(met_path)

    # ## loads preparsed metastases
    # for pat in parsed:
    #     print('== loading patient:', pat)
    #     p = load_patient(met_path/pat)
    #     if p:
    #         p.resample_all_timeseries(360, 6, 'linear')
    #     else:
    #         print('== failed to load patient:', pat)
