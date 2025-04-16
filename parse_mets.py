from core import Metastasis
from core import MetastasisTimeSeries
from core import Patient, load_patient, PatientMetCounter
import csv
import pathlib as pl
import os
from PrettyPrint import *

if __name__ == '__main__':
    dataset_path = pl.Path('/mnt/nas6/data/Target/PROCESSED_mrct1000_nobatch')
    met_path = pl.Path('/mnt/nas6/data/Target/task_524-504_REPARSED_METS_mrct1000_nobatch')
    
    os.makedirs(met_path, exist_ok=True)

    pats = [pat for pat in os.listdir(dataset_path) if pat.startswith('sub-PAT')]
    parsed = [pat for pat in os.listdir(met_path) if pat.startswith('sub-PAT')]
    pats = [pat for pat in pats if pat not in parsed]


    ## intially parses the metastases
    logger = Printer(log_type='txt')
    for pat in pats:
        print('== working on patient:', pat)
        p = Patient(dataset_path/pat, log=logger, met_dir_name='mets')
        p.print()
        p.save(met_path)
