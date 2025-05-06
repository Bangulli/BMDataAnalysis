from core import Metastasis
from core import MetastasisTimeSeries
from core import Patient, load_patient, PatientMetCounter
import csv
import pathlib as pl
import os
from PrettyPrint import *


if __name__ == '__main__':
    dataset_path = pl.Path('/mnt/nas6/data/Target/BMPipeline_full_rerun/229_patients faulty/PROCESSED_lenient_inclusion')
    met_path = pl.Path('/mnt/nas6/data/Target/BMPipeline_full_rerun/experimental')
    
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
        p.save(met_path)

    