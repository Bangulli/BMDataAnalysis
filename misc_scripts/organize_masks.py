from core import Metastasis
from core import MetastasisTimeSeries
from core import Patient, load_patient, PatientMetCounter
import csv
import pathlib as pl
import os
import numpy as np
from PrettyPrint import *
import matplotlib.pyplot as plt
import pandas as pd
import SimpleITK as sitk

if __name__ == '__main__':
    ##### source data
    task = '524-504'
    dataset_path = pl.Path('/mnt/nas6/data/Target/PROCESSED_mrct1000_nobatch')
    parsed_path = pl.Path(f'/mnt/nas6/data/Target/task_{task}_PARSED_METS_mrct1000_nobatch')
    mask_path = pl.Path(f'/mnt/nas6/data/Target/temp_parse_to_gt_comparison')
    os.makedirs(mask_path, exist_ok=True)
    pats = [pat for pat in os.listdir(dataset_path) if pat.startswith('sub-PAT')]

    met_sourcs = []
    mets = []
    for pat in pats:
        print('working on pat:', pat)
        p = PatientMetCounter(dataset_path/pat, pl.Path('/mnt/nas6/data/Target/PROCESSED_mrct1000_nobatch/nnUNet_mapping.csv'))
        msk = p.mask
        sitk.WriteImage(msk, mask_path/f'{pat}_rt_t-1.nii.gz')