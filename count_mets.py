from core import Metastasis
from core import MetastasisTimeSeries
from core import Patient, load_patient, PatientMetCounter
import csv
import pathlib as pl
import os
import numpy as np
from PrettyPrint import *
import matplotlib.pyplot as plt
if __name__ == '__main__':
    ##### source data
    # dataset_path = pl.Path('/mnt/nas6/data/Target/PROCESSED_mrct1000_nobatch')

    # pats = [pat for pat in os.listdir(dataset_path) if pat.startswith('sub-PAT')]

    # mets = []
    # for pat in pats:
    #     p = PatientMetCounter(dataset_path/pat, pl.Path('/mnt/nas6/data/Target/PROCESSED_mrct1000_nobatch/nnUNet_mapping.csv'))
    #     mets.append(p.mets)
    #     print(f"== patient {pat} has {p.mets} metastases in the source data")

    # mets = np.asarray(mets)

    # print(f"found {mets.sum()} metastases in {len(pats)} patients in the pre reseg data")

    # counts = np.bincount(mets)
    # x = np.arange(len(counts))

    # plt.figure(figsize=(12, 6))
    # plt.bar(x, counts, width=0.6, edgecolor='black') 
    # plt.xlabel("Metastases per Patient")
    # plt.ylabel("Frequency")
    # plt.title("Dataset Histogram")
    # plt.grid()
    # plt.savefig('source_dataset_patient-mets_histogram.png')

    ##### parsed resegmented data
    parsed_path = pl.Path('/mnt/nas6/data/Target/task_524-504_PARSED_METS_mrct1000_nobatch')

    pats = [pat for pat in os.listdir(parsed_path) if pat.startswith('sub-PAT')]

    mets = []
    for pat in pats:
        m = [f for f in os.listdir(parsed_path/pat) if f.startswith('Metastasis')]
        ms = 0
        for i in m:
            t0 = [f for f in os.listdir(parsed_path/pat/i) if f.startswith('t0')][0]
            if (parsed_path/pat/i/t0/'metastasis_mask_binary.nii.gz').is_file():
                ms+=1
        mets.append(ms)
        print(f"== patient {pat} has {ms} metastases in the source data")

    mets = np.asarray(mets)

    print(f"found {mets.sum()} metastases in {len(pats)} patients in the reseg and parsed data")

    counts = np.bincount(mets)
    x = np.arange(len(counts))

    plt.figure(figsize=(12, 6))
    plt.bar(x, counts, width=0.6, edgecolor='black') 
    plt.xlabel("Metastases per Patient")
    plt.ylabel("Frequency")
    plt.title("Dataset Histogram")
    plt.grid()
    plt.savefig('reseg_dataset_patient-mets_histogram.png')


     