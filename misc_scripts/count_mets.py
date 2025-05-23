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
    dataset_path = pl.Path('/mnt/nas6/data/Target/BMPipeline_full_rerun/PROCESSED_lenient_inclusion')
    parsed_path = pl.Path('/mnt/nas6/data/Target/BMPipeline_full_rerun/PARSED_METS_task_502')

    pats = [pat for pat in os.listdir(dataset_path) if pat.startswith('sub-PAT')]

    met_sourcs = []
    mets = []
    for pat in pats:
        p = PatientMetCounter(dataset_path/pat, pl.Path(dataset_path/'nnUNet_mapping.csv'))
        met_sourcs.append(p.mets)
        print(f"== patient {pat} has {p.mets} metastases in the source data")
        m = [f for f in os.listdir(parsed_path/pat) if f.startswith('Metastasis')]
        ms = 0
        for i in m:
            t0 = [f for f in os.listdir(parsed_path/pat/i) if f.startswith('t0')][0]
            if (parsed_path/pat/i/t0/'metastasis_mask_binary.nii.gz').is_file():
                ms+=1
        mets.append(ms)
        print(f"== patient {pat} has {ms} metastases in the parsed data")
        

    met_sourcs = np.asarray(met_sourcs)

    print(f"found {met_sourcs.sum()} metastases in {len(pats)} patients in the pre reseg data")

    counts = np.bincount(met_sourcs)
    x = np.arange(len(counts))

    plt.figure(figsize=(12, 6))
    plt.bar(x, counts, width=0.6, edgecolor='black') 
    plt.xlabel("Metastases per Patient")
    plt.ylabel("Frequency")
    plt.title("Dataset Histogram")
    plt.grid()
    plt.savefig('source_dataset_patient-mets_histogram.png')

    ##### parsed resegmented data
    

    
 
        

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

    disparity = []
    with open('met_disparity.txt', 'w') as file:
        for i, pat in enumerate(pats):
            disparity.append(met_sourcs[i]-mets[i])
            print(f'{pat} has {met_sourcs[i]} in source and {mets[i]} in parsed, disparity of {met_sourcs[i]-mets[i]}')
            file.writelines(f'{pat} has {met_sourcs[i]} in source and {mets[i]} in parsed, disparity of {met_sourcs[i]-mets[i]}\n')
    
    disp = np.asarray(disparity)

    print(f"Reseg introduced {np.sum(disp[disp<0])} mets not in source")
    print(f"Reseg matched {np.sum(disp==0)} patients perfectly")
    print(f"Reseg missed {np.sum(disp[disp>0])} mets in source")
    print(f"Raw data has {np.sum(met_sourcs)} mets")
    print(f"Reseg has {np.sum(mets)} mets")
