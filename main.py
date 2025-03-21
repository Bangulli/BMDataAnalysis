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

import pathlib as pl
import os

if __name__ == '__main__':
    dataset_path = pl.Path('/mnt/nas6/data/Target/PROCESSED_mrct1000_nobatch')
    met_path = pl.Path('/mnt/nas6/data/Target/batch_copy/reseg_test/parsed_mets')
    os.makedirs(met_path, exist_ok=True)

    pats = [pat for pat in os.listdir(dataset_path) if pat.startswith('sub-PAT')]
    dataset_mets = 0
    for pat in pats:
        print('searching patient: ', pat)
        p = PatientMetCounter(dataset_path/pat, dataset_path/'nnUNet_mapping.csv')
        dataset_mets += p.mets
        print(f'found {p.mets} metastases')

    print(f'Found {dataset_mets} Metastases from RTs in {len(pats)} patients')

    # pats = [pat for pat in os.listdir(dataset_path) if pat.startswith('sub-PAT')]
    # dataset_patients = []
    # for pat in pats:
    #     p = Patient(dataset_path/pat)
    #     p.print()
    #     p.save(met_path)
    #     dataset_patients.append(p)

    # loaded_patients = []
    # for pat in pats:
    #     p = load_patient(met_path/pat)
    #     print(p.mets)
    #     p.print()
    #     loaded_patients.append(p)
    #     break

    # for met in loaded_patients:
    #     for i, series in enumerate(met.mets):
    #         new = series.resample(method='nearest')
    #         if new is not None:
    #             print('nearest neighbor resampled series:', i)
    #             new.print()
    #             #new._plot_trajectory_comparison(f"comparison - patient-{'x'}_metastasis-{i}_nn_interpolation.png", series)

    #         new = series.resample(method='linear')
    #         if new is not None:
    #             print('linear resampled series:', i)
    #             new.print()
    #             #new._plot_trajectory_comparison(f"comparison - patient-{'x'}_metastasis-{i}_linear_interpolation.png", series)

            
    #         new = series.resample(method='bspline')
    #         if new is not None:
    #             print('bspline resampled series:', i)
    #             new.print()
    #             #new._plot_trajectory_comparison(f"comparison - patient-{'x'}_metastasis-{i}_bspline_interpolation.png", series)
   