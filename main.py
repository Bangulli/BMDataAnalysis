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

from src.metastasis import Metastasis
from src.metastasis_series import MetastasisTimeSeries
from src.patient import Patient

import pathlib as pl
import os

if __name__ == '__main__':
    dataset_path = pl.Path('/mnt/nas6/data/Target/batch_copy/reseg_test/set')
    pats = [pat for pat in os.listdir(dataset_path) if pat.startswith('sub-PAT')]
    dataset_patients = []
    for pat in pats:
        p = Patient(dataset_path/pat)
        p.print()
        dataset_patients.append(p)
        break

    for i, series in enumerate(dataset_patients[0].mets):
        new = series.resample()
        if new is not None:
            print('resampled series:', i)
            new.print()

   