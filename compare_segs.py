from core import Metastasis, load_metastasis
from core import MetastasisTimeSeries
from core import Patient, load_patient, PatientMetCounter
import csv
import pathlib as pl
import os
import numpy as np
from PrettyPrint import *
import matplotlib.pyplot as plt

def correspondence_metrics(met, gt_mask, gt_sitk):
        """
        computes the measurements that are used to identify a lesion as correspondence or not
        Returns the overlap between the target and the closest lesion in time in the series
        Returns the centroid distance between the target and the closest lesion in time in the series
        Returns the diameter of a perfect sphere with the volume of the closest lesion
        These measurements can then be used to match the lesions later on.

        returns [overlap, centroid_distance(mm), ref_diameter]
        """
        if met.image is None : # return extreme values if the met to check is empty
            print('Got empty metastasis')
            return [0, np.Infinity, 0]
        
        v_vol = gt_sitk.GetSpacing()

        if not met.voxel_spacing == v_vol: # move target to the same space as the reference
            print('Reference and Target do not share the same space, falling back to resampling target.')
            met.resample(gt_sitk)

        v_vol = v_vol[0]*v_vol[1]*v_vol[2]

        metric_list = []
        for l in np.unique(gt_mask):
            if l == 0:
                #skip backgorund
                continue

            overlap = np.sum(np.bitwise_and(met.image, gt_mask==l))
            target_centroid = center_of_mass(gt_mask==l)
            candidate_centroid = center_of_mass(met.image)
            centroid_distance = np.sqrt(np.sum((np.asarray(target_centroid)-np.asarray(candidate_centroid))**2))
            candidate_radius = ((np.sum(gt_mask==l)*v_vol) / ((4/3) * np.pi)) ** (1/3)
        return overlap, centroid_distance, candidate_radius*2

if __name__ == '__main__':
    ##### source data
    dataset_path = pl.Path('/mnt/nas6/data/Target/PROCESSED_mrct1000_nobatch')
    parsed_path = pl.Path('/mnt/nas6/data/Target/task_524-504_PARSED_METS_mrct1000_nobatch')

    pats = [pat for pat in os.listdir(dataset_path) if pat.startswith('sub-PAT')]

    met_sourcs = []
    met_parsed = []
    for pat in pats:
        p = PatientMetCounter(dataset_path/pat, pl.Path('/mnt/nas6/data/Target/PROCESSED_mrct1000_nobatch/nnUNet_mapping.csv'))
        met_sourcs.append(p.mets)
        print(f"== patient {pat} has {p.mets} metastases in the source data")
        m = [f for f in os.listdir(parsed_path/pat) if f.startswith('Metastasis')]
        ms = 0
        met_objects = []
        for i in m:
            t0 = [f for f in os.listdir(parsed_path/pat/i) if f.startswith('t0')][0]
            if (parsed_path/pat/i/t0/'metastasis_mask_binary.nii.gz').is_file():
                ms+=1
            met_objects.append(load_metastasis(parsed_path/pat/i/t0))

        met_parsed.append(ms)
        print(f"== patient {pat} has {ms} metastases in the parsed data")


        gt = p.labels

        

    met_sourcs = np.asarray(met_sourcs)

    print(f"found {met_sourcs.sum()} metastases in {len(pats)} patients in the pre reseg data")

    met_parsed = np.asarray(met_parsed)

    print(f"found {met_parsed.sum()} metastases in {len(pats)} patients in the reseg and parsed data")


    disparity = []
    with open('met_disparity.txt', 'w') as file:
        for i, pat in enumerate(pats):
            disparity.append(met_sourcs[i]-met_parsed[i])
            print(f'{pat} has {met_sourcs[i]} in source and {met_parsed[i]} in parsed, disparity of {met_sourcs[i]-met_parsed[i]}')
            file.writelines(f'{pat} has {met_sourcs[i]} in source and {met_parsed[i]} in parsed, disparity of {met_sourcs[i]-met_parsed[i]}\n')
    
    disp = np.asarray(disparity)

    print(f"Reseg introduced {np.sum(disp[disp<0])} mets not in source")
    print(f"Reseg matched {np.sum(disp==0)} patients perfectly")
    print(f"Reseg missed {np.sum(disp[disp>0])} mets in source")
