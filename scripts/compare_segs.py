from core import Metastasis, load_metastasis
from core import MetastasisTimeSeries
from core import Patient, load_patient, PatientMetCounter
import csv
import pathlib as pl
import os
import SimpleITK as sitk
import numpy as np
from PrettyPrint import *
import matplotlib.pyplot as plt
from scipy.ndimage import center_of_mass

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

            metric_list.append([overlap, centroid_distance, candidate_radius*2])

        return metric_list

def find_best_series_index(metrics, method: str='both'):
    """
    Identifies the best matchin lesion series by finding the maximum overlap, minimum centroid distance
    metrics is a list of lists where the index in the list corresponds to the label in the mask the sub list has 3 elements[overlap, centroid distance, reference metastasis radius]
    mode 'overlap': uses maximum overlap
    mode 'centroid': uses minimum centroid distance, if less than lesion diameter
    mode 'both': finds the lesion with max overlap and breaks ties with centroid distance, if no overlap just uses centroid distance
                
    """
    
    if 'overlap' == method:
        best_series_overlap = None
        best_overlap = 0
        for i, m in enumerate(metrics):
            if m[0] > best_overlap: # look for maximum overlap
                best_overlap = m[0]
                best_series_overlap = i
        return best_series_overlap
    
    elif 'centroid' == method:
        best_series_distance = None
        best_dist = np.Infinity
        for i, m in enumerate(metrics):
            if m[1] < best_dist and m[1]<=min(m[2], 6): # look for minimum distance and distance less then ref diameter
                best_dist=m[1]
                best_series_distance = i
        return best_series_distance
    
    else:
        best_series = None
        best_overlap = 0
        best_dist = np.Infinity 
        # frist try overlap
        for i, m in enumerate(metrics):

            if m[0] > best_overlap: # look for maximum overlap
                best_series = i
                best_dist = m[1]
                best_overlap = m[0]
            elif m[0] == best_overlap: # if maximum overlap ambiguous use distance
                if m[1] < best_dist and m[1]<=min(m[2], 6):
                    best_series = i
                    best_dist = m[1]

        # if overlap fails use distance instead
        if best_series is None:
            for i, m in enumerate(metrics):
                if m[1] < best_dist and m[1]<=min(m[2], 6): # look for minimum distance and distance less then ref diameter
                    best_series = i

        return best_series
    
def compare_segs(output, dataset, parsed):
    os.makedirs(output.parent, exist_ok=True)
    pats = [pat for pat in os.listdir(dataset) if pat.startswith('sub-PAT')]

    met_sourcs = []
    met_parsed = []


    ambiguous_matches = 0
    matches = 0
    parsed_unmatched = 0
    gt_unmatched = 0
    t0_zero_parsed = 0

    with open(output.parent/'matching_report.txt', 'w') as report:
        with open(output, 'w') as mcsv:
            header = [
                'patient_id',
                'labels_in_gt',
                'parsed_mets',
                't0_nonzero_parsed_mets',
                'matched_mets',
                'unmatched_parsed_mets',
                'unmatched_gt_mets',
                'ambiguoug_parsed_matches',
                't0_zero_mets'
            ]

            writer = csv.DictWriter(mcsv, fieldnames=header, delimiter=';')
            writer.writeheader()
            for pat in pats:
                p = PatientMetCounter(dataset/pat, pl.Path(dataset/'nnUNet_mapping.csv'))
                
                met_sourcs.append(p.mets)
                #print(f"== patient {pat} has {p.mets} metastases in the source data")
                m = [f for f in os.listdir(parsed/pat) if f.startswith('Metastasis')]
                ms = 0
                met_objects = []
                for i in m:
                    t0 = [f for f in os.listdir(parsed/pat/i) if f.startswith('t0')][0]
                    if (parsed/pat/i/t0/'metastasis_mask_binary.nii.gz').is_file():
                        ms+=1
                    met_objects.append(load_metastasis(parsed/pat/i/t0))

                met_parsed.append(ms)
                #print(f"== patient {pat} has {ms} metastases in the parsed data")
                gt_sitk = sitk.GetImageFromArray(p.labels)
                gt_sitk.CopyInformation(p.mask)

                report_dict = {}
                report_dict['patient_id'] = pat
                report_dict['labels_in_gt'] = np.max(p.labels)
                report_dict['parsed_mets'] = len(met_objects)
                report_dict['t0_nonzero_parsed_mets'] = ms
                report_dict['matched_mets'] = {}
                report_dict['unmatched_parsed_mets'] = []
                report_dict['unmatched_gt_mets'] = []
                report_dict['ambiguoug_parsed_matches'] = {}
                report_dict['t0_zero_mets'] = []

                print(f"=== Working on Patient {pat}")
                for metastasis_object in met_objects:
                    if np.sum(p.labels) == 0:
                        report_dict['t0_zero_mets'].append(metastasis_object.t1_path.parent.parent.name)
                        string = f"Metastasis {metastasis_object.t1_path.parent.parent.name} is empty at t0"
                    
                    else:
                        metrics = correspondence_metrics(metastasis_object, p.labels, p.mask)
                        
                        best_idx = find_best_series_index(metrics, 'both') 
                        
                        best_idx = best_idx + 1 if best_idx is not None else None

                        if best_idx is not None and best_idx not in list(report_dict['matched_mets'].keys()):
                            report_dict['matched_mets'][best_idx] = metastasis_object.t1_path.parent.parent.name
                            string = f"Patient {pat} Metastasis {metastasis_object.t1_path.parent.parent.name} matches label {best_idx} with metrics {metrics[best_idx-1]} in the ground truth label\n"
                        elif best_idx is not None and best_idx not in list(report_dict['matched_mets'].keys()):
                            if not best_idx in list(report_dict['ambiguoug_parsed_matches'].keys()):
                                report_dict['ambiguoug_parsed_matches'][best_idx] = [metastasis_object.t1_path.parent.parent.name, report_dict['matched_mets'][best_idx]]
                            else:
                                report_dict['ambiguoug_parsed_matches'][best_idx].append(metastasis_object.t1_path.parent.parent.name)
                            string = f"Patient {pat} Metastasis at location {metastasis_object.t1_path.parent.parent.name} has matches with gt label that is already matched: {best_idx}, previous matching lesion is {report_dict['matched_mets'][best_idx]}"
                        else:
                            report_dict['unmatched_parsed_mets'].append(metastasis_object.t1_path.parent.parent.name)
                            string = f"Patient {pat} Metastasis at location {metastasis_object.t1_path.parent.parent.name} has no match in the ground truth\n"

                    print(string)
                    report.writelines(string)

                for label in np.unique(p.labels):
                    if label == 0:
                        continue
                    elif label not in list(report_dict['matched_mets'].keys()):
                        report_dict['unmatched_gt_mets'].append(label)

                print(report_dict)
                writer.writerow(report_dict)

                matches += len(report_dict['matched_mets'].keys())
                gt_unmatched += len(report_dict['unmatched_gt_mets'])
                parsed_unmatched += len(report_dict['unmatched_parsed_mets'])
                t0_zero_parsed += len(report_dict['t0_zero_mets'])
                ambiguous_matches += len(report_dict['ambiguoug_parsed_matches'].keys())
    
    print(f"Found {matches} matching metastases in gt and parsing")
    print(f"Found {gt_unmatched} metastases in the GT that are not found in parsing")
    print(f"Found {parsed_unmatched} metastases in parsing that are not found in GT")
    print(f"Found {t0_zero_parsed} parsed metastases that are empty at their first occurance")
    print(f"Found {ambiguous_matches} GT metastases that have ambiguoug matches in the parsing")

        

    met_sourcs = np.asarray(met_sourcs)

    print(f"found {met_sourcs.sum()} metastases in {len(pats)} patients in the pre reseg data")

    met_parsed = np.asarray(met_parsed)

    print(f"found {met_parsed.sum()} metastases in {len(pats)} patients in the reseg and parsed data")


    disparity = []
    with open(output.parent/'met_disparity.txt', 'w') as file:
        for i, pat in enumerate(pats):
            disparity.append(met_sourcs[i]-met_parsed[i])
            print(f'{pat} has {met_sourcs[i]} in source and {met_parsed[i]} in parsed, disparity of {met_sourcs[i]-met_parsed[i]}')
            file.writelines(f'{pat} has {met_sourcs[i]} in source and {met_parsed[i]} in parsed, disparity of {met_sourcs[i]-met_parsed[i]}\n')
    
    disp = np.asarray(disparity)

    print(f"Reseg introduced {np.sum(disp[disp<0])} mets not in source")
    print(f"Reseg matched {np.sum(disp==0)} patients perfectly")
    print(f"Reseg missed {np.sum(disp[disp>0])} mets in source")
    
if __name__ == '__main__':
    ##### source data
    dataset_path = pl.Path('/mnt/nas6/data/Target/PROCESSED_lenient_inclusion')
    parsed_path = pl.Path('/mnt/nas6/data/Target/task_502_PARSED_METS_mrct1000_nobatch')


    pats = [pat for pat in os.listdir(dataset_path) if pat.startswith('sub-PAT')]

    met_sourcs = []
    met_parsed = []


    ambiguous_matches = 0
    matches = 0
    parsed_unmatched = 0
    gt_unmatched = 0
    t0_zero_parsed = 0

    with open('/home/lorenz/BMDataAnalysis/logs/502/matching_report.txt', 'w') as report:
        with open('/home/lorenz/BMDataAnalysis/logs/502/metrics.csv', 'w') as mcsv:
            header = [
                'patient_id',
                'labels_in_gt',
                'parsed_mets',
                't0_nonzero_parsed_mets',
                'matched_mets',
                'unmatched_parsed_mets',
                'unmatched_gt_mets',
                'ambiguoug_parsed_matches',
                't0_zero_mets'
            ]

            writer = csv.DictWriter(mcsv, fieldnames=header, delimiter=';')
            writer.writeheader()
            for pat in pats:
                p = PatientMetCounter(dataset_path/pat, pl.Path('/mnt/nas6/data/Target/PROCESSED_mrct1000_nobatch/nnUNet_mapping.csv'))
                met_sourcs.append(p.mets)
                #print(f"== patient {pat} has {p.mets} metastases in the source data")
                m = [f for f in os.listdir(parsed_path/pat) if f.startswith('Metastasis')]
                ms = 0
                met_objects = []
                for i in m:
                    t0 = [f for f in os.listdir(parsed_path/pat/i) if f.startswith('t0')][0]
                    if (parsed_path/pat/i/t0/'metastasis_mask_binary.nii.gz').is_file():
                        ms+=1
                    met_objects.append(load_metastasis(parsed_path/pat/i/t0))

                met_parsed.append(ms)
                #print(f"== patient {pat} has {ms} metastases in the parsed data")
                gt_sitk = sitk.GetImageFromArray(p.labels)
                gt_sitk.CopyInformation(p.mask)

                report_dict = {}
                report_dict['patient_id'] = pat
                report_dict['labels_in_gt'] = np.max(p.labels)
                report_dict['parsed_mets'] = len(met_objects)
                report_dict['t0_nonzero_parsed_mets'] = ms
                report_dict['matched_mets'] = {}
                report_dict['unmatched_parsed_mets'] = []
                report_dict['unmatched_gt_mets'] = []
                report_dict['ambiguoug_parsed_matches'] = {}
                report_dict['t0_zero_mets'] = []

                print(f"=== Working on Patient {pat}")
                for metastasis_object in met_objects:
                    if np.sum(p.labels) == 0:
                        report_dict['t0_zero_mets'].append(metastasis_object.t1_path.parent.parent.name)
                        string = f"Metastasis {metastasis_object.t1_path.parent.parent.name} is empty at t0"
                    
                    else:
                        metrics = correspondence_metrics(metastasis_object, p.labels, p.mask)
                        
                        best_idx = find_best_series_index(metrics, 'both') 
                        
                        best_idx = best_idx + 1 if best_idx is not None else None

                        if best_idx is not None and best_idx not in list(report_dict['matched_mets'].keys()):
                            report_dict['matched_mets'][best_idx] = metastasis_object.t1_path.parent.parent.name
                            string = f"Patient {pat} Metastasis {metastasis_object.t1_path.parent.parent.name} matches label {best_idx} with metrics {metrics[best_idx-1]} in the ground truth label\n"
                        elif best_idx is not None and best_idx not in list(report_dict['matched_mets'].keys()):
                            if not best_idx in list(report_dict['ambiguoug_parsed_matches'].keys()):
                                report_dict['ambiguoug_parsed_matches'][best_idx] = [metastasis_object.t1_path.parent.parent.name, report_dict['matched_mets'][best_idx]]
                            else:
                                report_dict['ambiguoug_parsed_matches'][best_idx].append(metastasis_object.t1_path.parent.parent.name)
                            string = f"Patient {pat} Metastasis at location {metastasis_object.t1_path.parent.parent.name} has matches with gt label that is already matched: {best_idx}, previous matching lesion is {report_dict['matched_mets'][best_idx]}"
                        else:
                            report_dict['unmatched_parsed_mets'].append(metastasis_object.t1_path.parent.parent.name)
                            string = f"Patient {pat} Metastasis at location {metastasis_object.t1_path.parent.parent.name} has no match in the ground truth\n"

                    print(string)
                    report.writelines(string)

                for label in np.unique(p.labels):
                    if label == 0:
                        continue
                    elif label not in list(report_dict['matched_mets'].keys()):
                        report_dict['unmatched_gt_mets'].append(label)

                print(report_dict)
                writer.writerow(report_dict)

                matches += len(report_dict['matched_mets'].keys())
                gt_unmatched += len(report_dict['unmatched_gt_mets'])
                parsed_unmatched += len(report_dict['unmatched_parsed_mets'])
                t0_zero_parsed += len(report_dict['t0_zero_mets'])
                ambiguous_matches += len(report_dict['ambiguoug_parsed_matches'].keys())
    
    print(f"Found {matches} matching metastases in gt and parsing")
    print(f"Found {gt_unmatched} metastases in the GT that are not found in parsing")
    print(f"Found {parsed_unmatched} metastases in parsing that are not found in GT")
    print(f"Found {t0_zero_parsed} parsed metastases that are empty at their first occurance")
    print(f"Found {ambiguous_matches} GT metastases that have ambiguoug matches in the parsing")

        

    met_sourcs = np.asarray(met_sourcs)

    print(f"found {met_sourcs.sum()} metastases in {len(pats)} patients in the pre reseg data")

    met_parsed = np.asarray(met_parsed)

    print(f"found {met_parsed.sum()} metastases in {len(pats)} patients in the reseg and parsed data")


    disparity = []
    with open('/home/lorenz/BMDataAnalysis/logs/502/met_disparity.txt', 'w') as file:
        for i, pat in enumerate(pats):
            disparity.append(met_sourcs[i]-met_parsed[i])
            print(f'{pat} has {met_sourcs[i]} in source and {met_parsed[i]} in parsed, disparity of {met_sourcs[i]-met_parsed[i]}')
            file.writelines(f'{pat} has {met_sourcs[i]} in source and {met_parsed[i]} in parsed, disparity of {met_sourcs[i]-met_parsed[i]}\n')
    
    disp = np.asarray(disparity)

    print(f"Reseg introduced {np.sum(disp[disp<0])} mets not in source")
    print(f"Reseg matched {np.sum(disp==0)} patients perfectly")
    print(f"Reseg missed {np.sum(disp[disp>0])} mets in source")
