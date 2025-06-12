import pathlib as pl
from datetime import datetime
import re
import os
import SimpleITK as sitk
import numpy as np
from .metastasis import Metastasis, EmptyMetastasis
from .metastasis_series import MetastasisTimeSeries, parse_to_timeseries, load_series
from scipy import ndimage
import pandas as pd
import copy
from PrettyPrint import *

def load_patient(path:pl.Path):
    """
    Utility function to load a patient
    """
    if any([f for f in os.listdir(path) if f.startswith('Metastasis')]):
        return Patient(path, True)
    else:
        print('== Found no Metastases in patient directory!')
        return False

class Patient():
    """
    Represents a patient in the dataset
    """
    def __init__(self, path: pl.Path, load_mets = False, log=Printer(), met_dir_name:str='mets'):
        """
        BrainMetsDataAnalysis = BMDA
        path = pl.Path object, the path do the patient source, can be the output of the pipeline or a stored patient in the BMDA format
        load_mets = loads mets from disk instead of parsing them, set to true if path is patient in BMDA format
        log = prettyprint.Printer object will be instantiated on setup, but can be an object from a higher hierarchy script
        met_dir_name = string, relevant when mets are parsed from Pipeline output, looks for this directory to find the binary mask of lesion resegmentations
        """
        self.path = path
        self.id = path.name
        self.log = log
        self.met_dir_name = met_dir_name
        if not load_mets:        
            self.dates_dict = self._sort_directories([d for d in os.listdir(self.path) if os.path.isdir(os.path.join(self.path, d)) and ((self.path/d/'anat').is_dir() or (self.path/d/'rt').is_dir())], r"^ses-(\d{14})$")
            self.dates = list(self.dates_dict.keys())
            self.studies = len(self.dates)
            self._set_metadata()
            self.mets = self._parse_mets()
        else:
            mets = [f for f in os.listdir(self.path) if (self.path/f).is_dir()]
            dirs = []
            for met in mets:
                files = [f for f in os.listdir(self.path/met) if (self.path/met/f).is_dir()]
                for file in files:
                    dummy = file.split('-')[-1]
                    dummy = 'tX -'+dummy
                    if dummy not in dirs:
                        dirs.append(dummy)
            self.dates_dict = self._sort_directories(dirs, r"^tX - (\d{14})$")
            self.dates = list(self.dates_dict.keys())
            self.studies = len(self.dates)
            self._set_metadata(load_mets)
            self.mets = self._load_mets()
        self._assign_lesion_load()

    def print(self):
        """
        Prints object metadata to the console
        """
        print('--------------- BM Analysis Patient object ---------------')
        print('-- Patient ID:', self.id)
        print('-- Source Data Path:', self.path)
        #print('-- Dates:', self.dates)
        if hasattr(self, 'studies'):
            print(r'-- #Studies:', self.studies)
        print('-- Patient Brain Volume [mm³]:', self.brain_volume)
        if hasattr(self, 'start_date'):
            print('-- Start Date:', self.start_date)
        if hasattr(self, 'end_date'):
            print('-- End Date:', self.end_date)
        if hasattr(self, 'observation_days'):
            print('-- Observed Days:', self.observation_days)
        if hasattr(self, 'avg_study_interval_days'):
            print('-- Average Days in between Studies:', self.avg_study_interval_days)
        print(r'-- #Metastases:', len(self.mets))
        for i, met in self.mets.items():
            print(f'-- Metastasis time series {i}:')
            met.print()
        print('----------------------------------------------------------')

    def save(self, path:pl.Path):
        """
        Saves the patients seperated and matched metastases to a new directory
        """
        os.mkdir(path/self.id)
        brain = [(self.path/self.dates[0]/'anat'/elem) for elem in os.listdir(self.path/self.dates[0]/'anat') if elem.startswith('MASK_')][0]
        (path/self.id/'whole_brain.nii.gz').symlink_to(self.path/self.dates[0]/'anat'/brain)
        for i, series in self.mets.items():
            series.save(path/self.id)

    def discard_swings(self, relevant_period=420, verbose=False):
        self.mets = {i:met for i, met in self.mets.items() if not met.check_has_CR_swings(relevant_period, verbose)}
        if verbose: [print('I still have', met.id) for k, met in self.mets.items()]

    def resample_all_timeseries(self, timeframe_days:int=360, timepoints: int=6, method:str='linear'):
        """
        applies the resampling to all metastases in the patient
        """
        self.mets = {i:met.resample(timeframe_days, timepoints, method) for i, met in self.mets.items()}

    def drop_short_timeseries(self, min_length_days=360):
        """
        applies the resampling to all metastases in the patient
        """
        self.mets = {i:met for i, met in self.mets.items() if met.get_observation_days()>=min_length_days}

    def discard_unmatched(self, matched:list):
        """
        In case the matching to RT is used, drop metastases that are not matched
        """
        initial = len(self.mets)
        self.mets = {i:met for i,met in self.mets.items() if met.id in matched}
        print(f"discarded {initial-len(self.mets)} metastases series, kept metastses: {[met for met in list(self.mets.keys())]}")
    
    def discard_gaps(self, max_gap=120, max_period=360, verbose=False):
        if verbose: [print(f"Dropping {met.id} for having a gap of more than {max_gap} days in the period of {max_period}") for i, met in self.mets.items() if not met.check_interval(max_gap, max_period, verbose)]
        self.mets = {i:met for i,met in self.mets.items() if met.check_has_no_large_intervals(max_gap, max_period, False)}
    
    def tag_unmatched(self, matched:list):
        """
        In case the matching to RT is used, tag metastases that are not matched
        This is later a feature in the data extraction
        """
        print('tagging matched metastases', matched)
        for i in list(self.mets.keys()):
            if self.mets[i] is not None: self.mets[i].set_match(self.mets[i].id in matched)
    
    def validate(self, raise_on_invalid=True):
        """
        sanity check to se if the time order is correct and if all keys correspond to each other in the underlying datastructures
        """
        for k, met in self.mets.items():
            #print('validating', k)
            met.validate(raise_on_invalid)

####### GETTERS    
    def get_features(self, features='all', get_keys=True, deep_extractor=None):
        """
        Returns a list of dictionaries to be written with the csv library for example
        each dict is a time series

        :param features: can either be a string or a list of strings containing the features to extract for each lesion, optional, default 'all'
        """
        dict_list = []
        keys = None
        for i, ts in self.mets.items():
                
            #try:
                if ts is not None:
                    print(f'== working on {ts.id}')
                    
                    cur_dict = {}
                    id = f"{self.id}:{ts.id.split(' ')[-1]}"
                    cur_dict['Lesion ID'] = id
                    if hasattr(ts, 'is_match'): 
                        cur_dict['RT_matched'] = ts.is_match

                    ts_data = ts.get_time_delta()
                    prefixes = list(ts_data.keys())
                    cur_dict = {**cur_dict, **ts_data}

                    if isinstance(features, str):
                        features = [features]

                    for feature in features:
                        if feature in ['all', 'volume']:
                            cur_dict = {**cur_dict, **ts.get_volume()}
                        
                        if feature in ['all', 'rano']:
                            cur_dict = {**cur_dict, **ts.get_rano('3d')}

                        if feature in ['all', 'radiomics']:
                            cur_dict = {**cur_dict, **ts.get_radiomics(), **ts.get_border_radiomics()}

                        if feature in ['all', 'patient_meta']: 
                            cur_dict = {**cur_dict, 'Brain Volume': self.brain_volume}

                        if feature in ['all', 'total_load']: 
                            cur_dict = {**cur_dict, **ts.get_total_load()}
                            
                        if feature in ['all', 'lesion_meta']: # location in brain, primary, etc
                            cur_dict = {**cur_dict, **ts.get_location_in_brain()}

                        if feature in ['all', 'deep'] and deep_extractor is not None: # encoded vector from vincents foundation model
                            cur_dict = {**cur_dict, **ts.get_deep_vectors(deep_extractor)}
                    
                    dict_list.append(cur_dict)

                    if keys is None:
                        keys = list(cur_dict.keys())
            # except Exception as e:
            #     print(f"==== Exception occured while processing: {e}")
            #     print(f"==== Skipping extraction for {i}")

        if get_keys: return dict_list, keys
        else: return dict_list

####### Private internal utils
    def _sort_directories(self, dirs, pattern): # courtesy of chatgpt
        """
        Finds directories in the specified base_dir that match the pattern
        ses-yyyymmddhhmmss, parses the timestamp, and returns a list of directory
        names sorted in chronological order.
        """
        # List all directories in the base directory
        matching_dirs = []
        for d in dirs:
            match = re.match(pattern, d)
            if match:
                timestamp_str = match.group(1)
                # Convert the timestamp string to a datetime object
                dt = datetime.strptime(timestamp_str, "%Y%m%d%H%M%S")
                matching_dirs.append((d, dt))
        # Sort the directories based on the datetime objects
        matching_dirs.sort(key=lambda x: x[1])
        # Return just the sorted directory names
        res = {}
        for d in matching_dirs:
            res[d[0]] = d[1]
        return res

    def _set_metadata(self, load_saved = False):
        """
        writes some useful metadata to the attribute dict
        """
        if load_saved:
            init_brain = sitk.ReadImage(self.path/'whole_brain.nii.gz')
        else:
            init_brain = [sitk.ReadImage(self.path/self.dates[0]/'anat'/elem) for elem in os.listdir(self.path/self.dates[0]/'anat') if elem.startswith('MASK_')][0]

        self.start_date = self.dates_dict[self.dates[0]]
        self.end_date = self.dates_dict[self.dates[-1]]
        self.observation_days = (self.end_date-self.start_date).days
        self.avg_study_interval_days = self.observation_days/self.studies

        spacing = init_brain.GetSpacing()
        voxel_volume = spacing[0]*spacing[1]*spacing[2]
        init_brain = sitk.GetArrayFromImage(init_brain)
        self.brain_volume = np.sum(init_brain) * voxel_volume
        

    def _load_mets(self):
        """
        calls the load metastases fucntion to load mets from existing data instead of manually parsing them from the raw set again
        """
        mets = [file for file in os.listdir(self.path) if file.startswith('Metastasis')]
        return {int(file.split(' ')[-1]):load_series(self.path/file) for file in mets}

    def _parse_mets(self):
        """
        Reads time series, seperates metastasis masks into unique entities and stores them in a dictionary with the same keys as studies, but each value is a list of metastais objects
        """
        struct_el = ndimage.generate_binary_structure(rank=3, connectivity=3)
        mets_dict = {}
        treatments_dict = {}
        for date in self.dates:
            mets = []
            mask = sitk.ReadImage(self.path/date/self.met_dir_name/'metastasis_labels_1_class.nii.gz')

            t1 = [file for file in os.listdir(self.path/date/'anat') if file.endswith('T1w.nii.gz') and not file.startswith('MASK_')][0]
            t1 = self.path/date/'anat'/t1
    
            t2 = [file for file in os.listdir(self.path/date/'anat') if file.endswith('T2w.nii.gz')]
            if t2:
                t2 = self.path/date/'anat'/t2[0]
            else: t2 = None

            mask_arr = sitk.GetArrayFromImage(mask)
            mask_arr = ndimage.binary_closing(mask_arr, struct_el) # fill small holes in the foreground
            label_arr, n_labels = ndimage.label(mask_arr, structure=struct_el)

            if n_labels != 0: # check whether there even are labels at the timepoint
                for label in range(1, n_labels+1):
                    # generate ROI mask
                    cur_arr = np.zeros_like(label_arr)
                    cur_arr[label_arr==label]=1
                    met = sitk.GetImageFromArray(cur_arr.astype(np.uint8))
                    met.CopyInformation(mask)

                    mets.append(
                        Metastasis(
                            met, 
                            binary_source=None, # here its redundant, because we already pass an sitk image as binary source
                            multiclass_source= self.path/date/self.met_dir_name/'metastasis_labels_3_class.nii.gz' if (self.path/date/self.met_dir_name/'metastasis_labels_3_class.nii.gz').is_file() else None, 
                            t1_path=t1, 
                            t2_path=t2
                            )
                        )
                mets_dict[date] = mets
            else: # append an empty metastasis if not
                mets_dict[date] = [EmptyMetastasis(t1, t2)]

            if (self.path/date/'rt').is_dir():
                treatments_dict[date]=True
            else:
                treatments_dict[date]=False
        ts = parse_to_timeseries(mets_dict, self.dates_dict, treatments_dict, self.log)
        ts = {i:v for i,v in enumerate(ts)}
        return ts
    
    def _assign_lesion_load(self):
        for tp in self.dates:
            proc_tp = 'ses-'+tp[5:]
            existing = [met.time_series[proc_tp].lesion_volume for k, met in self.mets.items() if proc_tp in met.keys and met.time_series[proc_tp].lesion_volume>0] # extract the lesion volume at timepoint if the lesion has the timepoint
            count = len(existing)
            load = sum(existing)
            #print(f"found #{count} with combine volume {load}")
            #print(existing)
            [met.set_total_lesion_load_at_tp(count, load, proc_tp) for k, met in self.mets.items() if proc_tp in met.keys]



class PatientMetCounter(Patient):
    """
    This is a hacky way to extract information from the dataset
    """
    def __init__(self, path, mapping):
        self.path = path
        self.mapping = pd.read_csv(mapping)
        #print(self.mapping)
        self.id = path.name
        self.dates_dict = self._sort_directories([d for d in os.listdir(self.path) if os.path.isdir(os.path.join(self.path, d)) and ((self.path/d/'anat').is_dir() or (self.path/d/'rt').is_dir())], r"^ses-(\d{14})$")
        self.dates = list(self.dates_dict.keys())
        self.studies = len(self.dates)
        self.mets, self.mask = self._count_mets()

    def _count_mets(self):
        print("WARNING this counter uses morph ops on the rt masks: closing->opening->labeling this is because the objects in the masks are very rough and noisy, which will artificially inflate the number of objects found!")
        struct_el = ndimage.generate_binary_structure(rank=3, connectivity=3)
        mets = 0
        mask = None
        rts = [d for d in self.dates if (self.path/d/'rt').is_dir()]

        mapped_path = self.mapping.loc[self.mapping['source_study_path'] == str(self.path/self.dates[-1])] # dates are in chronological order so the last entry in the series should have all metastases found in rt
        #print('mapped_path', mapped_path)
        if not mapped_path.empty:
            path = str(mapped_path.iloc[0]['nnUNet_set_dir'])
            name = str(mapped_path.iloc[0]['nnUNet_UID'])
            name += '0001.nii.gz' # mask
            rt_gt_path = pl.Path(path)
            mask = sitk.ReadImage(rt_gt_path/name)
            mask_arr = sitk.GetArrayFromImage(mask)
            mask_arr = ndimage.binary_closing(mask_arr, struct_el) # fill small holes
            mask_arr = ndimage.binary_opening(mask_arr, struct_el) # remove small objects
            label_arr, n_labels = ndimage.label(mask_arr, structure=struct_el)
            mets+=n_labels

            self.mask = mask
            self.mask_arr = mask_arr
            self.labels = label_arr
        return mets, mask
