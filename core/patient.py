import pathlib as pl
from datetime import datetime
import re
import os
import SimpleITK as sitk
import numpy as np
from .metastasis import Metastasis
from .metastasis_series import MetastasisTimeSeries, parse_to_timeseries, load_series
from scipy import ndimage
import pandas as pd
import copy

def load_patient(path:pl.Path):
    """
    Utility function to load a patient
    """
    return Patient(path, True)

class Patient():
    """
    Represents a patient in the dataset
    """
    def __init__(self, path: pl.Path, load_mets = False):
        self.path = path
        self.id = path.name
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
        for i, met in enumerate(self.mets):
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
        for i, series in enumerate(self.mets):
            met_name = f"Metastasis {i}"
            os.mkdir(path/self.id/met_name)
            series.save(path/self.id/met_name)

    def to_numpy(self):
        """
        Returns a numpy array where the columns are the timepoints and the rows are the metastses
        """
        raise NotImplementedError('WIP')
        return
    
    def to_df(self):
        """
        Returns a pandas DataFrame where the columns are the timepoints and the rows are the metastases
        Column names are t0, t30, ... where the bumber is the difference in days from t0 at the current timepoint
        """
        raise NotImplementedError('WIP')
        return
    
    def to_csv(self, path: pl.Path):
        """
        Saves the metastasis dataframe from self.to_df as a csv file and stores a supplementary csv with patient metadata
        """
        raise NotImplementedError('WIP')
        return

###### Private internal utils
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
        return [load_series(self.path/file) for file in mets]

    def _parse_mets(self):
        """
        Reads time series, seperates metastasis masks into unique entities and stores them in a dictionary with the same keys as studies, but each value is a list of metastais objects
        """
        struct_el = ndimage.generate_binary_structure(rank=3, connectivity=2)
        mets_dict = {}
        treatments_dict = {}
        for date in self.dates:
            mets = []
            mask = sitk.ReadImage(self.path/date/'mets'/'metastasis_labels_1_class.nii.gz')
            t1 = [file for file in os.listdir(self.path/date/'anat') if file.endswith('T1w.nii.gz') and not file.startswith('MASK_')][0]
            t1 = self.path/date/'anat'/t1
    
            t2 = [file for file in os.listdir(self.path/date/'anat') if file.endswith('T2w.nii.gz')]
            if t2:
                t2 = self.path/date/'anat'/t2[0]
            else: t2 = None

            mask_arr = sitk.GetArrayFromImage(mask)
            label_arr, n_labels = ndimage.label(mask_arr, structure=struct_el)
            if n_labels != 0:
                for label in range(1, n_labels+1):
                    met = sitk.GetImageFromArray((label_arr == label).astype(int))
                    met.CopyInformation(mask)
                    mets.append(
                        Metastasis(
                            met, 
                            binary_source=None, # here its redundant, because we already pass an sitk image as binary source
                            multiclass_source=self.path/date/'mets'/'metastasis_labels_3_class.nii.gz', 
                            t1_path=t1, 
                            t2_path=t2
                            )
                        )
            mets_dict[date] = mets
            if (self.path/date/'rt').is_dir():
                treatments_dict[date]=True
            else:
                treatments_dict[date]=False
        return parse_to_timeseries(mets_dict, self.dates_dict, treatments_dict)



class PatientMetCounter(Patient):
    """
    This is a hacky way to extract information from the dataset
    """
    def __init__(self, path, mapping):
        self.path = path
        self.mapping = pd.read_csv(mapping)
        self.id = path.name
        self.dates_dict = self._sort_directories([d for d in os.listdir(self.path) if os.path.isdir(os.path.join(self.path, d)) and ((self.path/d/'anat').is_dir() or (self.path/d/'rt').is_dir())], r"^ses-(\d{14})$")
        self.dates = list(self.dates_dict.keys())
        self.studies = len(self.dates)
        self.mets = self._count_mets()

    def _count_mets(self):
        struct_el = ndimage.generate_binary_structure(rank=3, connectivity=2)
        mets = 0
        rts = [d for d in self.dates if (self.path/d/'rt').is_dir()]

        mapped_path = self.mapping.loc[self.mapping['source_study_path'] == str(self.path/rts[-1])] # dates are in chronological order so the last rt in the series should have all metastases found in rt
        if not mapped_path.empty:
            path = str(mapped_path.iloc[0]['nnUNet_set_dir'])
            name = str(mapped_path.iloc[0]['nnUNet_UID'])
            name += '0001.nii.gz' # mask
            rt_gt_path = pl.Path(path)
            mask = sitk.ReadImage(rt_gt_path/name)
            mask_arr = sitk.GetArrayFromImage(mask)
            label_arr, n_labels = ndimage.label(mask_arr, structure=struct_el)
            mets+=n_labels
        return mets
