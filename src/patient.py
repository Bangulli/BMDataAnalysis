import pathlib as pl
from datetime import datetime
import re
import os
import SimpleITK as sitk
import numpy as np
from .metastasis import Metastasis
from .metastasis_series import MetastasisTimeSeries, parse_to_timeseries
from scipy import ndimage

class Patient():
    def __init__(self, path: pl.Path):
        self.path = path
        self.id = path.name
        self.dates_dict = self._sort_directories()
        self.dates = list(self.dates_dict.keys())
        self.studies = len(self.dates)
        self._set_metadata()
        self.mets = self._parse_mets()
    
###### Private internal utils
    def _sort_directories(self): # courtesy of chatgpt
        """
        Finds directories in the specified base_dir that match the pattern
        ses-yyyymmddhhmmss, parses the timestamp, and returns a list of directory
        names sorted in chronological order.
        """
        base_dir = self.path
        # List all directories in the base directory
        dirs = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d)) and ((base_dir/d/'anat').is_dir() or (base_dir/d/'rt').is_dir())]
        pattern = r"^ses-(\d{14})$"
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

    def _set_metadata(self):
        init_brain = [sitk.ReadImage(self.path/self.dates[0]/'anat'/elem) for elem in os.listdir(self.path/self.dates[0]/'anat') if elem.startswith('MASK_')][0]
        spacing = init_brain.GetSpacing()
        voxel_volume = spacing[0]*spacing[1]*spacing[2]
        init_brain = sitk.GetArrayFromImage(init_brain)
        
        self.brain_volume = np.sum(init_brain) * voxel_volume
        self.start_date = self.dates_dict[self.dates[0]]
        self.end_date = self.dates_dict[self.dates[-1]]
        self.observation_days = (self.end_date-self.start_date).days
        self.avg_study_interval_days = self.observation_days/self.studies
        
    def _parse_mets(self):
        """
        Reads time series, seperates metastasis masks into unique entities and stores them in a dictionary with the same keys as studies, but each value is a list of metastais objects
        """
        struct_el = ndimage.generate_binary_structure(rank=3, connectivity=2)
        mets_dict = {}
        for date in self.dates:
            mets = []
            mask = sitk.ReadImage(self.path/date/'mets'/'metastasis_labels_1_class.nii.gz')
            mask_arr = sitk.GetArrayFromImage(mask)
            label_arr, n_labels = ndimage.label(mask_arr, structure=struct_el)
            if n_labels != 0:
                for label in range(1, n_labels+1):
                    met = sitk.GetImageFromArray((label_arr == label).astype(int))
                    met.CopyInformation(mask)
                    mets.append(Metastasis(met))
            mets_dict[date] = mets
        return parse_to_timeseries(mets_dict, self.dates_dict)

    def print(self):
        print('--------------- BM Analysis Patient object ---------------')
        print('-- Patient ID:', self.id)
        print('-- Source Data Path:', self.path)
        #print('-- Dates:', self.dates)
        print(r'-- #Studies:', self.studies)
        print('-- Patient Brain Volume [mm³]:', self.brain_volume)
        print('-- Start Date:', self.start_date)
        print('-- End Date:', self.end_date)
        print('-- Observed Days:', self.observation_days)
        print('-- Average Days in between Studies:', self.avg_study_interval_days)
        print(r'-- #Metastases:', len(self.mets))
        for i, met in enumerate(self.mets):
            print(f'-- Metastasis time series {i}:')
            met.print()
        print('----------------------------------------------------------')
