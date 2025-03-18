import SimpleITK as sitk
import numpy as np
from .metastasis import Metastasis, generate_empty_met_from_met
from datetime import datetime, timedelta
from scipy.ndimage import center_of_mass
import copy
from PrettyPrint import *

def parse_to_timeseries(mets_dict: dict, dates_dict: dict, log: Printer=Printer()):
    inf = PPFormat([ColourText('blue'), Effect('bold'), Effect('underlined')]) 
    dates = mets_dict.keys()
    if dates != dates_dict.keys():
        raise RuntimeError("Dates in metastasis dict do not correspond with dates in date dict")
    dates = list(dates)
    time_series_list = [MetastasisTimeSeries(elem, dates_dict[dates[0]], dates[0]) for elem in mets_dict[dates[0]]]
    log.tagged_print('INFO', f'Parsed study {dates[0]}, found {len(mets_dict[dates[0]])} metastases at t0', inf)
    empty_met_image = generate_empty_met_from_met(mets_dict[dates[0]][0]) # an empty image that fits the met requirements, used to fill in timepoints in series when the metastasis is already gone
    for d in dates:
        new = 0
        existing = 0
        if d == dates[0]: # skip date zero cause that was used to initialize the mets list
            continue
        if mets_dict[d] is None: # skip empty time points
            continue
        else:
            extended_at_timepoint = [] # stores what series have a timepoint added, used to add an empty met in other series, that dont get expanded at this point
            new_at_timepoint = []
            for met in mets_dict[d]:
                metrics = [] # stores the correspondence metrics for the lesion to each series
                for i, t in enumerate(time_series_list): # check for correspondence and append if True
                    metrics.append(t.correspondence_metrics(met, dates_dict[d])) # appends the tuple of metrics to the list
                best = find_best_series_index(metrics, 'both')
                if best is not None: # append to timeseries
                    time_series_list[best].append(copy.deepcopy(met), dates_dict[d], d)
                    existing += 1
                    extended_at_timepoint.append(best)
                else: # make new timeseries and add to a new list, to avoid confusing the rest of the code
                    new_at_timepoint.append(MetastasisTimeSeries(copy.deepcopy(met), dates_dict[d], d))
                    new += 1
            # add empty met objects if no new one was added at the timepoint        
            for i, t in enumerate(time_series_list):
                if not i in extended_at_timepoint:
                    t.append(empty_met_image, dates_dict[d], d)
            # transcribe new additions from new list to main list
            for t in new_at_timepoint:
                time_series_list.append(t)


        log.tagged_print('INFO', f'Parsed study {d}, found {len(mets_dict[d])} metastases: {new} new mets and attached {existing} to existing metastasis series', inf)

    return time_series_list

def find_best_series_index(metrics, method: str='both'):
    """
    Identifies the best matchin lesion series by finding the maximum overlap, minimum centroid distance
    mode 'overlap': uses maximum overlap
    mode 'centroid': uses minimum centroid distance, if less than lesion diameter
    mode 'both': finds the lesion with max overlap and min distance given diameter threshold
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
            if m[1] < best_dist and m[1]<=m[2]: # look for minimum distance and distance less then ref diameter
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
                if m[1] < best_dist and m[1]<=m[2]:
                    best_series = i
                    best_dist = m[1]

        # if overlap fails use distance instead
        if best_series is None:
            for i, m in enumerate(metrics):
                if m[1] < best_dist and m[1]<=m[2]: # look for minimum distance and distance less then ref diameter
                    best_series = i

        return best_series
    


class MetastasisTimeSeries():
    def __init__(self, t0_metastasis: Metastasis, t0_date: datetime, t0_date_str: str):
        self.time_series = {}
        self.time_series[t0_date_str] = t0_metastasis
        self.dates = {}
        self.dates[t0_date_str] = t0_date	
        self.keys = [t0_date_str]

######## Public utils
    def get_observation_days(self):
        """
        computes and returns the time in days from t0 to tn-1
        """
        return (self.dates[self.keys[-1]] - self.dates[self.keys[0]]).days
    
    def resample(self, timeframe_days:int=360, timepoints: int=6, method:str='nearest'):
        """
        Resamples the timeseries to a given timeframe with given timepoints
        for example the default config 360/6 gives a timeseries over a duration of 360 days with 6 timepoints with 60 days in between each
        usefult to reorganize the data for data preparation
        """
        if self.get_observation_days() < timeframe_days - (timeframe_days/timepoints)*0.5: # the series needs to have at least timeframe-0.5deltaT days to be interpolated, deltaT is the period of time in between studies
            print(f'This series is missing too much data, cannot interpolate {timepoints} timepoints for timeframe {timeframe_days} with an observation period of {self.get_observation_days()}')
            return None
        if method == 'nearest': # basic nearest neighbor interpolation
            t0 = self[0]
            tps = self._generate_timepoints(t0[1], timeframe_days, timepoints)
            new_series = MetastasisTimeSeries(*t0)
            for i, tp in enumerate(list(tps.keys())):
                if i == 0:
                    continue
                closest_met_key, _ = self._find_closest_entry(tps[tp])
                new_series.append(self.time_series[closest_met_key], tps[tp], tp)
            return new_series
        else:
            raise RuntimeError(f'Invalid interpolation method: {method}')

    def correspondence_metrics(self, met: Metastasis, date: datetime):
        """
        computes the measurements that are used to identify a lesion as correspondence or not
        Returns the overlap between the target and the closest lesion in time in the series
        Returns the centroid distance between the target and the closest lesion in time in the series
        Returns the diameter of a perfect sphere with the volume of the closest lesion
        These measurements can then be used to match the lesions later on.
        """
        ref_key, _ = self._find_closest_nonzero_entry(date)
        ref_met = self.time_series[ref_key]
        if not ref_met.same_space(met):
            return False
        overlap = np.sum(np.bitwise_and(met.image, ref_met.image))
        target_centroid = center_of_mass(ref_met.image)
        candidate_centroid = center_of_mass(met.image)
        centroid_distance = np.sqrt(np.sum((np.asarray(target_centroid)-np.asarray(candidate_centroid))**2))
        candidate_radius = (ref_met.lesion_volume_voxel / ((4/3) * np.pi)) ** (1/3)
        return overlap, centroid_distance, candidate_radius

    def append(self, metastasis: Metastasis, date: datetime, date_str:str):
        """
        Adds a new Metastasis timepoint to the time series
        """
        self.time_series[date_str]=metastasis
        self.dates[date_str]=date
        self.keys.append(date_str)

    def print(self):
        """
        Reports the lesion volume and date to the console
        """
        for i, k in enumerate(self.keys):
            if self.time_series[k] is not None:
                print(f"    t{i}: {str(self.time_series[k])}, date = {self.dates[k]}")
    
    def __getitem__(self, idx):
        return self.time_series[self.keys[idx]], self.dates[self.keys[idx]], self.keys[idx]

    def __len__(self):
        return len(self.keys)

######## Private Utils
    def _find_closest_nonzero_entry(self, date: datetime):
        """
        Iterates over the time series and returns the key for the closest date as well as the time delta in days to a given reference date
        """
        ref_key = self.keys[0]
        ref_delta_t = abs((self.dates[self.keys[0]]-date).days)
        for k in self.keys:
            if self.time_series[k].lesion_volume != 0: # ignore empty images
                delta_t = abs((self.dates[k]-date).days)
                if delta_t < ref_delta_t:
                    ref_delta_t = delta_t
                    ref_key = k
        return ref_key, ref_delta_t
    
    def _find_closest_entry(self, date: datetime):
        """
        Iterates over the time series and returns the key for the closest date as well as the time delta in days to a given reference date
        """
        ref_key = self.keys[0]
        ref_delta_t = abs((self.dates[self.keys[0]]-date).days)
        for k in self.keys:
            delta_t = abs((self.dates[k]-date).days)
            if delta_t < ref_delta_t:
                ref_delta_t = delta_t
                ref_key = k
        return ref_key, ref_delta_t
    
    def _generate_timepoints(self, t0:datetime, timeframe_days:int, timepoints:int):
        """
        Utility function that computes the timestamps at regular intervals over a period of time in days
        returns a dictionary of date-strings as keys and datetimes as values
        used in resampling
        """
        tps = {}
        tps[datetime.strftime(t0, "%Y%m%d%H%M%S")] = t0
        delta_t = timedelta(days=timeframe_days/timepoints)
        for d in range(1, timepoints):
            tp = t0+d*delta_t
            tps[datetime.strftime(tp, "%Y%m%d%H%M%S")] = tp
        return tps

