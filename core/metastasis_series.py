import SimpleITK as sitk
import numpy as np
from .metastasis import Metastasis, generate_empty_met_from_met, generate_interpolated_met_from_met, InterpolatedMetastasis, load_metastasis
from datetime import datetime, timedelta
from scipy.ndimage import center_of_mass
import copy
import os
import pathlib as pl
from PrettyPrint import *
from scipy.interpolate import make_interp_spline, BSpline
import matplotlib.pyplot as plt

#### Global utility functions
def parse_to_timeseries(mets_dict: dict, dates_dict: dict, treatments_dict:dict, log: Printer=Printer()):
    """
    parses a dictionary of metastases and a dictionary of dates into N MetastasisTimeSeries objects
    Matches lesions to each other and expands existing time series or spawns new ones depending on correspondence
    mets_dict must be a dictionary of date strings as keys and metastasis objects as values, each metastasis object represents a single instance!! need to be seperated sucht that there is only one cluster in the mask
    dates_dict must have the same keys as mets dict and the values are just datetime objects 
    log is an instance of a PrettyPrint printer, for logging purposes
    """
    # init vars
    inf = PPFormat([ColourText('blue'), Effect('bold'), Effect('underlined')]) 
    dates = mets_dict.keys()
    # runtime checks
    if dates != dates_dict.keys():
        raise RuntimeError("Dates in metastasis dict do not correspond with dates in date dict")
    # init more vars
    dates = list(dates)
    time_series_list = [MetastasisTimeSeries(elem, dates_dict[dates[0]], dates[0]) for elem in mets_dict[dates[0]]]
    log.tagged_print('INFO', f'Parsed study {dates[0]}, found {len(mets_dict[dates[0]])} metastases at t0', inf)
    empty_met_image = generate_empty_met_from_met(mets_dict[dates[0]][0]) # an empty image that fits the met requirements, used to fill in timepoints in series when the metastasis is already gone
    # iterate over metastis dict to parse
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
                if treatments_dict[d]: # only append new time series on treatment days
                    time_series_list.append(t)


        log.tagged_print('INFO', f'Parsed study {d}, found {len(mets_dict[d])} metastases: {new} new mets and attached {existing} to existing metastasis series', inf)

    return time_series_list

def find_best_series_index(metrics, method: str='both'):
    """
    Identifies the best matchin lesion series by finding the maximum overlap, minimum centroid distance
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

def load_series(path:pl.Path):
    """
    Loads a series of metastases from disk
    """
    files = sorted(os.listdir(path), key=lambda x: int(x.split(' - ')[0][1:])) # generates a sorted list from t0 to tN
    t0 = load_metastasis(path/files[0])
    t0_date = datetime.strptime(files[0].split(' - ')[-1], "%Y%m%d%H%M%S")
    t0_date_str = 'ses-'+files[0].split(' - ')[-1]
    new_series = MetastasisTimeSeries(t0, t0_date, t0_date_str)
    for i, elem in enumerate(files):
        if i==0:
            continue
        tp = load_metastasis(path/elem)
        tp_date = datetime.strptime(elem.split(' - ')[-1], "%Y%m%d%H%M%S")
        tp_date_str = 'ses-'+elem.split(' - ')[-1]
        new_series.append(tp, tp_date, tp_date_str)
    return new_series

class MetastasisTimeSeries():
    """
    Represents a series of metastases over time in chronological order
    """
    def __init__(self, t0_metastasis: Metastasis, t0_date: datetime, t0_date_str: str):
        self.time_series = {}
        self.time_series[t0_date_str] = t0_metastasis
        self.dates = {}
        self.dates[t0_date_str] = t0_date	
        self.keys = [t0_date_str]

######## Public utils
    def save(self, path:pl.Path, use_symlinks=True):
        """
        Saves the metastasis series to a target directory
        """
        assert path.is_dir(), "Target path needs to be a directory"
        # if the series is all interpolated save the values as a csv
        if all([isinstance(elem, InterpolatedMetastasis) for elem in list(self.time_series.values())]):
            raise NotImplementedError('Put saving as csv for interpolated series here')
        # if the series is all natural mets, save them as images and stuff
        elif all([isinstance(elem, Metastasis) for elem in list(self.time_series.values())]):
            for i, k in enumerate(self.keys):
                date_str = datetime.strftime(self.dates[k], "%Y%m%d%H%M%S")
                tp_name = f"t{i} - {date_str}"
                os.mkdir(path/tp_name)
                self.time_series[k].save(path/tp_name, use_symlinks)
        # error case
        else:
            raise RuntimeError('Cannot save TimeSeriesObject, did not receive a homogeneous value dict, needs to be either all Metastasis or all InterpolatedMetastasis')

    def get_trajectory_bspline(self, degree=3):
        """
        Generates an interpolated bspline for the volume across time, taking the entire observation period into account
        """
        x = []
        y = []
        _, ref_t, _ = self[0] # just gets the date at t0
        for i in range(len(self)): # iterate over self to get all dts and mets
            met, t, _ = self[i]
            x.append((t-ref_t).days)
            y.append(met.lesion_volume)
        return make_interp_spline(x, y, k=degree) # call scipy make interp spline to return bspline

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
        supported interpolation methods: 
            - nearest = nearest neighbor
            - linear = linear interpolation between the most previous and least after timepoint
            - bspline = Represents all timepoints in a bspline interpolation and draws data from there
        """
        # check for data availability
        if self.get_observation_days() < timeframe_days - (timeframe_days/timepoints)*0.5: # the series needs to have at least timeframe-0.5deltaT days to be interpolated, deltaT is the period of time in between studies
            print(f'This series is missing too much data, cannot interpolate {timepoints} timepoints for timeframe {timeframe_days} with an observation period of {self.get_observation_days()}')
            return None
        
        ## interpolation cases
        ## basic nearest neighbor interpolation
        if method == 'nearest': 
            t0 = self[0]
            tps = self._generate_timepoints(t0[1], timeframe_days, timepoints)
            new_series = MetastasisTimeSeries(generate_interpolated_met_from_met(t0[0]), t0[1], t0[2])
            for i, tp in enumerate(list(tps.keys())):
                if i == 0:
                    continue
                closest_met_key, _ = self._find_closest_entry(tps[tp])
                new_series.append(generate_interpolated_met_from_met(self.time_series[closest_met_key]), tps[tp], tp)
            return new_series
        
        ## linear interpolation
        elif method == 'linear':
            t0 = self[0]
            tps = self._generate_timepoints(t0[1], timeframe_days, timepoints)
            new_series = MetastasisTimeSeries(generate_interpolated_met_from_met(t0[0]), t0[1], t0[2])
            for i, tp in enumerate(list(tps.keys())):
                if i == 0:
                    continue
                keys_dts = self._find_closest_bilateral(tps[tp])
                interpolated_volume = self._liner_interpolation(*keys_dts)
                new_series.append(InterpolatedMetastasis(interpolated_volume), tps[tp], tp)
            return new_series
        
        ## BSpline interpolation
        elif method == 'bspline':
            t0 = self[0]
            tps = self._generate_timepoints(t0[1], timeframe_days, timepoints)
            new_series = MetastasisTimeSeries(generate_interpolated_met_from_met(t0[0]), t0[1], t0[2])
            bspline = self.get_trajectory_bspline()
            for i, tp in enumerate(list(tps.keys())):
                if i == 0:
                    k0 = tp
                    continue
                delta_t = (tps[tp]-tps[k0]).days
                assert delta_t>0 # make sure that the delta between t0 and current to is not negative
                interpolated_volume = bspline(delta_t)
                interpolated_volume = interpolated_volume if interpolated_volume > 0 else 0
                new_series.append(InterpolatedMetastasis(interpolated_volume), tps[tp], tp)
            return new_series
        
        ## invalid method
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
        candidate_radius = (ref_met.lesion_size_voxel / ((4/3) * np.pi)) ** (1/3)
        return overlap, centroid_distance, candidate_radius*2

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

######## Builtins  
    def __getitem__(self, idx):
        return self.time_series[self.keys[idx]], self.dates[self.keys[idx]], self.keys[idx]

    def __len__(self):
        return len(self.keys)
    
    def __list__(self):
        """
        wraps the python list cast
        Returns a list of lesion volumes
        """
        return [self.time_series[met].lesion_volume for met in self.keys]
    
    def __dict__(self):
        """
        wraps the python dict cast
        Returns a dictionary of delta_t as keys and lesion volume as values
        """
        value_dict = {}
        value_dict[0] = self.time_series[0].lesion_volume
        for i, d in enumerate(self.keys):
            if i == 0:
                continue
            value_dict[int((self.dates[d]-self.dates[0]).days)] = self.time_series[d].lesion_volume
        return value_dict

######## Private Utils
    def _plot_trajectory(self, path):
        """
        Creates a line plot for the metastasis volume over time with lines connecting the points
        """
        x = []
        y = []
        _, ref_t, _ = self[0] # just gets the date at t0
        print(self[0])
        for i in range(len(self)): # iterate over self to get all dts and mets
            met, t, _ = self[i]
            x.append((t-ref_t).days)
            y.append(met.lesion_volume)

        # Plot the result
        plt.plot(x, y, 'o', label="Data Points", linestyle='-')
        plt.xlabel('Delta T [days]')
        plt.ylabel('Volume [mm³]')
        plt.legend()
        plt.savefig(path)
        plt.clf()

    def _plot_trajectory_comparison(self, path, ref):
        """
        creates a plot comparing the current metastasis trajectory to a reference metastasis trajectory
        """
        ### plot self
        x = []
        y = []
        _, ref_t, _ = self[0] # just gets the date at t0
        for i in range(len(self)): # iterate over self to get all dts and mets
            met, t, _ = self[i]
            x.append((t-ref_t).days)
            y.append(met.lesion_volume)
        # Plot the result
        plt.plot(x, y, 'o', label="Data Points Resampled", linestyle='-')

        ### plot ref
        x = []
        y = []
        _, ref_t, _ = ref[0] # just gets the date at t0
        for i in range(len(ref)): # iterate over self to get all dts and mets
            met, t, _ = ref[i]
            x.append((t-ref_t).days)
            y.append(met.lesion_volume)
        # Plot the result
        plt.plot(x, y, 'o', label="Data Points Original", linestyle='-')
        plt.legend()
        plt.xlabel('Delta T [days]')
        plt.ylabel('Volume [mm³]')
        plt.savefig(path)
        plt.clf()


    def _liner_interpolation(self, met1, dt1, met2, dt2):
        """
        Interpolates a volume by approximating a line between two time points
        Uses 2D space, takes delta_t as x and volume as y
        """
        met1 = self.time_series[met1]
        met2 = self.time_series[met2]
        v1 = met1.lesion_volume
        v2 = met2.lesion_volume
        # just a simple line formula where a point in 2D space is defined as p(x,y) where x = delta_t and y = volume
        a = (v2-v1)/(dt2-dt1) # slope
        b = v1-a*dt1 # intercept
        # here we could calc any point in time from these two points, but since dt is already the dimediff from our target the intercept is our desired value
        return b

    def _find_closest_nonzero_entry(self, date: datetime):
        """
        Iterates over the time series and returns the key for the closest date as well as the time delta in days to a given reference date ignoring dates with no data
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
    
    def _find_closest_bilateral(self, date:datetime):
        """
        Iterate over the time series and find the closest date before AND after the reference date
        """
        # init references
        ref_key_before = self.keys[0]
        ref_key_after = self.keys[-1]
        ref_delta_t_before = (self.dates[self.keys[0]]-date).days
        ref_delta_t_after = (self.dates[self.keys[-1]]-date).days
        # assertain valid values
        assert ref_delta_t_before < 0, "The reference date should be chornologically after the first timepoint"  # make sure that the date is negative, so the ref date is bigger than the first date, if not given it cant find the closest bilateral
        assert ref_delta_t_after > 0, "The reference date should be chornologically before the last timepoint"  # make sure that the date is positive, so the ref date is smaller than the last date, if not given it cant find the closest bilateral
        # iterate over entries
        for k in self.keys:
            delta_t = (self.dates[k]-date).days
            if delta_t > ref_delta_t_before and delta_t < 0: # update before
                ref_delta_t_before = delta_t
                ref_key_before = k
            if delta_t < ref_delta_t_after and delta_t >= 0: # update after
                ref_key_after = k
                ref_delta_t_after = delta_t
        return ref_key_before, ref_delta_t_before, ref_key_after, ref_delta_t_after
    
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

