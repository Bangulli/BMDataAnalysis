# Brain Metastases Data Analytics
This is the repository for our paper: titel to be decided

We track and match lesion segmentations in longitudinal MRI (assumes preprocessing with [BMPipeline](https://github.com/Bangulli/BMPipeline)) and store them as individual lesion time series per patient.
This repository provides multiple approaches for longitudinal time series resampling and feature extraction.

## Contents
- [Brain Metastases Data Analytics](#brain-metastases-data-analytics)
  - [Contents](#contents)
  - [Usage](#usage)
  - [Approach](#approach)
  - [MICCAI LMID](#miccai-lmid)
  - [Datastructure](#datastructure)
  - [Packages](#packages)
    - [Core](#core)
      - [Patient](#patient)
      - [Metastasis time series](#metastasis-time-series)
      - [Metastasis](#metastasis)
    - [Clustering](#clustering)
    - [Data](#data)
    - [Prediction](#prediction)
    - [visualization](#visualization)
    - [Deep Features](#deep-features)
    - [misc](#misc)

## Usage
The environment is managed by anaconda (Python version 3.10)
Anaconda [requirements](requirements.txt)
Some packages are not available on conda, so it is supplemented with pip [requirements](requirements_pip.txt)

- [parse_mets](parse_mets.py) is an example on how to use the data structure to load and parse the results from the [Pipeline](https://github.com/Bangulli/BMPipeline) to a lesion level dataset
- [feature_extraction](feature_extraction.py) is an example on how to extract features from a lesion level dataset.

## Approach
We extract features from the lesion time series and resample it into a series with homogeneous intervals in between datapoints. Volume resampling is done in either Nearest neighbor, Linear or BSpine interpolation. The time series itself (dataobjects with images) can only be resampled with nearest neighbor to keep the link between lesion volume and image.
Features are stored in csv files with a lesion ID, patient level clinical features and timepoints. Time point columns are denoted by tX as a prefix, so column.split('_') gives the timepoint prefix.
The column tX_timedelta_days stores the days elapsed from t0.

Missing values are imputed by time weighted linear interpolation for volume and radiomic features.
Prediction can be performed with classical methods (by concatenating timepoint feature vectors) or graph methods (by modeling the timeseries itself as a graph)
Clustering is done with stepmix, tsxmeans or tsxshapes. Attempts were made with rpy2 interaction to run more powerful clustering algorithms in R

## MICCAI LMID
The code for experiments and evaluations run for the submission to the MICCAI workshop on Learning with Longitudinal Medical Images and Data are located in this [folder](/MICCAI_submission/).
- [_classical_ml_5fold](/MICCAI_submission/_classical_ml_5fold.py)
  - The script used to train the LGBM model for both prediction tasks with Radiomic, Volume and Clinical features
- [_gml_general_5fold](/MICCAI_submission/_gml_general_5fold.py)
  - The script used to train a general GML model for both prediction tasks with Radiomic, Volume and Clinical features
- [_gml_time_specific_5fold](/MICCAI_submission/_gml_time_specific_5fold.py)
  - The script used to train a time specific GML model for both prediction tasks with Radiomic, Volume and Clinical features
- [bootstrap_auc](/MICCAI_submission/bootstrap_auc.py)
  - The script used to compute the AUC confidence interval with bootstrapping
- [clustering_stepmix](/MICCAI_submission/clustering_stepmix.py)
  - The script used to perform clustering on the time series data with the StepMix package
- [permutation_test](/MICCAI_submission/permutation_test.py)
  - The script used to perform the permutation test within the same model across timepoints and within the same timepoint across models

## Datastructure
A Brain Mets Data Analysis (BMDA) is structured like this
- Parent
  - Patient 1
    - whole_brain.nii.gz
    - Metastasis 0
      - t0-yyyymmddhhmmss
        - t1.nii.gz
        - t2.nii.gz
        - metastasis_mask_binary.nii.gz
      - t1-yyyymmddhhmmss
      - ...
      - tN-yyyymmddhhmmss
    - Metastasis 1
      - ...
  - Patient 2
    - ...
  - ...

Patient folders have a brain mask and metastasis subfolders
Metastasis foders have timepoint subfolders
timepoint subfolders have 
- A T1 image
- A T2 image if available
- A lesion mask if available
If no lesion mask is provided, that means that the nnUNet did not produce an output for this timepoint, i.e. the lesion disappeared or couldnt be segmented (non-gado image)

## Packages
### [Core](core)
Contains data objects to handle parsing, saving, loading, filtering, resampling and feature extraction from lesion time series.
Ideally you only interact with the patient object, which handles time series and indivdual timepoints internally.
If you need to load individual time series, use the MetastasisTimeSeries object
If you need to load individual time points, use the Metastsis object. Or build your on proprietary function that handles the [data structure](#datastructure)

These are the most important objects in this repo, they handle loading and feature generateion, the rest is just regular PyTorch and SKlearn using Pandas.DataFrames
#### Patient
- Patient (object)
  - Models a patient, has multiple (MetastasisTimeSeries) in a list. Can parse mets from the Pipeline output or load them from Brain Mets Data Analysis (BMDA) custom lesion data format
  - Arguments
    - path = pathlib.Path, path to a patient directory in BMPipeline (BMP) output or BMDA format
    - load_mets = bool, default=False, if True trys to load a BMDA format patient file, if False trys to parse mets from BMP output format
    - log = PrettyPrint.Printer object, optional, if passed will use the passed printer and its logfile, if not will make its own
    - met_dir_name = string, default='mets', the directory in which to look for nnUNet outputs. only used if load_mets is False
  - Functions
    - print: print informatient about the patient to console
    - save: save the object to disk in BMDA format
    - discard_swings: remove lesion timeseries from the patient if it has swings to CR, used in data cleaning
    - resample_all_timeseries: resample all timeseries to the specified intervals and timepoint counts, applies min observation span filter
    - drop_short_timeseries: min observation span filter, used if no resampling is done 
    - discard_unmatched: experimental, dont use
    - discard_gaps: drops time series if they have long gaps in between timepoints
    - validate: checks if all timeseries are built correctly(correct time order, and key matching)
    - get_features: Feature extractor
      - featues = string or lis, if 'all' will get all features, if list of string will get the feature keys specified in the list, if single string will get only that feature
        - options (key strings):
        - all = all features
        - volume = timepoint lesion mask volumes
        - rano = time point rano response
        - radiomics = radiomics for lesion mask and lesion border, 107*2
        - patient_meta = total brain volume
        - total_load = total lesion load and lesion count for patient at timepoints
        - lesion_meta = lesion location in brain (categorical)
        - deep = deep feature vector using the passed extractor (380 if vincents CL model is used)
      - get_keys, bool, default =True returns a list of keys as well if True
      - deep_extractor, object, deep extractor object to get deep feature vector. Object must implement an "exctract_features" method that takes a 5D (batch, channel, x, y, z) tensor as input and returns a feature vector (batch, channel, X)
- load_patient (function): a functional way to load patients, so you dont have to deal with the flags in the Patient object constructor, takes a path to a patient in BMDA format and returns a loaded patient object

#### Metastasis time series
- MetastasisTimeSeries (object)
  - Models an individual lesion timeseries
  - Arguments
    - t0_metastasis = Metastasis object at t0
    - t0_date = Datetime object with the time of t0
    - t0_date_str = the strftime of the t0_datetime object
    - id = String, the metastasis id 'Metastasis X'
  - Functions
  - set_total_lesion_load_at_tp: store the total lesion count and volume in the object for all timepoints
  - check_has_CR_swings: check if long swings exist, if so return True, if not will tag individual swings. Used by discard_swings in patient
  - validate: check if the timeseries is built correctly
  - check_has_no_large_intervals: checks if there are large gaps in between timepoints, returns true if no timegaps are found for the relevant period, used by discard_gaps in patient
  - save: save the timeseries in BMDA format
  - get_trajectory_bspline. computes a bsipline that models the volume over time, returns a scipy bspline object
  - get_observation_day: returns the number of days between t0 and tN
  - resample: resample the timeseries into a different time interval model used by resample_all_time series in patient. Returns a new timeseries object with volumes computed and dummy metastes if nearest neighbor will be real metastis objects. supports bspline, liner and nearest interpolation.
  - correspondence_metrics used by the parsing function to compute overlap and centroid distance for self and a passed metastis in argument
  - append: add a new metastis (i.e. timepoint) to the series
  - print: print a representation of self into the console
  - set_match: experimental, ignore
  - getters get dictionaries of tX as keys and variables (features) as values
  - has some plotting functions to show self as a volume time series plot
- load_series (function)
  - loads a time series object in BMDA format from disk, used by Patient objct if load_patients is True
- cluster_to_series (function)
  - make a dummy series from the lesion cluster centers. experimental, ignore.

#### Metastasis
- Metastasis (object)
  - Models a lesion timepoint, handles correspondence to images and masks
  - Arguments
    - metastasis_mask = sitk.Image object, the binary mask as an sitk image
    - binary_source = the path to where the mask is stored
    - multiclass_source =  if task 524 or 504 were used, the path to the multiclass mask output
    - t1_path =  the path to the t1 image
    - t2_path = the path to the t2 image
  - Functions
    - same_space = checks if another metastasis object have the same spacing, return true if so
    - same_voxel_volume = checks if the met and another have the same volume per voxel
    - print = print a string representation fo the TP into the console
    - rano = compute the rano assignment for the current timepoint
    - resample = resample the images (mask, t1, t2) to a target image
    - getters get single values or lists of values for the current tp
- EmptyMetastasis (object) 
  - dummy object for empty timepoints (i.e. when no mask is available, because no output was generetaed. Essentailly a 0 volume met object)
- InterpolatedMetastasis (object)
  - a dummy object for a metastasis with interpolated volume, loses correspondence to images
- load_metastasis (function)
  - load a metastasis time point in BMDA object from disk


### [Clustering](clustering)
Implementations of clustering algorithms Xmeans and Xshapes, unused and buggy implementations of rpy2 based clustering. Utils to estimate BIC/AIC.
Used in [these scripts](/_clustering_experiments/).

### [Data](data)
Implementations of data loading functions. Loads csvs and prerpocesses them by encoding categoricals, normalization, etc. Also has graph dataset definitions for resampled and raw time series data. 
if the script has the suffix _noisy it means it handles non-resampled data and computes data targets internally

- graph_classification (object)
  - These scripts convert a dataframe to PyG.Data object and implement corresponding datasets
  - Arguments
    - df = the underlying dataframe, loaded by dataframe_preprocessing, with add_index_as_col and time_required as bool True
    - used_timedelta, int or list of ints (noisy), str or list of strs (resampled), the used timedelta days or timepoint prefixes. If list, random_period should be true so the graph timepoint config is randomized. Used for general models
    - ignored_suffixes = list of strings, columns that are in the dataframe but shouldn be in the features
    - target_name = string, the column of the target variable in the dataframe. 
    - fully_connected, bool, default True whether to make the graph fully connected (links from all nodes to all other nodes) or sparsely conected (links to neighbors only)
    - rano_encoding, dictionary, maps rano strings to integer categoricals to encode the target variable
    - extra_features, list of strings, list of features that should be used at all timepoints but dont have the "tX" prefix. Used for demographical features such as sex and age
    - transforms, compose of transforms for graphs, like in any other Torch dataset object
    - direction, str, default none, controls which way the direction is encoded
      - None: undirected
      - "past": directed to past
      - "future": directed to future
    - random_period, bool, default False, whether to randomize the TP config, if true needs a list of values as used_timedelta, used in general models
- feature_eliminators (objects)
  - Implement feature elimination objects (feature selection basically the inverse) to be used in dataframe_preprocessing
- dataframe_preprocessing (function)
  - Dataloader function. loads a dataframe from a feature csv on disk.
  - Arguments
    - path = pathlib.Path object, location of the source csv
    - discard = list of strings, which lesion IDs to ignore, used for manual data cleaning, if "infer" uses Lasso feature eliminator to infer which to drop, if a FeatureEliminator object, uses that.
    - caregorical = list of strings, the columns in the dataframe that should be one-hot encoded. One hot encoding uses the pd.get_dummy function for the specified columns
    - test_size = float, default 0.2 the percentage of data that is used as test set. Uses stratified splitting. uses f"{prefixes[-1]}_{target_suffix}" as split stratifier
    - target_suffix = the column suffix that is used as target
    - prefixes = list of strings, the column prefixes ["t0", "t1", ..., "tN"]
    - drop_suffix = columns to drop always. unused in the project
    - fill = any, passed as the fill variable to pd.fillna
    - normalize_suffix= the columns to normalize, will be standardized with zscore
    - time_required = bool, if timedelta_days should be kept or dropped
    - normalize_volume = the normalization strategy for the volume variable, string. Can be a combination of implemented strategies but just leave it as "std" to standardize because other techniques work but are not useful for classificatin, theyre aimed at regression by log linking or fracturing volumes.
    - save_processed = pl.Path object, the file path (must end in ".csv") where the processed dataframe should be saved. if the path already exists, skip processing and load directly. 
    - add_index_as_col = bool, will add the Lesion ID as a column into the dataframe instead of using it as index
    - target_time (only in noisy ((uninterpolated)) implementation!) = int, the time at which the target variable should be. uses linear interpolation internally to compute a rano variable for this timedelta_days and includes it into the dataframe as the "target_rano" column
- exp_node_prediction = experiments for node prediction in regression tasks, but untested and unused
- derivatives (function)
  - computes the derivatives of the volume over time

### [Prediction](prediction)
Implementations of classical machine learning techniques and evaluation methods. Definition of models, and torch training engine.

### [visualization](visualization) 
Plotting functions for clustering results and evaluation metrics.

### [Deep Features](deep_features)
Feature extractors for deep vectors for the feature generation process.

### [misc](misc_scripts)
diverse scripts for counting, extracting and supplementing data and various other explorative tasks. not much use here.