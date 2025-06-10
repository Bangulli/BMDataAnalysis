# Brain Metastases Data Analytics
This is the repository for our paper: titel to be decided

We track and match lesion segmentations in longitudinal MRI (assumes preprocessing with [BMPipeline](https://github.com/Bangulli/BMPipeline)) and store them as individual lesion time series per patient.
This repository provides multiple approaches for longitudinal time series resampling and feature extraction.

## Contents
- [Usage](Usage)
- [Approach](Approach)
- [Packages](Packages)
- [Feature Extraction](feature_extraction)
- [Clustering](clustering)
- [Classical Prediction](classical_prediction)
- [Graph Methods](graph_methods)

## Usage
The environment is managed by anaconda (Python version 3.10)
Anaconda [requirements](requirements.txt)
Some packages are not available on conda, so it is supplemented with pip [requirements](requirements_pip.txt)

- [parse_mets](parse_mets.py) is an example on how to use the data structure to load and parse the results from the [Pipeline](https://github.com/Bangulli/BMPipeline) to a lesion level dataset
- [feature_extraction](feature_extraction.py) is an example on how to extract features from a lesion level dataset.
- The folder [_classification_prediction_experiments](/_classification_prediction_experiments/) contains the scripts used in training
- The folder [_clustering_experiments](/_clustering_experiments/) contains the scripts used for clustering
- The folder [_regression_prediction_experiments](/_regression_prediction_experiments/) contains scripts for experimental regression experiments

## Approach
We extract features from the lesion time series and resample it into a series with homogeneous intervals in between datapoints. Volume resampling is done in either Nearest neighbor, Linear or BSpine interpolation. The time series itself (dataobjects with images) can only be resampled with nearest neighbor to keep the link between lesion volume and image.
Features are stored in csv files with a lesion ID, patient level clinical features and timepoints. Time point columns are denoted by tX as a prefix, so column.split('_') gives the timepoint prefix.
The column tX_timedelta_days stores the days elapsed from t0.

Missing values are imputed by time weighted linear interpolation for volume and radiomic features.
Prediction can be performed with classical methods (by concatenating timepoint feature vectors) or graph methods (by modeling the timeseries itself as a graph)
Clustering is done with stepmix, tsxmeans or tsxshapes. Attempts were made with rpy2 interaction to run more powerful clustering algorithms in R

## Packages
