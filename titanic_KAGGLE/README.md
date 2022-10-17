# Folder summary

There are two notebooks and one Python script file in this folder. The notebooks are:

- `data_prep` ;
- `ml_analysis`

The script file is:

- `ml_generals.py`

## Requirements

To run these notebooks you will need to have the Python packages `pandas`, `numpy` and `scikit-learn` installed.

## Data preparation

The notebook `data_prep` concerns the data preparation component of KAGGLE's Titanic problem. 

Run this notebook to download relevant datasets via the KAGGLE API and transform them into datasets to be parsed by machine learning models. See `data_prep` for an explanation of the strategy in preparing and transforming the data.

## Machine learning analysis

The notebook `ml_analysis` concerns the modeling component of KAGGLE's Titanic problem. 

Run this notebook to parse the datasets prepared in the `data_prep` notebook and submit the model returning the best results (with respect to a metric of your choosing) via the KAGGLE API.

## Machine learning generals

The script file `ml_generals.py` contains the function `modelValidationResults()` which is a general purpose function. It passes a classifier, training and validation data and returns performance measures of the classifier on this data via the metrics:

- accuracy;
- roc_auc;
- confusion matrix;
- mean cross validation;
- aggregate

where aggregate is an aggregation of the numerical scores `accuracy`, `roc_auc` and `mean cross validation`.

The notebook `ml_analysis` imports this script file and call `modelValidationResults()` in the course of its analysis.






