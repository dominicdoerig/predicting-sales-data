# predicting-sales-data
see https://github.com/dominicdoerig/predicting-sales-data/blob/master/README.md


## Folder Structure 
- data/ -> Folder containing all data (.csv files)
- models/ -> Folders where the LightGBM models will be saved
- benchmarks/ -> Folder containing files regarding benchmark models
- benchmarks/multithreading/ -> Folder containing Python files using multithreading to forecast using fbprophet  
- data_exploration/ -> Folder with one notebook used for exploratory data analysis
- competitive_model/ -> Folder containing all files regading the competitive model (LightGBM)
- utils.py -> Python file containing util functions


## Data
The data is not in the repo. First, download the data either from https://www.kaggle.com/c/m5-forecasting-accuracy/data or using the Kaggle API and the command ’kaggle competitions download -c m5-forecastingaccuracy’
Store unzipped input data in path m5_challenge/data/input/


## Pipeline to Run Experiments
1) store unzipped input data in path m5_challenge/data/input/
2) run competitive_model/preprocess_data.ipynb
3) run competitive_model/gbm_feature_engineering.ipynb
4) run competitive_model/gbm_forecasting.ipynb

