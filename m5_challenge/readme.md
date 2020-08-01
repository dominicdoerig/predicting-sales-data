# predicting-sales-data
see https://github.com/dominicdoerig/predicting-sales-data/blob/master/README.md


## Folder Structure 
- data/ -> Folder containing all data (.csv files)
- models/ -> Folders where the LightGBM models will be saved
- multithreading/ -> Folder containing one Python file using multithreading to forecast using fbprophet  
- 01_data_exploration.ipynb -> Notebook used for exploratory data analysis
- 02_preprocess_data.ipynb -> Notebook that transforms the raw data into a usable format
- 03_forecasting_snaive.ipynb -> Notebook computing forecasts using Seasonal Naive approach (Benchmark Model)
- 04_forecast_fbprophet.ipynb -> Notebook computeing forecasts using FB-Prophet on aggregated data and then desaggregate 
- 05_forecast_fbprophet_item_level.ipynb -> Notebook computing forecasts using FB-Prophet approach (Benchmark Model)
- 06_gbm_compute_scaling_factors_for_rmsse.ipynb -> Notebook computing the scaling factors of WRMSSE
- 06_gbm_compute_weights_for_wrmsse.ipynb -> Notebook computing the weights of WRMSSE
- 06_gbm_feature_engineering.ipynb -> Notebook used for feature engineering
- 06_gbm_feature_engineering_iterative_forecasting.ipynb -> Notebook used for feature engineering used for iterative forecasting
- 06_gbm_forecasting-iterative_forecasting.ipynb -> Notebook used for computation of forecasts using iterative forecasting (4x 7-step-forecast)
- 06_gbm_forecasting.ipynb -> Notebook used to compute forecasts (Competitive Model)
- 06_gbm_residual_analysis.ipynb -> Notebook used for residual analysis
- forecaster.py -> Python file used to run FB-Prophet
- utils.py -> Python file containing util functions


## Data
The data is not in the repo. First, download the data either from https://www.kaggle.com/c/m5-forecasting-accuracy/data or using the Kaggle API and the command ’kaggle competitions download -c m5-forecastingaccuracy’
Store unzipped input data in path m5_challenge/data/input/


## Pipeline to Run Experiments
store unzipped input data in path m5_challenge/data/input/
run 02_preprocess_data.ipynb
run 06_gbm_feature_engineering.ipynb
run 06_gbm_forecasting.ipynb

