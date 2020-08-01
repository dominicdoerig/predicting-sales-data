# predicting-sales-data
## Overview
This repo is related to the participation the in the M5-Challenge; a worldwide forecasting competition that is carried out on hierarchical sales data provided by Walmart. The goal is to forecast daily sales for the next 28 days. 

The sales data is covers stores in three US States (California, Texas, and Wisconsin) and includes item level, department, product categories, and store details. Totally, the data set contains 42,840 time series. In addition, it has explanatory variables such as price, promotions, day of the week, and special events. \cite{kaggle_m5_accuracy}

The M5 challenge is the latest episode of the Makridakis Competitions, a series of time-series forecasting competitions organized and led by forecasting researcher Prof. Spyros Makridakis, intending to evaluate and compare the accuracy of different forecasting methods. These challenges enjoy an excellent reputation in the market and have an enormous influence on the field of forecasting as they focus attention on what models produce good forecasts, rather than on the mathematical properties of those models.

In this work, we focus on Gradient Boosting Decision Trees (GBDT). The implementation of these models requires only limited data pre-processing, allow the combination of the sales data with the provided explanatory variables and a fast computation. Furthermore, GBDTs are proven successful across many domains and robust to outliers. We iteratively improve the performance of the model by adding new features and increase the complexity of the model.

The implemented model outperforms all implement benchmarks and I overall achieved a top 28% rank in the M5 competition.

## Packages
- The packages reported in requirements.txt are used
- To install all packages, run 'pip install -r requirements.txt'

## Data
- The data is not in the repo. First, download the data either from https://www.kaggle.com/c/
m5-forecasting-accuracy/data or using the Kaggle API and the command ’kaggle competitions download -c m5-forecastingaccuracy’
- Store unzipped input data in path m5_challenge/data/input/ 

## Pipeline to Run Experiments
1. store unzipped input data in path m5_challenge/data/input/ 
2. run 02_preprocess_data.ipynb
3. run 06_gbm_feature_engineering.ipynb
4. run 06_gbm_forecasting.ipynb

## Repo Structure
- warm_up: Folder related to a preparatory project executed before the launch of the M5 challenge 
- m5_challenge: Folder containing all files related to the M5 competition
