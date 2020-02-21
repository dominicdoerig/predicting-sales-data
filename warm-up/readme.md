# Warm-Up

## Goal
The warm-up is executed before the start of the M-5 competition and should give us valuable insights for the challenge.
We will perform some data exploration on highly related datasets (the dataset of M5 is not yet available).

## Dataset
- 'Walmart Recruiting - Store Sales Forecasting' available on https://www.kaggle.com/c/walmart-recruiting-store-sales-forecasting/data

- The dataset is not in the repository. It must be downloaded manually, unzipped and placed in warm-up/data/ .

- Download the data either manually using aforementioned link or using the Kaggle API (Command: kaggle competitions download -c walmart-recruiting-store-sales-forecasting)

## Difference Between This Dataset and DataSet of M-5 Challenge
- Granularity: This dataset contain only data on store and department level and the goal is to predict the total sale of a store. On the other hand, the dataset of the M-5 will be more fine-granular with hierarchy State > Store > Category > Department > Item. The goal of the M-5 is to predict the sales on item-level.
- Daily vs. Weekly Data: This dataset contain data on weekly basis whereas the data of the M5-Challenge contain daily data.
- Meta-Information: The dataset of the M5 Challenge includes much more meta-information like sales prices than this dataset here.
