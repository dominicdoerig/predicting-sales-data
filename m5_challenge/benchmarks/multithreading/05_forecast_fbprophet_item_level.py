import time
from multiprocessing import Pool, cpu_count

import numpy as np
import pandas as pd

from tqdm import tqdm

from m5_challenge.benchmarks.multithreading import forecaster

from m5_challenge import utils


def create_prophet_df(pd_series):
    """
    Brings a pandas.Series into the form required by Prophet

    :param pd_series: pandas.Series to be transformed.
    :return: transformed pandas.Dataframe
    """
    d_string = [f'd_{di}' for di in list(range(1, 1914))]
    ds = pd.date_range(start='2011-01-29', end='2016-04-24')
    df = pd.DataFrame({
        'ds': ds,
        'y': pd_series[d_string].values
    })
    return df


if __name__ == '__main__':

    debugging = True
    save_results = False

    print(f' ***** DEBUGGING = {debugging} *****')
    print(f' ***** SAVE RESULTS = {save_results} *****')

    print('Start Main Method')

    start = time.time()

    print('Data import')
    submission = pd.read_csv(f'{utils.get_m5_root_dir()}/data/input/sample_submission.csv')
    df_calendar, df_sales, df_prices = utils.import_m5_data(reduce_memory=False)

    # Reduce size of dataframe if debugging is True
    if debugging is True:
        df_sales = df_sales.head(50)

    print('Create a list of dataframes containing the sales figures of one item')
    list_of_df_sales = [create_prophet_df(row) for i, row in df_sales.iterrows()]

    print('Initiate a pool of workers')
    pool = Pool(cpu_count())

    print('Paralell computation of forecasts')
    results = pool.map(forecaster.run_prophet, tqdm(list_of_df_sales))

    print('Transform forecasts into a dataframes of the structure Kaggele requires')
    results_df = pd.DataFrame(results, columns=[f'F{di}' for di in list(range(1, 29))])

    print('Add column wiht item-ids to the dataframe')
    results_df.insert(0, 'id', df_sales['id'])

    print('Save dataframe in submissions folder')
    df_submission = results_df.append(
        results_df.replace(to_replace='validation', value='evaluation', regex=True))
    filename = utils.get_m5_root_dir() + '/data/submissions/fbprophet_on_item_level_v2.csv'

    if save_results is True:
        print('Saving results')
        df_submission.to_csv(filename, index=False)

    print('**********************************************************************************')
    print('********************************* FINISHED ***************************************')

    print(f'Computation took {np.round((time.time() - start) / 60, 2)} minutes in total.')
    print(f'Computation took {np.round((time.time() - start) / df_sales.shape[0], 2)} seconds per item.')

    # Submit via Kaggle API or with website https://www.kaggle.com/c/m5-forecasting-accuracy/submissions
    # !kaggle competitions submit -c m5-forecasting-accuracy -f data/submissions/fbprophet_on_item_level.csv -m"fbprophet_on_item_level.csv"
