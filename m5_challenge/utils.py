import pandas as pd
import os
import numpy as np


def get_m5_root_dir():
    """Returns project root folder."""
    return os.path.dirname(os.path.realpath(__file__))



def transform_dataframe(df_sales, df_calendar, df_prices, save_to_path=None):
    """
    Transforms the sales dataframe and merges it with calendar and price data.
    """
    df = df_sales.melt(
        id_vars=['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id'],
        var_name='d',
        value_name='sale')
    df = df.merge(df_calendar, how='left')
    df = df.merge(df_prices, how='left')

    if save_to_path is not None:
        df.to_csv(save_to_path, index=False)

    return df


def plot_forecasts(df_hist, df_pred, title=None, figsize=(15, 3)):
    """
    Plot the historical time series and its point forecasts.
    
    :param df_hist: Dataframe with the historical 'sale' and their 'date'
    :type df_hist: Pandas.DataFrame
    :param df_pred: Dataframe with the predicted 'sale' and their 'date'
    :type df_pred: Pandas.DataFrame
    :param title: Title of the plot
    :type title: str
    :param figsize: (width, height) in inches
    :type figsize: tuple
    :return: None
    :rtype: None
    """

    df_combined = pd.concat([df_hist, df_pred])

    ax = df_combined.plot(x='date', y='sale',
                          figsize=figsize, color='orange', legend=False)
    df_hist.plot(x='date', y='sale', ax=ax, legend=False, title=title)


def import_m5_data(reduce_memory=False):
    """
    Imports the input data for the M5-Challenge

    :param reduce_memory: bool
    :type reduce_memory: whether or not the memory usage of the dataframe should be reduced (default: False)
    :return: Three dataframes: calendar, sales and price data
    :rtype: pandas.DataFrame
    """
    root_dir = get_m5_root_dir()
    if reduce_memory is True:
        df_calendar = pd.read_csv(root_dir + '/data/input/calendar.csv').pipe(reduce_mem_usage)
        df_sales = pd.read_csv(root_dir + '/data/input/sales_train_validation.csv').pipe(reduce_mem_usage)
        df_prices = pd.read_csv(root_dir + '/data/input/sell_prices.csv').pipe(reduce_mem_usage)
    else:
        df_calendar = pd.read_csv(root_dir + '/data/input/calendar.csv')
        df_sales = pd.read_csv(root_dir + '/data/input/sales_train_validation.csv')
        df_prices = pd.read_csv(root_dir + '/data/input/sell_prices.csv')

    return df_calendar, df_sales, df_prices


def reduce_mem_usage(df, verbose=True):
    """
    Reduces the required memory of a pandas.Dataframe by reducing the storage capacity of numerical data types

    """
    numerics = ["int16", "int32", "int64", "float16", "float32", "float64"]
    start_mem = df.memory_usage().sum() / 1024 ** 2
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == "int":
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if (
                        c_min > np.finfo(np.float16).min
                        and c_max < np.finfo(np.float16).max
                ):
                    df[col] = df[col].astype(np.float16)
                elif (
                        c_min > np.finfo(np.float32).min
                        and c_max < np.finfo(np.float32).max
                ):
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
    end_mem = df.memory_usage().sum() / 1024 ** 2
    if verbose:
        print(
            "Mem. usage of decreased to {:5.2f} Mb ({:.1f}% reduction)".format(
                end_mem, 100 * (start_mem - end_mem) / start_mem
            )
        )
    return df
