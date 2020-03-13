import pandas as pd
import os


def get_m5_root_dir():
    """Returns project root folder."""
    return os.getcwd()


def transform_dataframe(df_sales, df_calendar, df_prices, save_to_path=None):
    df = df_sales.melt(
        id_vars=['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id'],
        var_name='d',
        value_name='sale')
    df = df.merge(df_calendar, how='left')
    df = df.merge(df_prices, how='left')

    if save_to_path is not None:
        df.to_csv(save_to_path, index=False)

    return df
    
    
def import_m5_data():
    root_dir = get_m5_root_dir()
    df_calendar = pd.read_csv(root_dir + '/data/input/calendar.csv')
    df_sales = pd.read_csv(root_dir + '/data/input/sales_train_validation.csv')
    df_prices = pd.read_csv(root_dir + '/data/input/sell_prices.csv')
    return df_calendar, df_sales, df_prices
    
    