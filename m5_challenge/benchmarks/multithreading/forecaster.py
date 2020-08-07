from fbprophet import Prophet


def run_prophet(df_prophet):
    """
    Forecast next 28 days using Facebook's Prophet

    :param df_prophet: pandas.DataFrame with columns ds and y as required by Prophet
    :return: list containing the forecasts
    """
    model = Prophet(daily_seasonality=True, yearly_seasonality=True)
    model.fit(df_prophet)
    future = model.make_future_dataframe(periods=28, include_history=False)
    forecast = model.predict(future)['yhat'].to_list()
    return forecast
