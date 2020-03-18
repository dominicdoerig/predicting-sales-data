
from fbprophet import Prophet

def worker_prophet(df_prophet):
    model = Prophet(daily_seasonality=True, yearly_seasonality=True) 
    model.fit(df_prophet)
    future = model.make_future_dataframe(periods=28, include_history=False)
    forecast = model.predict(future)['yhat'].to_list()
    return forecast
