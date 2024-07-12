import pandas as pd 
import numpy as np  
from matplotlib import pyplot

stockprice = pd.read_csv("Dataset3.csv",parse_dates=['TimeStamp'])
print(stockprice.head())

stockprice = stockprice.rename(columns={'TimeStamp': 'ds','close': 'y'})
print(stockprice.head())


threshold = int((0.8)*len(stockprice))
stock_train = stockprice.iloc[:threshold,]
stock_valid = stockprice.iloc[threshold:]


from prophet import Prophet
model = Prophet()

model.fit(stockprice)


future = model.make_future_dataframe(periods=365)
future.tail()

forecast = model.predict(future)
forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()

fig1 = model.plot(forecast)

pyplot.show()

