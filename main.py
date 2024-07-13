import pandas as pd 
import numpy as np  
from matplotlib import pyplot
from sklearn.metrics import mean_absolute_error
stockprice = pd.read_csv("Dataset3.csv",parse_dates=['TimeStamp'])
stockprice = stockprice.rename(columns={'TimeStamp': 'ds','close': 'y'})


threshold = int((0.2)*len(stockprice))
stock_test = stockprice.iloc[:threshold,]
stock_train = stockprice.iloc[threshold:]



from prophet import Prophet
model = Prophet(daily_seasonality = True)

model.fit(stock_train)

future = model.make_future_dataframe(periods=1241,include_history = False)
#future.tail()




forecast = model.predict(future)
#forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()
print(len(forecast))
fig1 = model.plot(forecast)

pyplot.show()

# calculate MAE between expected and predicted values for the test set 
y_true = stock_test['y'].values
y_pred = forecast['yhat'].values
mae = mean_absolute_error(y_true, y_pred)
print('MAE: %.3f' % mae)
#MAE = 72.252