# making a model to forecast the adj close value to use it in the fbprohet predict method ,
import pandas as pd 
import numpy as np  
from matplotlib import pyplot
import json 
from datetime import date 

stockprice = pd.read_csv("Dataset3.csv",parse_dates=['TimeStamp'])
stockprice = stockprice.rename(columns={'TimeStamp': 'ds','Adj Close': 'y'})
stockprice = stockprice.drop('close', axis=1)


from prophet import Prophet, serialize
from prophet.diagnostics import cross_validation, performance_metrics


model = Prophet().fit(stockprice)

metric_keys = ["mse", "rmse", "mae", "mape", "mdape", "smape", "coverage"]

#cross validation 
metrics_raw = cross_validation(
    model=model,
    horizon="365 days",
    period="180 days",
    initial="900 days",
    parallel="threads",
    disable_tqdm=True,
)
cv_metrics = performance_metrics(metrics_raw)
metrics = {k: cv_metrics[k].mean() for k in metric_keys}

print(f"Logged Metrics: \n{json.dumps(metrics, indent=2)}")

from prophet.serialize import model_to_json

with open('Models/Adj_close_regressor.json', 'w') as fout:
    fout.write(model_to_json(model)) 








