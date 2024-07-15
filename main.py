import pandas as pd 
import numpy as np  
from matplotlib import pyplot
import json 
import mlflow
#setting the tracking uri  
mlflow.set_tracking_uri(uri="http://127.0.0.1:8080")
mlflow.set_experiment("MLflow Quickstart")
from mlflow.models import infer_signature

#reading the datset 
stockprice = pd.read_csv("Dataset3.csv",parse_dates=['TimeStamp'])
stockprice = stockprice.rename(columns={'TimeStamp': 'ds','close': 'y'})

#the artifact path 
ARTIFACT_PATH = "model"
np.random.seed(12345)

#a function to extract the parameters of the model 
def extract_params(pr_model):
    return {attr: getattr(pr_model, attr) for attr in serialize.SIMPLE_ATTRIBUTES}



from prophet import Prophet, serialize
from prophet.diagnostics import cross_validation, performance_metrics
from datetime import date 
from prophet.serialize import model_from_json 

def get_future_timeframe():
    with open('Models/Adj_close_regressor.json', 'r') as fin:
        m_model = model_from_json(fin.read())  # Load model

    # Returns the current local date
    today = date.today() 
    desired_start_date = pd.to_datetime(today)  # Change this to your preferred date

    # Create a future dataframe starting from the specified date
    future = pd.DataFrame({'ds': pd.date_range(start=desired_start_date, periods=30, freq='D')})
    predictions = m_model.predict(future)
    predictions = predictions.rename(columns={'yhat': 'Adj Close'})
    predictions = predictions[['ds', 'Adj Close']]
    return predictions 


def main():
    with mlflow.start_run():
        model = Prophet()
        model.add_country_holidays(country_name='US')
        model.add_regressor('Adj Close')
        model.fit(stockprice)
        params = extract_params(model)

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
        print(f"Logged Params: \n{json.dumps(params, indent=2)}")

        train = model.history

        x = get_future_timeframe()

        predictions = model.predict(x)
        signature = infer_signature(train, predictions)

        mlflow.prophet.log_model(model, artifact_path=ARTIFACT_PATH, signature=signature)
        from prophet.serialize import model_to_json, model_from_json

        with open('Models/model.json', 'w') as fout:
            fout.write(model_to_json(model))  # Save model
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
        model_uri = mlflow.get_artifact_uri(ARTIFACT_PATH)
        print(f"Model artifact logged to: {model_uri}")


if __name__ == "__main__": 
    main()
# before adding the regressor
"""Logged Metrics: 
{
  "mse": 19098.543388232367,
  "rmse": 135.8235318715018,
  "mae": 93.14423708898222,
  "mape": 0.6932897938210141,
  "mdape": 0.35059560700397335,
  "smape": 0.610249119667118,
  "coverage": 0.3863736769751051
}
"""
#after adding the regressor 
"""Logged Metrics: 
{
  "mse": 48226.18267991633,
  "rmse": 216.0939769121088,
  "mae": 134.33397875193845,
  "mape": 0.16141093030538026,
  "mdape": 0.1183625422282204,
  "smape": 0.16733870142477913,
  "coverage": 0.5072936367950653
}
"""