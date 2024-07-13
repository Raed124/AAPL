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
print(stockprice.shape)
fig2 = stockprice.plot(kind='scatter',
        x='ds',
        y='y',
        color='red')
#the artifact path 
ARTIFACT_PATH = "model"
np.random.seed(12345)

#a function to extract the parameters of the model 
def extract_params(pr_model):
    return {attr: getattr(pr_model, attr) for attr in serialize.SIMPLE_ATTRIBUTES}


"""#train test split 
threshold = int((0.2)*len(stockprice))
stock_test = stockprice.iloc[:threshold,]
stock_train = stockprice.iloc[threshold:]"""



from prophet import Prophet, serialize
from prophet.diagnostics import cross_validation, performance_metrics


with mlflow.start_run():
    model = Prophet().fit(stockprice)

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
    #print(train.shape())
    predictions = model.predict(model.make_future_dataframe(30))
    signature = infer_signature(train, predictions)

    mlflow.prophet.log_model(model, artifact_path=ARTIFACT_PATH, signature=signature)
    Path = './Models/'
    mlflow.prophet.save_model(model,Path)
    mlflow.log_params(params)
    mlflow.log_metrics(metrics)
    model_uri = mlflow.get_artifact_uri(ARTIFACT_PATH)
    #model_uri = 'models/'
    print(f"Model artifact logged to: {model_uri}")


#loaded_model = mlflow.prophet.load_model(model_uri)

loaded_model_2 = mlflow.prophet.load_model(Path)

forecast = loaded_model_2.predict(loaded_model_2.make_future_dataframe(60,include_history = False))

print(f"forecast:\n${forecast.head(30)}")


#include_history = False

fig1 = model.plot(forecast)

pyplot.show()

# calculate MAE between expected and predicted values for the test set 
#MAE = 72.252