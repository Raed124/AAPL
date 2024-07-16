#!/usr/bin/env python3
from datetime import date 
from prophet.serialize import model_from_json 
import pandas as pd 

with open('Models/model.json', 'r') as fin:
    m_model = model_from_json(fin.read())  # Load model

from main import get_future_timeframe

x = get_future_timeframe() 

predictions = m_model.predict(x)

print(f"forecast:\n${predictions.head(30)}")





