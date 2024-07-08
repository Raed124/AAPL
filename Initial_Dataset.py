import requests 
import pandas as pd 
import json 
import csv
from Api_Key import Key
url = 'https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol=AAPL&interval=5min&outputsize=full&apikey='+Key
r = requests.get(url)
data = r.json()
T = data['Meta Data']
print(T)
P = data["Time Series (Daily)"]
rows = []
Dict = {}
for v ,(i,k) in enumerate(P.items()):
    Dict['TimeStamp'] = i
    Dict['open'] = k['1. open']
    Dict['high'] = k['2. high']
    Dict['low'] = k['3. low']
    Dict['close'] = k['4. close']
    Dict['volume'] = k['5. volume']
    rows.append(Dict)

with open("Dataset.csv", "w", newline="") as f:
    w = csv.DictWriter(f, Dict.keys())
    w.writeheader()
    w.writerows(rows)    

