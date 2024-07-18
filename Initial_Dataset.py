#!/usr/bin/env python3
import requests 
import pandas as pd 
import csv
from Api_Key import Key

Dataset = 'Dataset3.csv'

#Downlaoding the AAPL stock for the past 20 years  from alphvantage api , please check their website 
url = 'https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol=AAPL&interval=5min&outputsize=full&apikey='+Key
r = requests.get(url)
if(r.status_code != 200):
    print("Error : Could not connect ! ")
    exit(1)
else:
    print("Connection Successful ! ")
    
data = r.json()
T = data['Meta Data']
print(T)



P = data["Time Series (Daily)"]
rows = []

for v ,(i,k) in enumerate(P.items()):
    Dict = {}
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
# downloading the S&P 500 Information technology sector index over for the past 20 years 
import yfinance as yf
from datetime import date
today = date.today()
ticker = "^SP500-45"
data = yf.download(ticker, start="1999-11-01", end=today)
data.to_csv('Dataset2.csv')


#this will happen in either cases
Dataframe = pd.read_csv('Dataset.csv')
Dataframe2 =pd.read_csv('Dataset2.csv')

Dataframe2 = Dataframe2.rename(columns={'Date': 'TimeStamp'})
#I've tried to rename the Dtae column to Timestamp so that i can join them on time stamp but I've failed to rename the column so I've rename the open column and joined them on the open
x = pd.merge(Dataframe, Dataframe2, on = "TimeStamp", how = "inner") 

x = x[['TimeStamp','close','Adj Close']].copy()

x.to_csv(Dataset,index=False)  
