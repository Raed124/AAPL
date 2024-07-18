#!/usr/bin/env python3
import csv
from datetime import date,datetime,timedelta
today = date.today()

#checking wether for the presence of the dataset

Dataset = 'Dataset3.csv'

def check_last_month():
    with open(Dataset, "r", newline="") as f:
        r = csv.DictReader(f)
        x = next(r)['TimeStamp']
        x = datetime.strptime(x,'%Y-%m-%d').date()
        print(today - x)
        delta = timedelta(days=30)
        print(delta)
        if((today - x) < delta):
            return True
        else:
            return False 

def Scrape_Until_Yesterday():
    pass


from os.path import exists
if(exists(Dataset)):
    if(check_last_month()):
        Scrape_Until_Yesterday()
        print('Dataset is up to date proceeding to training')
    else:
       import Initial_Dataset 
else:
    import Initial_Dataset 