import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import timedelta, date
import scipy.stats as sstats
from os.path import exists
from sklearn.model_selection import train_test_split
import math

df = pd.read_csv("precipitation.csv")
# print(df.head())

rename_dict={"PS" : "Pressure", 
            "RH2M" : "Relative Humidity",
            "QV2M" : "Humidity",
            "PRECTOT" : "Precipitation"}

df = df.rename(columns=rename_dict)

processed = pd.read_csv("processed.csv")

precip_dict = {}
c= 0
for _, row in df.iterrows():
    day = date(int(row["YEAR"]), int(row["MO"]), int(row["DY"]))
    prev_day = day + timedelta(days = -1)
    precip_dict[str(prev_day)] = row["Precipitation"]
    if c < 10: print(str(prev_day))
    c += 1

useful = []

for _, row in processed.iterrows():
    day = row['Date']
    useful += [precip_dict[day]]

precip = pd.DataFrame({'Precip.' : useful})

processed['Precip.'] = precip['Precip.']

print(processed.head())

processed.to_csv('processed_rain.csv')