import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import scipy.stats as sstats
from os.path import exists
from sklearn.model_selection import train_test_split
import math
from sklearn.cluster import KMeans
import xgboost as xgb

from explore import read_or_process_data

def isnight(time_t):
    t = time_t.split()[0]
    h = int(t.split(':')[0])
    if 'PM' in time_t:
        return h > 4 and h != 12
    return h < 7 or h == 12


results_df = pd.read_csv("1HotEncoded.csv")

# results_df = results_df.drop(columns=['Unnamed: 0', 'Unnamed: 0.1', 'index'])

# print(results_df.head())

# słońce wstaje koło 7 w połowie lutego, a idzie spać kolo 17

results_night = results_df[results_df['Time'].apply(isnight)]
results_day = results_df[results_df['Time'].apply(isnight)==False]

print(results_night.head())

results_night.to_csv("night.csv")
results_day.to_csv("day.csv")