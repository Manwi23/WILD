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

def apply_pearson(target, column):
    r, p = sstats.pearsonr(target, column)
    # print(r, p, what)
    return p

def apply_chi2(target_grouped, column):
    cross = pd.crosstab(target_grouped, column)
    stat,p,dof,expec = sstats.chi2_contingency(cross)
    return p

def group_column(column, name):
    kmeans = KMeans(n_clusters=20).fit(np.array(column).reshape(-1,1))
    return pd.DataFrame({name: list(map(lambda x: str(x), kmeans.labels_))})

def calc_score(true_vals, predicted, mean):
    v = 0
    d = 0
    for t, p in zip(true_vals, predicted):
        val = (t - p)**2
        base = (t - mean)**2
        if math.isnan(val):
            v+=base
        else:
            v+=val
        d += base

    return (1 - v/d)

# results_df = read_or_process_data()
dataname = "day_p.csv"
results_df = pd.read_csv(dataname)

ps = []

target = results_df['target']
target_grouped = group_column(target, 'target_g')
# print(target_grouped)
forbidden = ['target', 'index', 'Unnamed: 0', 'Date', 'Time', 'Unnamed: 0.1']

# print(results_df.dtypes['target'])

res_data = results_df.copy()

for i in results_df.columns:
    if i not in forbidden:
        if results_df.dtypes[i] != 'object':
            # print(i, results_df.dtypes[i])
            # print(i)
            p = apply_pearson(target, results_df[i])
            ps += [(i,p)]
        else:
            # print(i)
            # col = group_column(results_df[i])
            p = apply_chi2(target_grouped['target_g'], results_df[i])
            ps += [(i,p)]
            res_data = res_data.drop(columns=[i])
        # res_data += [results_df[i]]
    else:
        # print(i, 'dropped')
        res_data = res_data.drop(columns=[i])

# res_data = pd.DataFrame(res_data)
# print(res_data)

# print(ps)
for i in ps:
    print(i)

res_data['target'] = target
train_df, test_df = train_test_split(res_data, test_size = 0.14285714285, shuffle=False)

# xgb_model = xgb.XGBRegressor(objective="reg:squarederror", 
#                             booster='gblinear', eta=0.5, n_estimators=8)

xgb_model = xgb.XGBRegressor(booster="gblinear",objective="reg:squarederror", n_estimators=8)

target_train = train_df['target']
train_df = train_df.drop(columns=['target'])
target_test = test_df['target']
test_df = test_df.drop(columns=['target'])

xgb_model.fit(train_df, target_train)

ytarget = xgb_model.predict(train_df)
mm = target_train.mean()
print("* train:", calc_score(target_train, ytarget, mm))

ytarget = xgb_model.predict(test_df)
mm = target_test.mean()
print("* test:", calc_score(target_test, ytarget, mm))

print(dataname)