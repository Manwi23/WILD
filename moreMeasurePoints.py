from DataProcessing import Process, DeleteUnwanted
from os.path import exists
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import xgboost as xgb
from trees import calc_score

def LinearRegressionTest(train_df, test_df):

    features = list(train_df.columns)
    features.remove('target')

    Y_train = np.array(train_df['target'])
    Y_train = Y_train.reshape(-1, 1)
    X_train = np.array(train_df[features])

    X_test = np.array(test_df[features])
    Y_test = np.array(test_df['target'])
    Y_test = Y_test.reshape(-1, 1)

    reg = LinearRegression().fit(X_train, Y_train)
    print("=====Linear Regression=====")
    #print("features:", features)
    print("Train reg.score:", reg.score(X_train, Y_train))
    print("Test reg.score:", reg.score(X_test, Y_test))

# based on Kasia's code
def RGBoostTest(train_df, test_df):
    train_df, val_df = train_test_split(train_df, test_size = 0.16666666666, shuffle=False)

    xgb_model = xgb.XGBRegressor(booster="gbtree",objective="reg:squarederror")

    target_train = train_df['target']
    train_df = train_df.drop(columns=['target'])
    target_val = val_df['target']
    val_df = val_df.drop(columns=['target'])
    target_test = test_df['target']
    test_df = test_df.drop(columns=['target'])

    xgb_model.fit(train_df, target_train, early_stopping_rounds=5, eval_set=[(train_df, target_train), (val_df, target_val)])

    print("=====RGBoost=====")
    ytarget = xgb_model.predict(train_df)
    mm = target_train.mean()
    print("Train score:", calc_score(target_train, ytarget, mm))

    ytarget = xgb_model.predict(val_df)
    mm = target_val.mean()
    print("Val score:", calc_score(target_val, ytarget, mm))

    ytarget = xgb_model.predict(test_df)
    mm = target_test.mean()
    print("Test score:", calc_score(target_test, ytarget, mm))


if __name__ == '__main__':

    number_of_points = 3
    #number_of_points = 5
    HotEncodedColumns = ['Wind', 'Condition']
    repeatedColumns = ["Temperature"]

    time_deltas = [4, 8, 12, 24, 48] # 2h, 4h, 6h, 12h, 24h
    dfs = {}

    for time_delta in time_deltas:
        filename = f"data/Weather-n{number_of_points}-every{str(time_delta/2)}h-measureT"
        if exists(filename):
            print(f"The file {filename} already exists; reading database from file.")
            df = pd.read_csv(filename)
        else:
            timestamps = [time_delta*(i+1) for i in range(number_of_points)]
            df = Process(timestamps, repeatedColumns, HotEncodedColumns)
            print(f"\n\nDataframe {time_delta/2} columns:\n")
            print(df.columns)
            df.to_csv(filename)
        dfs[time_delta] = df
        
    for time_delta, df in dfs.items():
        print("\n-------------------------------------------------------------")
        print(f"Results for dataframe with measurements every {time_delta/2}h\n")
        df = DeleteUnwanted(df, ['Date', 'index', 'Time'])
        train_df, test_df = train_test_split(df, test_size = 0.14285714285, shuffle=False)
        LinearRegressionTest(train_df, test_df)
        RGBoostTest(train_df, test_df)

"""
Results for measurments every 24h are low.
Probably because we have less measurements of 
the same time of the day as target

Best results for 4, 6 and 12 hours
"""
