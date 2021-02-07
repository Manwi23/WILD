from DataProcessing import process, deleteUnwanted
from os.path import exists
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import xgboost as xgb
from trees import calc_score
from neural_nets import NNTest
from datetime import datetime

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
def XGBoostTest(train_df, test_df):
    train_df, val_df = train_test_split(train_df, test_size = 0.16666666666, shuffle=False)

    print("=====XGBoost=====")
    
    target_train = train_df['target']
    train_df = train_df.drop(columns=['target'])
    target_val = val_df['target']
    val_df = val_df.drop(columns=['target'])
    target_test = test_df['target']
    test_df = test_df.drop(columns=['target'])

    matrix = xgb.DMatrix(train_df, target_train)
    matrix_val = xgb.DMatrix(val_df, target_val)

    xgb_model = xgb.train({'booster':"gbtree",'objective':"reg:squarederror"}, 
                matrix, early_stopping_rounds=10, 
                evals=[(matrix, "target_train"), (matrix_val, "target_val")],
                verbose_eval=False)

    m_train = xgb.DMatrix(train_df)
    ytarget = xgb_model.predict(m_train)
    mm = target_train.mean()
    print("Train score:", calc_score(target_train, ytarget, mm))

    m_val = xgb.DMatrix(val_df)
    ytarget = xgb_model.predict(m_val)
    mm = target_val.mean()
    print("Val score:", calc_score(target_val, ytarget, mm))

    m_test = xgb.DMatrix(test_df)
    ytarget = xgb_model.predict(m_test)
    mm = target_test.mean()
    print("Test score:", calc_score(target_test, ytarget, mm))


def single_location(years = [2014,2015,2016,2017,2018,2019,2020], 
                    date_start="01-01", date_end='03-31', rain_present=True,
                    colab_presentation = False, last_part = []):

    number_of_points = 3
    #number_of_points = 5
    HotEncodedColumns = ['Wind', 'Condition']
    repeatedColumns = ["Temperature",'Wind', 'Condition']
    time_deltas = [4, 8, 12, 24, 48] # 2h, 4h, 6h, 12h, 24h
    dfs = {}

    for time_delta in time_deltas:
        filename = f"data/Weather-n{number_of_points}-every{str(time_delta/2)}h-measureT" if not colab_presentation else \
                    f"data/Weather-n{number_of_points}-every{str(time_delta/2)}h-measureT-predicting-now"
        if exists(filename):
            print(f"The file {filename} already exists; reading database from file.")
            df = pd.read_csv(filename)
        else:
            timestamps = [time_delta*(i+1) for i in range(number_of_points)]
            if colab_presentation:
                df = process(timestamps, repeatedColumns, HotEncodedColumns,
                                years, date_start, date_end, rain_present,
                                last_different=True, last_one=last_part)
            else: 
                df = process(timestamps, repeatedColumns, HotEncodedColumns, years, date_start, date_end, rain_present)
            print(f"\n\nDataframe {time_delta/2} columns:\n")
            print(df.columns)
            df.to_csv(filename)
        dfs[time_delta] = df
        
    for time_delta, df in dfs.items():
        print("\n-------------------------------------------------------------")
        print(f"Results for dataframe with measurements every {time_delta/2}h\n")
        df = deleteUnwanted(df, ['Date', 'index', 'Time'])
        train_df, test_df = train_test_split(df, test_size = 0.14285714285, shuffle=False)
        LinearRegressionTest(train_df, test_df)
        XGBoostTest(train_df, test_df)
        NNTest(train_df, test_df)

def extend_column_list(cols, suffixes):
    new_cols = []
    for col in cols:
        new_cols.append(col)
        for s in suffixes:
            new_cols.append(col+"_"+s)
    
    return new_cols


def multi_location():
    number_of_points = 3
    #number_of_points = 5
    HotEncodedColumns = ['Wind', 'Condition']
    repeatedColumns = ["Temperature",'Wind', 'Condition']
    time_deltas = [4, 8, 12, 24, 48] # 2h, 4h, 6h, 12h, 24h
    places = ["wroclaw", "poznan", "katowice", "prague", "dresden"]
    HotEncodedColumns = extend_column_list(HotEncodedColumns, places[1:])
    repeatedColumns = extend_column_list(repeatedColumns, places[1:])

    dfs = {}

    for time_delta in time_deltas:
        filename = f"data/WeatherJoint-n{number_of_points}-every{str(time_delta/2)}h-measureT"
        if exists(filename):
            print(f"The file {filename} already exists; reading database from file.")
            df = pd.read_csv(filename)
            # def make_float(v):
            #     try:
            #         return float(v)
            #     except:
            #         print(v)
            #         return 0
            # # print(df[df["Precip._dresden"] == "Cloudy"])

            # df["Precip._dresden"] = df["Precip._dresden"].apply(make_float)
        else:
            timestamps = [time_delta*(i+1) for i in range(number_of_points)]
            df = process(timestamps, repeatedColumns, HotEncodedColumns, place=None, places=places)
            print(f"\n\nDataframe {time_delta/2} columns:\n")
            print(df.columns)
            df.to_csv(filename)
        dfs[time_delta] = df
        
    for time_delta, df in dfs.items():
        print("\n-------------------------------------------------------------")
        print(f"Results for dataframe with measurements every {time_delta/2}h\n")
        df = deleteUnwanted(df, ['Date', 'index', 'Time'])
        train_df, test_df = train_test_split(df, test_size = 0.14285714285, shuffle=False)
        LinearRegressionTest(train_df, test_df)
        XGBoostTest(train_df, test_df)
        NNTest(train_df, test_df)

if __name__ == '__main__':
    multi_location()
    

"""
Results for measurments every 24h are low.
Probably because we have less measurements of 
the same time of the day as target

Best results for 4, 6 and 12 hours
"""
