from DataProcessing import process, deleteUnwanted
from os.path import exists
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge
import xgboost as xgb
from trees import calc_score
from datetime import datetime
import seaborn as sns
import matplotlib.pyplot as plt
from weathermodel import Model
from neural_nets import NNModel, NNTest


class LinearRegressionModel(Model):
    def __init__(self, train_df, test_df, label = ""):
        super().__init__(train_df, test_df, label)
        self.label = "LinearReg " + self.label
        # prepare data for fitting:
        self.features = list(self.train_df.columns)
        self.features.remove('target')

    def prepare_data(self, df):
        Y = np.array(df['target'])
        Y = Y.reshape(-1, 1)
        X = np.array(df[self.features])
        return X, Y

    def fit(self, ridge = True):
        X_train, Y_train = self.prepare_data(self.train_df)
        if ridge:
            self.model = Ridge().fit(X_train, Y_train)
        else:
            self.model = LinearRegression().fit(X_train, Y_train)
    
    def compute_score(self, df):
        X, Y = self.prepare_data(df)
        return self.model.score(X, Y)

    def show_score(self):
        print(self.label + " Scores:")
        print("Train reg.score:", self.compute_score(self.train_df))
        print("Test reg.score:", self.compute_score(self.test_df))

    def predict(self, X):
        #X, _ = self.prepare_data(df)
        return self.model.predict(X)
    
    def predict_df(self, df):
        X, _ = self.prepare_data(df)
        return self.predict(X)
    


class XGBoostModel(Model):
    def __init__(self, train_df, test_df, label = ""):
        super().__init__(train_df, test_df, label)
        self.label = "XGBoost " + self.label
        self.train_df, self.val_df = train_test_split(self.train_df, test_size = 0.16666666666, shuffle=False)
        
    def prepare_data(self, df):
        target = df['target']
        X_df = df.drop(columns=['target'])
        matrix = xgb.DMatrix(X_df, target)
        return X_df, target, matrix

    def fit(self):
        _, _, matrix = self.prepare_data(self.train_df)
        _, _, matrix_val = self.prepare_data(self.val_df)
        self.model = xgb.train({'booster':"gbtree",'objective':"reg:squarederror"}, 
                matrix, early_stopping_rounds=10, 
                evals=[(matrix, "target_train"), (matrix_val, "target_val")],
                verbose_eval=False)

    def show_score(self):
        print(self.label + " Scores:")
        print("Train score:", self.compute_score(self.train_df))
        print("Val score:", self.compute_score(self.val_df))
        print("Test score:", self.compute_score(self.test_df))

    def predict_df(self, df):
        #X, _ = self.prepare_data(df)
        X_df, _, _ = self.prepare_data(df)
        return self.predict(X_df)

    def predict(self, X_df):
        m = xgb.DMatrix(X_df)
        return self.model.predict(m).reshape(-1,1)
    

def LinearRegressionTest(train_df, test_df, histograms=True):

    model = LinearRegressionModel(train_df, test_df)
    model.fit() # model.fit(ridge=False) 
    model.show_score()
    if histograms:
        model.error_histogram(model.train_df)
        model.error_histogram(model.test_df)

    return model

def XGBoostTest(train_df, test_df, histograms=True):

    model = XGBoostModel(train_df, test_df)
    model.fit()
    model.show_score()
    if histograms:
        model.error_histogram(model.train_df)
        model.error_histogram(model.val_df)
        model.error_histogram(model.test_df)

    return model

def single_location(rain=True, histograms=False):

    number_of_points = 3
    #number_of_points = 5
    HotEncodedColumns = ['Wind', 'Condition']
    repeatedColumns = ["Temperature",'Wind', 'Condition']
    time_deltas = [4, 8, 12, 24, 48] # 2h, 4h, 6h, 12h, 24h
    dfs = {}
    rain_text = ("-norain" if not rain else "")
    for time_delta in time_deltas:
        filename = f"data/Weather{rain_text}-n{number_of_points}-every{str(time_delta/2)}h-measureT"
        if exists(filename):
            print(f"The file {filename} already exists; reading database from file.")
            df = pd.read_csv(filename)
        else:
            timestamps = [time_delta*(i+1) for i in range(number_of_points)]
            df = process(timestamps, repeatedColumns, HotEncodedColumns, rain_present=rain)
            print(f"\n\nDataframe {time_delta/2} columns:\n")
            print(df.columns)
            df.to_csv(filename)
        dfs[time_delta] = df

    models = {}

    for time_delta, df in dfs.items():
        print("\n-------------------------------------------------------------")
        print(f"Results for dataframe with measurements every {time_delta/2}h\n")
        df = deleteUnwanted(df, ['Date', 'index', 'Time'])
        train_df, test_df = train_test_split(df, test_size = 0.14285714285, shuffle=False)
        models[time_delta] = {
            "linear":LinearRegressionTest(train_df, test_df, histograms=False),
            "xgboost":XGBoostTest(train_df, test_df, histograms=False),
            "neuralnet":NNTest(train_df, test_df, histograms=False)
        }
    
    return models

def extend_column_list(cols, suffixes):
    new_cols = []
    for col in cols:
        new_cols.append(col)
        for s in suffixes:
            new_cols.append(col+"_"+s)
    
    return new_cols


def multi_location(rain=True, histograms=False):
    number_of_points = 3
    #number_of_points = 5
    HotEncodedColumns = ['Wind', 'Condition']
    repeatedColumns = ["Temperature",'Wind', 'Condition']
    #repeatedColumns = ["Temperature",'Wind']
    time_deltas = [4, 8, 12, 24, 48] # 2h, 4h, 6h, 12h, 24h
    # time_deltas = [4]
    places = ["wroclaw", "poznan", "katowice", "prague", "dresden"]
    #places = ["wroclaw"]
    HotEncodedColumns = extend_column_list(HotEncodedColumns, places[1:])
    repeatedColumns = extend_column_list(repeatedColumns, places[1:])

    dfs = {}
    rain_text = ("-norain" if not rain else "")
    for time_delta in time_deltas:
        filename = f"data/WeatherJoint{rain_text}-n{number_of_points}-every{str(time_delta/2)}h-measureT"
        if exists(filename):
            print(f"The file {filename} already exists; reading database from file.")
            df = pd.read_csv(filename)
        else:
            print("making file:", filename)
            timestamps = [time_delta*(i+1) for i in range(number_of_points)]
            df = process(timestamps, repeatedColumns, HotEncodedColumns, place=None, places=places)
            print(f"\n\nDataframe {time_delta/2} columns:\n")
            print(df.columns)
            df.to_csv(filename)
        print("got database", filename)
        dfs[time_delta] = df
        
    models = {}

    for time_delta, df in dfs.items():
        print("\n-------------------------------------------------------------")
        print(f"Results for dataframe with measurements every {time_delta/2}h\n")
        df = deleteUnwanted(df, ['Date', 'index', 'Time'])
        train_df, test_df = train_test_split(df, test_size = 0.14285714285, shuffle=False)
        models[time_delta] = {
            "linear":LinearRegressionTest(train_df, test_df, histograms),
            "xgboost":XGBoostTest(train_df, test_df, histograms),
            "neuralnet":NNTest(train_df, test_df, histograms)
        }
    
    return models

if __name__ == '__main__':
    multi_location()
    #single_location()
    

"""
Results for measurments every 24h are low.
Probably because we have less measurements of 
the same time of the day as target

Best results for 4, 6 and 12 hours
"""
