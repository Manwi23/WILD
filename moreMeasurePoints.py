from datetime import date, datetime, timedelta
from os.path import exists

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import train_test_split

from DataProcessing import deleteUnwanted, process
from neural_nets import NNModel, NNTest
from trees import calc_score
from weathermodel import Model
from get_data import scrape_date_range


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

    def show_score(self):
        print(self.label + "scores:")
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
        print(self.label + "scores:")
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
    

def LinearRegressionTest(train_df, test_df, histograms=True, scores=True):

    model = LinearRegressionModel(train_df, test_df)
    model.fit() # model.fit(ridge=False) 
    if scores:
        model.show_score()
    if histograms:
        model.error_histogram(model.train_df)
        model.error_histogram(model.test_df)

    return model

def XGBoostTest(train_df, test_df, histograms=True, scores=True):

    model = XGBoostModel(train_df, test_df)
    model.fit()
    if scores:
        model.show_score()
    if histograms:
        model.error_histogram(model.train_df)
        model.error_histogram(model.val_df)
        model.error_histogram(model.test_df)

    return model

def current_weather(models, multiple_locations=[]):
    def get_good_date(dt):
        m = str(dt.month)
        if len(m) == 1: m = '0'+m
        d = str(dt.day)
        if len(d) == 1: d = '0'+d
        return m + '-' + d

    def fill_columns(train, test):
        d = pd.DataFrame(0, index=np.arange(len(test)), columns=train.columns)
        for col in test.columns:
            if col in train.columns:
                d[col] = test[col]
        return d

    today = date.today()
    past = today - timedelta(days=5)
    future = today + timedelta(days=1)

    scrape_date_range('wroclaw', start=date(past.year,past.month,past.day),
                        end=date(future.year,future.month,future.day),
                        chromedriver='./chromedriver')

    for m in multiple_locations:
        if m != 'wroclaw':
            scrape_date_range(m, start=date(past.year,past.month,past.day),
                            end=date(future.year,future.month,future.day),
                            chromedriver='./chromedriver')

    number_of_points = 3
    HotEncodedColumns = ['Wind', 'Condition']
    if multiple_locations:
        HotEncodedColumns = extend_column_list(HotEncodedColumns, multiple_locations[1:])
    repeatedColumns = ["Temperature",'Wind', 'Condition']
    if multiple_locations:
        repeatedColumns = extend_column_list(repeatedColumns, multiple_locations[1:])
    time_deltas = [4, 8, 12, 24, 48] # 2h, 4h, 6h, 12h, 24h
    # time_deltas = [4]
    dfs = {}

    for time_delta in time_deltas:
        timestamps = [time_delta*(i+1) for i in range(number_of_points)]
        df = process(timestamps, repeatedColumns, HotEncodedColumns,
                    [today.year], get_good_date(past), get_good_date(future), rain_present=False,
                    current_time=0) if not multiple_locations else \
                    process(timestamps, repeatedColumns, HotEncodedColumns,
                    [today.year], get_good_date(past), get_good_date(future), rain_present=False,
                    current_time=0, place=None, places=multiple_locations)
        df = deleteUnwanted(df, ['Date', 'index', 'Time'])
        # print(f"\n\nDataframe {time_delta/2} columns:\n")
        # print(df.columns)
        dfs[time_delta] = df

    all_predictions = {}
    
    for time_delta in time_deltas:
        models_for_timedelta = models[time_delta]
        data = dfs[time_delta]
        all_predictions[time_delta] = {}
        for m in models_for_timedelta:
            # print(m)
            model = models_for_timedelta[m]
            new_test = fill_columns(model.train_df, data)
            result = model.predict_df(new_test).reshape(-1)
            measurements = dfs[24]['TemperaturePrev12h']
            predicted = []
            for (res, mes) in zip(result, measurements):
                predicted += [res + mes]
            all_predictions[time_delta][m] = predicted

    return all_predictions, dfs[24]['Temperature']

def single_location(rain=True, histograms=False, scores=False):

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
            print(f"The file {filename} doesn't exists; preprocessing data.")
            timestamps = [time_delta*(i+1) for i in range(number_of_points)]
            df = process(timestamps, repeatedColumns, HotEncodedColumns, rain_present=rain)
            # print(f"\n\nDataframe {time_delta/2} columns:\n")
            # print(df.columns)
            df.to_csv(filename)
            print("Got database: ", filename)
        dfs[time_delta] = df

    models = {}

    for time_delta, df in dfs.items():
        if scores:
            print("\n-------------------------------------------------------------")
            print(f"Results for dataframe with measurements every {time_delta/2}h\n")
        df = deleteUnwanted(df, ['Date', 'index', 'Time'])
        train_df, test_df = train_test_split(df, test_size = 0.14285714285, shuffle=False)
        models[time_delta] = {
            "linear":LinearRegressionTest(train_df, test_df, histograms=histograms, scores=scores),
            "xgboost":XGBoostTest(train_df, test_df, histograms=histograms, scores=scores),
            "neuralnet":NNTest(train_df, test_df, histograms=histograms, scores=scores)
        }
    
    return models

def extend_column_list(cols, suffixes):
    new_cols = []
    for col in cols:
        new_cols.append(col)
        for s in suffixes:
            new_cols.append(col+"_"+s)
    
    return new_cols


def multi_location(rain=True, histograms=False, scores=False):
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
            print(f"The file {filename} doesn't exists; preprocessing data.")
            # print("making file:", filename)
            timestamps = [time_delta*(i+1) for i in range(number_of_points)]
            df = process(timestamps, repeatedColumns, HotEncodedColumns, place=None, places=places)
            # print(f"\n\nDataframe {time_delta/2} columns:\n")
            # print(df.columns)
            df.to_csv(filename)
        print("Got database: ", filename)
        dfs[time_delta] = df
        
    models = {}

    for time_delta, df in dfs.items():
        if scores:
            print("\n-------------------------------------------------------------")
            print(f"Results for dataframe with measurements every {time_delta/2}h\n")
        df = deleteUnwanted(df, ['Date', 'index', 'Time'])
        train_df, test_df = train_test_split(df, test_size = 0.14285714285, shuffle=False)
        models[time_delta] = {
            "linear":LinearRegressionTest(train_df, test_df, histograms, scores=scores),
            "xgboost":XGBoostTest(train_df, test_df, histograms, scores=scores),
            "neuralnet":NNTest(train_df, test_df, histograms, scores=scores)
        }
    
    return models

if __name__ == '__main__':
    # multi_location()
    # single_location()
    # models = single_location(rain=False)
    models = multi_location(rain=False)
    ans, _ = current_weather(models, multiple_locations=["wroclaw", "poznan", "katowice", "prague", "dresden"])
    for td in ans:
        for m in ans[td]:
            print(td, m)
            print(ans[td][m][-10:])
    

"""
Results for measurments every 24h are low.
Probably because we have less measurements of 
the same time of the day as target

Best results for 4, 6 and 12 hours
"""
