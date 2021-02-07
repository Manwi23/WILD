import torch
import torch.nn as nn
from torch.optim import Adam

from DataProcessing import process, deleteUnwanted
from os.path import exists
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from trees import calc_score
from weathermodel import Model
from copy import deepcopy


class NNModel(Model):
    def __init__(self, train_df, test_df, label = ""):
        super().__init__(train_df, test_df, label)
        self.label = "NeuralNet " + self.label
        self.train_df, self.val_df = train_test_split(self.train_df, test_size = 0.16666666666, shuffle=False)
        
    def prepare_data(self, df):
        target = df['target']
        X_df = df.drop(columns=['target'])
        X = torch.tensor(X_df.to_numpy(), dtype=torch.float32)
        Y = torch.tensor(target.to_numpy(), dtype=torch.float32).unsqueeze(-1)
        return X, Y, target.to_numpy()

    def predict_df(self, df):
        with torch.no_grad():
            train_X, _, _ = self.prepare_data(df)
            return self.model(train_X).numpy()

    def show_score(self):
        print(self.label + " Scores:")
        print("Train score:", self.compute_score(self.train_df))
        print("Val score:", self.compute_score(self.val_df))
        print("Test score:", self.compute_score(self.test_df))

    def fit(self):
        torch.set_num_threads(8)
        in_size = len(self.train_df.columns) - 1
        hidden1_size = 80
        hidden2_size = 40
        out_size = 1

        model = nn.Sequential(
            nn.Linear(in_size, hidden1_size), nn.ReLU(),
            nn.Dropout(),
            nn.Linear(hidden1_size, hidden2_size), nn.ReLU(),
            nn.Dropout(),
            nn.Linear(hidden2_size, out_size))

        opt = Adam(model.parameters())
        opt.zero_grad()

        train_X, train_Y, _ = self.prepare_data(self.train_df)
        val_X, _, val_Y = self.prepare_data(self.val_df)
        best = deepcopy(model)
        best_val = 0
        iter_without_progress = 40
        no_progress = 0

        for i in range(100000):
            model.train()
            results = model(train_X)
            d = results - train_Y
            loss = (d*d).sum() / train_X.shape[0]

            # print(loss.item())
            loss.backward()
            opt.step()
            opt.zero_grad()
            
            with torch.no_grad():
                model.eval()
                val_res = model(val_X).numpy()
                score = calc_score(val_Y, val_res, val_Y.mean())
                if score > best_val:
                    best_val = score
                    best = deepcopy(model)
                    no_progress = 0
                    # print(score)
                else:
                    no_progress += 1
                
            if iter_without_progress <= no_progress:
                break

        self.model = best


def NNTest(train_df, test_df):

    model = NNModel(train_df, test_df)
    model.fit()
    model.show_score()
    model.error_histogram(model.train_df)
    model.error_histogram(model.val_df)
    model.error_histogram(model.test_df)

def NNTest2(train_df, test_df):
    
    train_df, val_df = train_test_split(train_df, test_size = 0.16666666666, shuffle=False)

    target_train = train_df['target']
    train_df = train_df.drop(columns=['target'])
    target_val = val_df['target']
    val_df = val_df.drop(columns=['target'])
    target_test = test_df['target']
    test_df = test_df.drop(columns=['target'])

    # print(train_df.columns)
    # print(len(train_df.columns))

    train_X = torch.tensor(train_df.to_numpy(), dtype=torch.float32)
    train_Y = torch.tensor(target_train.to_numpy(), dtype=torch.float32).unsqueeze(-1)

    val_X = torch.tensor(val_df.to_numpy(), dtype=torch.float32)
    val_Y = target_val.to_numpy().reshape(-1, 1)

    test_X = torch.tensor(test_df.to_numpy(), dtype=torch.float32)
    test_Y = target_test.to_numpy().reshape(-1, 1)
    # print(val_X.shape)
    # print(val_Y.shape)

    best = deepcopy(model)
    best_val = 0
    iter_without_progress = 40
    no_progress = 0

    for i in range(100000):
        model.train()
        results = model(train_X)
        d = results - train_Y
        loss = (d*d).sum() / train_X.shape[0]

        # print(loss.item())
        loss.backward()
        opt.step()
        opt.zero_grad()
        
        with torch.no_grad():
            model.eval()
            val_res = model(val_X).numpy()
            score = calc_score(val_Y, val_res, val_Y.mean())
            if score > best_val:
                best_val = score
                best = deepcopy(model)
                no_progress = 0
                # print(score)
            else:
                no_progress += 1
            
        if iter_without_progress <= no_progress:
            break
    

    

    # with torch.no_grad():
    #     test_res = model(test_X).numpy()
    #     score = calc_score(test_Y, test_res, test_Y.mean())
    #     print(score)


    # xgb_model.fit(train_df, target_train, early_stopping_rounds=10, eval_set=[(train_df, target_train), (val_df, target_val)])
    with torch.no_grad():
        print("====NeuralNet====")
        pred = best(train_X).numpy()
        mm = target_train.mean()
        print("Train score:", calc_score(target_train, pred, mm))

        pred = best(val_X).numpy()
        mm = target_val.mean()
        print("Val score:", calc_score(target_val, pred, mm))

        pred = best(test_X).numpy()
        mm = target_test.mean()
        print("Test score:", calc_score(target_test, pred, mm))


if __name__ == "__main__":
    filename = f"data/WeatherJoint-n{3}-every{6.0}h-measureT"
    if exists(filename):
        print(f"The file {filename} already exists; reading database from file.")
        df = pd.read_csv(filename)
    
    df = deleteUnwanted(df, ['Date', 'index', 'Time'])
    train_df, test_df = train_test_split(df, test_size = 0.14285714285, shuffle=False)
    NNTest(train_df, test_df)
