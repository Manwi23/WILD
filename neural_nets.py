import torch
import torch.nn as nn
from torch.optim import Adam

from DataProcessing import process, deleteUnwanted
from os.path import exists
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from trees import calc_score

from copy import deepcopy


def NNTest(train_df, test_df):
    torch.set_num_threads(8)
    train_df, val_df = train_test_split(train_df, test_size = 0.16666666666, shuffle=False)

    target_train = train_df['target']
    train_df = train_df.drop(columns=['target'])
    target_val = val_df['target']
    val_df = val_df.drop(columns=['target'])
    target_test = test_df['target']
    test_df = test_df.drop(columns=['target'])

    # print(train_df.columns)
    # print(len(train_df.columns))

    in_size = len(train_df.columns)
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
    filename = f"data/Weather-n{3}-every{6.0}h-measureT"
    if exists(filename):
        print(f"The file {filename} already exists; reading database from file.")
        df = pd.read_csv(filename)
    
    df = deleteUnwanted(df, ['Date', 'index', 'Time'])
    train_df, test_df = train_test_split(df, test_size = 0.14285714285, shuffle=False)
    NNTest(train_df, test_df)
