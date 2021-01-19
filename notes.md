## Notes on experiments

#### Day and night
Night is between 5 pm and 7 am, the split is done in the way the conditions we take into consideration take place in the night

1. XGBoost booster="gbtree",objective="reg:squarederror", n_estimators=8

* 1HotEncoded
    * train: 0.4530177592191743
    * test: 0.2764793977193325

* night
    * train: 0.4793779194018264
    * test: 0.18185038667072728

* day
    * train: 0.6211113992078736
    * test: 0.20086015126211987


2. XGBoost booster="gblinear",objective="reg:squarederror", n_estimators=8

* 1HotEncoded
    * train: 0.2490704065586914
    * test: 0.2641688128108173

* night
    * train: 0.22067633359242278
    * test: 0.2155086818877583

* day
    * train: 0.3502246044237699
    * test: 0.3684354899946424

3. XGBRegressor(booster="gbtree",objective="reg:squarederror")
xgb_model.fit(train_df, target_train, early_stopping_rounds=5, eval_set=[(train_df, target_train), (val_df, target_val)])

* 1HotEncoded
    * train: 0.5734636198667963
    * val: 0.33030948496898493
    * test: 0.24190311454487945

* night
    * train: 0.5835159987821079
    * val: 0.29409122136801735
    * test: 0.17317735068967965

* day
    * train: 0.7052744792920413
    * val: 0.34467925514467845
    * test: 0.3162153311684456

#### Maybe interesting facts:

* Looking at the pearson / chi2 stuff I noticed the condition is far more meaningful during the day than the night. Makes sense, at least a bit.