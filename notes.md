## Notes on experiments

#### Day and night
Night is between 5 pm and 7 am, the split is done in the way the conditions we take into consideration take place in the night (in night.csv)

1. XGBoost booster="gbtree",objective="reg:squarederror", n_estimators=8

1HotEncoded:
* train: 0.4530177592191743
* test: 0.2764793977193325

night:
* train: 0.4793779194018264
* test: 0.18185038667072728

day:
* train: 0.6211113992078736
* test: 0.20086015126211987


2. XGBoost booster="gblinear",objective="reg:squarederror", n_estimators=8

1HotEncoded:
* train: 0.2490704065586914
* test: 0.2641688128108173

night:
* train: 0.22067633359242278
* test: 0.2155086818877583

day:
* train: 0.3502246044237699
* test: 0.3684354899946424


#### Maybe interesting facts:

* Looking at the pearson / chi2 stuff I noticed the condition is far more meaningful during the day than the night. Makes sense, at least a bit.