import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

numerical_df = pd.read_csv("1HotEncoded.csv")

train_df, test_df = train_test_split(numerical_df, test_size = 0.14285714285, shuffle=False)

features = list(numerical_df.columns)
features.remove('index')
features.remove('Unnamed: 0')
features.remove('Unnamed: 0.1')
features.remove('Date')
features.remove('Time')
features.remove('target')
features.remove('Temperature')
#features.remove('TemperaturePrev')

features = list(filter(lambda x: "Condition_" in x, features))
print(features)

Y_train = np.array(train_df['target'])
Y_train = Y_train.reshape(-1, 1)

X_train = np.array(train_df[features])
# X_train = np.concatenate([np.array(train_df[features]), np.ones(Y_train.shape)], axis=1)

print(X_train.sum(axis=0))
print(X_train)

Y_test = np.array(test_df['target'])
print(Y_test)
Y_test = Y_test.reshape(-1, 1)

X_test = np.array(test_df[features])
# X_test = np.concatenate([np.array(test_df[features]), np.ones(Y_test.shape)], axis=1)

reg = LinearRegression().fit(X_train, Y_train)
print(reg.coef_, reg.intercept_, reg.coef_+reg.intercept_)
print("Train Score:", reg.score(X_train, Y_train))
print("Test Score:", reg.score(X_test, Y_test))



print(reg.predict([X_test[0]]))