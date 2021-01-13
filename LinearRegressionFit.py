import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

numerical_df = pd.read_csv("1HotEncoded.csv")

train_df, test_df = train_test_split(numerical_df, test_size = 0.1)

features = list(numerical_df.columns)
features.remove('Date')
features.remove('Time')
features.remove('target')

X_train = np.array(train_df[features])
Y_train = np.array(train_df['target'])
Y_train = Y_train.reshape(-1, 1)

X_test = np.array(test_df[features])
Y_test = np.array(test_df['target'])
Y_test = Y_test.reshape(-1, 1)

reg = LinearRegression().fit(X_train, Y_train)

print("Train Score:", reg.score(X_train, Y_train))
print("Test Score:", reg.score(X_test, Y_test))