import pandas as pd

processed_df = pd.read_csv("processed.csv")
numerical_df = pd.get_dummies(processed_df, columns = ['Wind', 'Condition'])
numerical_df.to_csv("1HotEncoded.csv")