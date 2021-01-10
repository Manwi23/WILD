import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

def cel(far):
    return (far - 32) * 5.0/9.0


dfs = []

vals = []

def process_df(df, label):

    rename_dict={}
    for c in df.columns:
        rename_dict[c] = c.split("|")[0]

    df = df.rename(columns=rename_dict)
    df["Temperature"] = df["Temperature"].apply(cel)
    print(np.mean(df["Temperature"]))
    print(df.groupby("Date")["Temperature"].mean())
    vals_for_year = list(map(cel, df.groupby("Date")["Temperature"].mean().values))
    plt.plot(vals_for_year, label=label)
    vals.append(vals_for_year[:89])
    return df

for y in [2014,2015,2016,2017,2018,2019,2020]:
    dfs.append(process_df(pd.read_csv(f"scraped/wroclaw{y}-01-01{y}-03-31.csv"), str(y)))

# plt.plot(np.mean(np.array(vals),axis=0), label="avg", linewidth=5.0)

# plt.legend()
# plt.show()

train_data = []
test_data = []

out = []

for df in dfs:
    rows = list(df.iterrows())
    length = len(rows)
    print(length)

    for i in range(length):
        if i + 48 >= length:
            continue

        _, prev = rows[i]
        _, curr = rows[i+24]
        _, next_ = rows[i+48]

        # print(prev["Time"],curr["Time"],next["Time"])
        # print(next_)
        target = next_["Temperature"] - prev["Temperature"]
        curr["TemperaturePrev"] = prev["Temperature"]
        curr["target"] = target
        out.append(curr)

result_df = pd.DataFrame(out).reset_index()
print(result_df.head())