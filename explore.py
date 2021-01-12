import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import scipy.stats
from os.path import exists

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

if exists("processed.csv"):
    result_df = pd.read_csv("processed.csv")
else:
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
    result_df.to_csv("processed.csv")


def low(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    return m-h

def high(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    return m+h

names = ["mean", "95%% low", "95%% high", "values count"]

def calculate_stats(column):
    gb = result_df.groupby(column)["target"]
    print(pd.concat([gb.mean().rename("mean"),
                     gb.apply(low).rename("95%% low"),
                     gb.apply(high).rename("95%% high"), 
                     gb.count().rename("count")], axis=1).sort_values("count",ascending=False))


print(result_df.head())
calculate_stats("Wind")
calculate_stats("Condition")
calculate_stats("Time")

cross = pd.crosstab(result_df["Wind"], result_df["Condition"])

s = cross.sum()
c_reduced = cross[s.sort_values(ascending=False).index[:8]]

print(c_reduced)


cross2 = pd.crosstab(result_df["Wind"], result_df["Condition"],values=result_df["target"], aggfunc="mean")
# cross2 = pd.crosstab(result_df["Wind"], result_df["Condition"],values=result_df["target"], aggfunc=(lambda x: (low(x), high(x))))
cross2 = cross2[s.sort_values(ascending=False).index[:8]]
print(cross2)