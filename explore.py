import pandas as pd
import numpy as np
import scipy.stats
from os.path import exists
from sklearn.model_selection import train_test_split
import math
from datetime import datetime as dt
from datetime import timedelta

def cel(far):
    return (far - 32) * 5.0/9.0

def fix_time(s):
    ttime = dt.strptime(s, "%I:%M %p")

    if ":20" in s or ":50" in s:
        ttime = ttime + timedelta(minutes=10)

    return ttime.strftime('%I:%M %p')


def process_df(df, label):

    rename_dict={}
    for c in df.columns:
        rename_dict[c] = c.split("|")[0]

    vals = []

    df = df.rename(columns=rename_dict)
    df["Temperature"] = df["Temperature"].apply(cel)

    df["Time"] = df["Time"].apply(fix_time)

    return df

def read_or_process_data():
    if exists("processed.csv"):
        result_df = pd.read_csv("processed.csv")
    else:
        dfs = []
        for y in [2014,2015,2016,2017,2018,2019,2020]:
            dfs.append(process_df(pd.read_csv(f"scraped/wroclaw{y}-01-01{y}-03-31.csv"), str(y)))

        # plt.plot(np.mean(np.array(vals),axis=0), label="avg", linewidth=5.0)

        # plt.legend()
        # plt.show()

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
    return result_df

if __name__ == "__main__":
    result_df = read_or_process_data()
    result_df, test_df = train_test_split(result_df, test_size = 0.14285714285, shuffle=False)

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
        val = pd.concat([gb.mean().rename("mean"),
                        gb.apply(low).rename("95% low"),
                        gb.apply(high).rename("95% high"), 
                        gb.count().rename("count")], axis=1).sort_values("count",ascending=False)
        
        print(val)
        return val



    print(result_df.head())
    wind = calculate_stats("Wind")
    cond = calculate_stats("Condition")
    time = calculate_stats("Time")

    cross = pd.crosstab(result_df["Wind"], result_df["Condition"])

    s = cross.sum()
    c_reduced = cross[s.sort_values(ascending=False).index[:8]]

    print(c_reduced)


    cross2 = pd.crosstab(result_df["Wind"], result_df["Condition"],values=result_df["target"], aggfunc="mean")
    # cross2 = pd.crosstab(result_df["Wind"], result_df["Condition"],values=result_df["target"], aggfunc=(lambda x: (low(x), high(x))))
    cross2cut = cross2[s.sort_values(ascending=False).index[:8]]
    print(cross2cut)

    test_df_mean = test_df["target"].mean()
    # print(test_df_mean)

    def calc_score(func, df):
        v = 0
        d = 0
        for i, row in df.iterrows():
            val = (row["target"] - func(row))**2
            base = (row["target"] - test_df_mean)**2
            if math.isnan(val):
                v+=base
            else:
                v+=val
            d += base

        print(1 - v/d)

    def cond_calc(row):
        # print(row["Wind"])
        return cond.loc[row["Condition"], "mean"]

    def wind_calc(row):
        # print(row["Wind"])
        return wind.loc[row["Wind"], "mean"]

    def wind_cond_calc(row):
        # print(row["Wind"])
        return cross2.loc[row["Wind"], row["Condition"]]

    print("wind only")
    calc_score(wind_calc, result_df)
    calc_score(wind_calc, test_df)

    print("wind and condition cross-table")
    calc_score(wind_cond_calc, result_df)
    calc_score(wind_cond_calc, test_df)

    print("condition only")
    calc_score(cond_calc, result_df)
    calc_score(cond_calc, test_df)
