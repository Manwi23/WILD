import pandas as pd
from os.path import exists
from explore import process_df
from datetime import timedelta, date

def deleteUnwanted(df, toDrop=[]):
    columns = list(df.columns)
    toDrop += list(filter(lambda t: t.startswith(("Unnamed")), columns))
    df = df.drop(columns=toDrop)
    if "" in columns:
        df = df.drop(columns=[""])
    return df

def readData(years, date_start, date_end, place):
    dfs = []
    for y in years:
        dfs.append(process_df(pd.read_csv(f"scraped/{place}{y}-{date_start}{y}-{date_end}.csv"), str(y)))
    return dfs

def readDataAndJoin(years, date_start, date_end, places):
    dfs = []
    for y in years:
        df = None
        prev_p = ""
        for p in places:
            v = process_df(pd.read_csv(f"scraped/{p}{y}-{date_start}{y}-{date_end}.csv"), str(y))
            if df is None:
                df = v
            else:
                df = df.merge(v, how='inner', on=["Date", "Time"], suffixes=(None, "_"+p))
        dfs.append(df)
    return dfs


# Based on Michał's code
"""
timestamps - list of times of measurements before current time
current_time - time before target
repeatedColumns - columns to be repeated in previous times of measurements
target is a difference of temperatures between target_time and target_time + target_delta
"""
def addTimePoints(dfs, timestamps, repeatedColumns, target_delta=48, current_time = 24):
    if timestamps == []:
        max_time_dist = max(target_delta, current_time)
    else:
        max_timestamp = max(timestamps)
        max_time_dist = max(target_delta, (max_timestamp + current_time))
    
    out = []
    for df in dfs:
        rows = list(df.iterrows())
        length = len(rows)
        for target_time in range(max_time_dist,length):

            ## Compute target:
            _, target_row_end = rows[target_time]
            _, target_row_start = rows[target_time-target_delta]
            target = target_row_end["Temperature"] - target_row_start["Temperature"]

            ## Get current weather conditions
            _, curr = rows[target_time - current_time]
            curr["target"] = target

            # Add previous time measurements
            for time in timestamps:
                _, prev = rows[target_time - current_time - time]
                for col in repeatedColumns:
                    col_name = col+"Prev"+str(time//2)+"h"
                    if time%2:
                        col_name += "30m"
                    curr[col_name] = prev[col]
            # add row to data
            out.append(curr)
    result_df = pd.DataFrame(out).reset_index()
    return result_df


def hotEncode(df, HotEncodedColumns):
    columns = list(df.columns)
    print("All Columns:", columns)
    toEncode = HotEncodedColumns + list(filter(lambda t: t.startswith(tuple([s + 'Prev' for s in HotEncodedColumns])), columns))
    encoded_df = pd.get_dummies(df, columns = toEncode)
    return encoded_df

# Based on Kasia's code
def addRainData(processed, places):
    if places is None:
        places = ['wroclaw']
    for place in places:
        df = pd.read_csv("csvs/precip_{}.csv".format(place))
        rename_dict={"PS" : "Pressure", 
                    "RH2M" : "Relative Humidity",
                    "QV2M" : "Humidity",
                    "PRECTOT" : "Precipitation"}
        df = df.rename(columns=rename_dict)

        precip_dict = {}

        for _, row in df.iterrows():
            day = date(int(row["YEAR"]), int(row["MO"]), int(row["DY"]))
            prev_day = day + timedelta(days = -1)
            precip_dict[str(prev_day)] = row["Precipitation"]

        useful = []

        for _, row in processed.iterrows():
            day = row['Date']
            useful += [precip_dict[day]]

        if place != 'wroclaw':
            precip = pd.DataFrame({f'Precip._{place}': useful})

            processed[f'Precip._{place}'] = precip[f'Precip._{place}']
        else:
            precip = pd.DataFrame({'Precip.': useful})

            processed['Precip.'] = precip['Precip.']
    return processed



def process(timestamps, repeatedColumns, HotEncodedColumns,
            years =  [2014,2015,2016,2017,2018,2019,2020], 
            date_start="01-01", date_end='03-31', place='wroclaw',places=None):
    
    # Read scraped data
    print("Reading data...")
    if place is not None:
        dfs = readData(years = years, date_start=date_start, date_end=date_end, place=place)
    elif places is not None:
        dfs = readDataAndJoin(years = years, date_start=date_start, date_end=date_end, places=places)
    # Add previous points in time and target
    print("Adding previous measurements...")
    df_with_time_points = addTimePoints(dfs, timestamps, repeatedColumns)
    # Apply 1 hot encoding 
    print("Applying 1 Hot Encoding...")
    df_processed = hotEncode(df_with_time_points, HotEncodedColumns)
    # Add rain Data 
    print("Adding rain...")
    df_processed = addRainData(df_processed, places)
    return df_processed

if __name__ == '__main__':
    
    filename="csvs/ProcessedDF36h.csv"
    HotEncodedColumns = ['Wind', 'Condition']
    repeatedColumns = ["Temperature"]
    timestamps = [24, 48, 72]
    # Check if the file already exist
    if exists(filename):
        print(f"The file {filename} already exists; reading database from file.")
        result_df = pd.read_csv(filename)
    else:
        result_df = process(timestamps, repeatedColumns, HotEncodedColumns)
        result_df.to_csv(filename)
    
    print(result_df.head(5))
    