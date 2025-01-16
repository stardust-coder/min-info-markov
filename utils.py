import numpy as np

def raw_to_df(rawdata,d):
    df = []
    for t in range(d,len(rawdata)):
        df.append(rawdata[t-d:t+1])
    df = np.array(df) #shape:[T-d,d+1]
    return df

def raw_to_dfs(rawdata,d):
    dfs = []
    for j in range(d):
        df = []
        for t in range(d,len(rawdata)):
            df.append(rawdata[t-d:t+1,j])
        dfs.append(df)        
    dfs = np.array(dfs)
    return dfs