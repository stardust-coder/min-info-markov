import numpy as np
import matplotlib.pyplot as plt
import statsmodels.graphics.api as smg
from statsmodels.graphics.tsaplots import plot_acf
from time import time
from tqdm import tqdm
import pandas as pd
import random
import pdb
from itertools import combinations
import copy
from math import log

#Data
def sample_plot(data):
    '''
    Input: (steps,dim)
    '''
    df = pd.DataFrame(data)
    df.plot(figsize=(15,5))
    plt.savefig("sample_plot.png")

def run(raw, verbose=False):
    if verbose == True:
        sample_plot(raw)
    #model parameters
    dim = 1
    order = 2

    def raw_to_dfs(rawdata):
        dfs = []
        for j in range(dim):
            df = []
            for t in range(order,len(rawdata)):
                df.append(rawdata[t-order:t+1,j])
            dfs.append(df)        
        dfs = np.array(dfs)
        return dfs # (dim, len(raw)-order,order+1)


    def func_h_t(df,t,dim,order):
        tmp_ = []
        for j in range(1,order+1):
            xt = df[:,t,-1]
            xtj = df[:,t,-1-j]
            tmp_.append(np.kron(xt,xtj))   #dependence modeling
        return np.concatenate(tmp_).reshape((dim*dim*order,1))
        
    def func_h_orig(df, dim, order):
        h_all = np.zeros((dim*dim*order,1))
        for t in range(df.shape[1]):
            h_all += func_h_t(df,t,dim,order)
        return h_all

    def func_h_v1(df, dim, order):
        T = df.shape[0]
        h_all = np.zeros((dim * dim * order, 1))
        for j in range(1, order + 1):
            xt = df[:, :, -1].T
            xtj = df[:, :, -1 - j].T
            h_j = xt.T @ xtj  # shape: (dim, dim) # 各時点のベクトル外積の総和：einsumでテンソル生成せずに行列積にする
            h_all[(j - 1) * dim * dim : j * dim * dim, 0] = h_j.ravel() 
            import pdb; pdb.set_trace()
        return h_all
    
    def func_h_v2(df, dim, order):
        T = df.shape[0]
        h_all = np.zeros((3, 1))
        xt = df[:, :, -1].T
        xtj = df[:, :, -1 - 1].T
        
        j = 1
        h_j = xt.T @ xtj  
        h_all[(j - 1) * dim * dim : j * dim * dim, 0] = h_j.ravel() 

        j = 2
        h_j = (xt*xt).T @ xtj  
        h_all[(j - 1) * dim * dim : j * dim * dim, 0] = h_j.ravel() 

        j = 3
        h_j = xt.T @ (xtj*xtj)
        h_all[(j - 1) * dim * dim : j * dim * dim, 0] = h_j.ravel() 
        return h_all
    
    K = 3
    def func_h_v3(df, dim, order):
        T = df.shape[0]
        h_all = np.zeros((K, 1))
        xt = df[:, :, -1].T
        xt1 = df[:, :, -1 - 1].T
        xt2 = df[:, :, -1 - 2].T

        j = 1
        h_j = np.sum(xt*xt1)
        h_all[(j - 1) * dim * dim : j * dim * dim, 0] = h_j.ravel() 

        j = 2
        h_j = np.sum(xt*xt2)
        h_all[(j - 1) * dim * dim : j * dim * dim, 0] = h_j.ravel() 

        j = 3
        h_j = np.sum(xt*xt1*xt2)
        h_all[(j - 1) * dim * dim : j * dim * dim, 0] = h_j.ravel() 

        return h_all

    def func_h_v4(df, dim, order):
        T = df.shape[0]
        h_all = np.zeros((dim * dim * order, 1))
        for j in range(1, order + 1):
            xt = df[:, :, -1].T
            xtj = df[:, :, -1 - j].T
            h_j = xt.T @ xtj  # shape: (dim, dim) # 各時点のベクトル外積の総和：einsumでテンソル生成せずに行列積にする
            h_all[(j - 1) * dim * dim : j * dim * dim, 0] = h_j.ravel() 
        return h_all


    from besag import LogisticRegression
    def func_h(df, dim, order):
        return func_h_v2(df,dim,order)
    
    def besag_PMLE(df,raw):
        n = len(raw)
        X = np.zeros((int((n-2*order)*(n-2*order-1)/2),K))
        base_h = func_h(df, dim, order)  # ← 固定値として1回だけ呼ぶ

        for i, (s, t) in enumerate(tqdm(combinations(range(order+1,n-order+1),2))):
            # print(i,"/", int(n*(n-1)/2))
            raw_tmp = copy.deepcopy(raw)
            raw_tmp[s-1],raw_tmp[t-1] = raw_tmp[t-1].copy(),raw_tmp[s-1].copy()
            df_tmp = raw_to_dfs(raw_tmp)
            x_ = base_h-func_h(df_tmp,dim,order) # time bottleneck   
            X[i] = x_.reshape(K,)

        y = np.ones(int((n-2*order)*(n-2*order-1)/2))
        print("Start Fitting ...")
        start_fit = time()
        clf = LogisticRegression(eta=1,n_iter=500)
        clf.fit(X, y)
        log_likelihood = -clf._log_loss(X, y) #log_loss is negative log likelihood
        end_fit = time()
        print(f"Optimization took {end_fit-start_fit} seconds.")
        return clf.w, log_likelihood, end_fit-start_fit

    ### Besag's PMLE for any model
    df = raw_to_dfs(raw)
    start_time = time()
    theta_hat, log_likelihood, optimization_time = besag_PMLE(df=df,raw=raw)
    end_time = time()

    #Result
    print("--- 推定するパラメタ数 --- ")
    print(K)
    print("--- 推定値 --- ")
    print(theta_hat.T)
    print("--- 所要時間 うち 勾配法 ---")
    comp_time = (end_time-start_time, optimization_time)
    print(comp_time)
    return theta_hat, log_likelihood, comp_time


def raster_plot(events, repeats, unit_index, condition_index):  
    num_trials = repeats[0][condition] #max: 126
    print("num_trials=",num_trials)
    fig, ax = plt.subplots(figsize=(10, 6))
    for trial_index in range(num_trials):
        spike_times = events[unit_index, condition_index, trial_index]
        if spike_times is not None and len(spike_times) > 0:
            ax.vlines(spike_times.flatten(), trial_index + 0.5, trial_index + 1.5, color='black')
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Trial")
    ax.set_title(f"Spike Train (Unit {unit_index}, Condition {condition_index})")
    ax.set_ylim(0.5, num_trials + 0.5)
    ax.invert_yaxis()
    plt.tight_layout()
    plt.show()
    plt.savefig(f"raster_unit{unit_index}_condition{condition_index}.png")

def unitwise_plot(events, condition_index):
    num_unit = 33
    trial_index = 0 #fix
    fig, ax = plt.subplots(figsize=(10, 6))
    for unit_index in range(num_unit):
        spike_times = events[unit_index, condition_index, trial_index]
        if spike_times is not None and len(spike_times) > 0:
            ax.vlines(spike_times.flatten(), unit_index + 0.5, unit_index + 1.5, color='black')
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Unit Index")
    ax.set_title(f"Spike Train (Condition {condition_index})")
    ax.set_ylim(0.5, num_unit + 0.5)
    ax.invert_yaxis()
    plt.tight_layout()
    plt.show()
    plt.savefig(f"unitwise_condition{condition_index}.png")

def convert2historgram(data, save_fig=False):
    # ビン境界の定義（たとえば -1 〜 2秒を30等分）
    bins = np.linspace(-1, 2, 101)
    # ヒストグラム計算
    counts, bin_edges = np.histogram(data, bins=bins)
    # 結果表示
    print("Bin edges:", bin_edges)
    print("Counts per bin:", counts)
    # ビンの中心を計算（棒グラフのx軸）
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    if save_fig:
        # プロット
        plt.figure(figsize=(8, 4))
        plt.bar(bin_centers, counts, width=np.diff(bin_edges)[0], edgecolor='black', align='center')
        plt.xlabel("Time (s)")
        plt.ylabel("Spike Count")
        plt.title("Spike Count per Time Bin")
        plt.tight_layout()
        plt.show()
        plt.savefig("count_data_histogram.png")
    return counts


def get_LFP():
    import h5py 
    with h5py.File('/home/sukeda/min-info-markov-new/data/V4 Utah Array Plaid Movie Data/Pe170417.mat', 'r') as f:
        ref_array = f["ex"]["LFP"][()]
        def resolve_deep_lfp(ref_array, h5file):
            trials, timepoints, _ = ref_array.shape
            channels = len(ref_array[0, 0])  # 96
            resolved = np.zeros((trials, timepoints, channels))  # ← float にしておく

            for i in range(trials):
                for j in range(timepoints):
                    refs = ref_array[i, j]
                    for k in range(channels):
                        try:
                            ref = refs[k]
                            if isinstance(ref, h5py.Reference) and ref:
                                val = h5file[ref][()]
                                resolved[i, j, k] = val if np.isscalar(val) else val[0]
                            else:
                                resolved[i, j, k] = np.nan  # 無効参照なら NaN
                        except Exception as e:
                            print(f"⚠️ LFP[{i},{j},{k}] 解決失敗: {e}")
                            resolved[i, j, k] = np.nan
            return resolved
        resolved_lfp = resolve_deep_lfp(ref_array, f)
    return resolved_lfp


if __name__ == "__main__":
    lfp_all = pd.read_csv("./data/V4 Utah Array Plaid Movie Data/Wi170428_LFP.csv")

    df = []
    for channel in tqdm(range(96)):
    # for channel in [0]:
        lfp = lfp_all.iloc[:,[channel]].to_numpy()
        #If standard scaling...
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        lfp = scaler.fit_transform(lfp)
        #If minmax scaling ...
        # lfp = (lfp - lfp.min(axis=0)) / (lfp.max(axis=0) - lfp.min(axis=0))

        theta_hat, log_likelihood, comp_time = run(lfp)
        K = 3
        print("Log likelihood", log_likelihood)
        print("AIC=", -2*log_likelihood + K * 2)
        print("PIC=", -2*log_likelihood + K * log(len(lfp)))
        df.append(list(theta_hat.T))
    import pdb; pdb.set_trace()