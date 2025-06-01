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
import h5py
import scipy.io

#Data
def sample_plot(data):
    '''
    Input: (steps,dim)
    '''
    df = pd.DataFrame(data)
    df.plot(figsize=(15,5))
    plt.savefig("sample_plot.png")

def objective_logistic(X,y,w):
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))    
    z = X.dot(w)
    p = sigmoid(z)
    p = np.clip(p, 1e-10, 1 - 1e-10)
    loss = -np.sum(y * np.log(p) + (1 - y) * np.log(1 - p))
    return loss

def run(raw, verbose=False):
    if verbose == True:
        sample_plot(raw)
    #model parameters
    dim = 2
    order = 1

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
            xt = df[:,t,0]
            xtj = df[:,t,j]
            tmp_.append(np.kron(xt,xtj))   #dependence modeling
        return np.concatenate(tmp_).reshape((dim*dim*order,1))
        
    def func_h_orig(df, dim, order):
        h_all = np.zeros((dim*dim*order,1))
        for t in range(df.shape[1]):
            h_all += func_h_t(df,t,dim,order)
        return h_all

    def func_h_var(df, dim, order): #df : (dim,T,order+1)
        T = df.shape[1]
        h_all = np.zeros((dim * dim * order, 1))
        for j in range(1, order + 1):
            xt = df[:, :, -1].T #(T,order+1)
            xtj = df[:, :, -1-j].T #(T,order+1)
            h_j = xt.T @ xtj  # shape: (dim, dim) # 各時点のベクトル外積の総和：einsumでテンソル生成せずに行列積にする
            h_all[(j - 1) * dim * dim : j * dim * dim, 0] = h_j.ravel() 
        return h_all
    
    K = 8
    def func_h_custom(df, dim, order): #df : (dim,T,order+1)
        h_all = np.zeros((K, 1))
        for j in range(1, order + 1):
            xt = df[:, :, -1].T #(T,order+1)
            xtj = df[:, :, -1-j].T #(T,order+1)
            
            # h_j = xt.T @ xtj
            h_j = xt.T @ np.concatenate([xtj,xtj*xtj],axis=1)
            # h_j = np.concatenate([xt,xt*xt],axis=1).T @xtj
            # h_j = np.concatenate([xt,xt*xt],axis=1).T @ np.concatenate([xtj,xtj*xtj],axis=1)

            h_all[(j - 1) * h_j.size : j * h_j.size, 0]= h_j.ravel() 
            assert order*h_j.size == K    
        return h_all
    
    def func_h(df, dim, order):
        return func_h_custom(df,dim,order)
    
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
        from besag import LogisticRegression
        clf = LogisticRegression(eta=0.1,n_iter=10000)
        clf.fit(X, y)
        clf.eta, clf.n_iter = 0.01, 10000
        clf.fit_add(X, y, True)
        clf.eta, clf.n_iter = 0.001, 10000
        clf.fit_add(X, y, True)
        
        log_likelihood = -clf._log_loss(X, y) #log_loss is negative log likelihood
        w = clf.w
        # from sklearn.linear_model import LogisticRegression
        # y[0] = 0
        # clf = LogisticRegression(penalty=None, dual=False, tol=1e-4, C=1.0, fit_intercept=False, intercept_scaling=1, class_weight=None, random_state=None, solver='lbfgs', max_iter=100000, multi_class='deprecated', verbose=0, warm_start=False, n_jobs=None, l1_ratio=None)
        # clf.fit(X, y)
        # w = clf.coef_.T
        # log_likelihood = -objective_logistic(X,y,w)
        end_fit = time()
        print(f"Optimization took {end_fit-start_fit} seconds.")
        return w, log_likelihood, end_fit-start_fit

    def besag_PMLE_online_SGD(df, raw, eta=0.01, n_iter=10000):
        from random import sample
        n = len(raw)
        base_h = func_h(df, dim, order)
        w = np.zeros(dim * dim * order)
        start_fit = time()
        for it in range(n_iter):
            s = sample(list(range(order+1, n-order+1)),1)[0]
            t = s
            while t==s:
                t = sample(list(range(order+1, n-order+1)),1)[0]
            assert s != t
            raw_tmp = copy.deepcopy(raw)
            raw_tmp[s-1], raw_tmp[t-1] = raw_tmp[t-1].copy(), raw_tmp[s-1].copy()
            df_tmp = raw_to_dfs(raw_tmp)
            x_ = base_h - func_h(df_tmp, dim, order)
            x = x_.reshape(-1)

            # y = 1 always
            pred = 1 / (1 + np.exp(-np.dot(w, x)))
            error = 1 - pred
            step = eta * error
            w += step * x  # SGD step

            if it % 1000 == 0:
                print(f"Iter {it}, pred: {pred:.4f}")
        end_fit = time()
        return w, None, end_fit-start_fit
    
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


def h5_to_dict(obj):
    result = {}
    for key in obj.keys():
        item = obj[key]
        if isinstance(item, h5py.Group):
            result[key] = h5_to_dict(item)  # 再帰的に辞書化
        elif isinstance(item, h5py.Dataset):
            result[key] = item[()]  # NumPy array として読み込み
    return result

if __name__ == "__main__":
    
    prepare = False
    if prepare:
        v4 = scipy.io.loadmat('/home/sukeda/min-info-markov-new/data/V4 Utah Array Plaid Movie Data/Wi170428_spikes.mat')
        channels = v4["ex"]["CHANNELS"][0, 0]
        events = v4["ex"]["EVENTS"][0,0] #{33×1000×126 cell} % unit # X condition # X repeat).
        orilist = v4["ex"]["ORILIST"][0,0]
        repeats = v4["ex"]["REPEATS"][0,0] #(1, 1000)
        sc = v4["ex"]["SC"][0, 0]  # shape: (33, 1)
        movidx = v4["ex"]["MOVIDX"][0,0] #(1, 1000) # ex.MOVIDX lists the indices which correspond to ex.ORILIST
        movori = v4["ex"]["MOVORI"][0,0] #(1, 1000) #ex.MOVORI actually lists the two orientations (or blank) that were shown
        unit, condition, repeat = 30, 0, 0 #unit: 0~32, conditionは0と1だとrepeatは124,126まで埋まっているが, 2以降はrepeatは1しか埋まっていない.
        spike_times = events[unit, condition, repeat] #(number, 1)
        channel = channels[unit,0]
        print("Channel:",channel)
        
        with h5py.File('/home/sukeda/min-info-markov-new/data/V4 Utah Array Plaid Movie Data/Wi170428.mat', 'r') as f:
            v4 = h5_to_dict(f)
            lfp = f[v4["ex"]["LFP"][repeat,condition,channel-1]][::4,:]
            nstime = f[v4["ex"]["NSTIME"][repeat,condition]][::4,:]
            
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        lfp = scaler.fit_transform(lfp)

        # NSTIMEに対応するバイナリスパイク列を作成（0/1）
        spike_binary = np.zeros_like(nstime, dtype=int)
        # スパイク時刻を、nstime上で最も近いインデックスにマッピング
        for spk_time in spike_times:
            idx = np.argmin(np.abs(nstime - spk_time))
            spike_binary[idx] = 1

        # ---------- プロット ----------
        fig, axs = plt.subplots(2, 1, figsize=(10, 4), sharex=True)

        # 上段：スパイク列
        axs[0].plot(nstime, spike_binary, drawstyle='steps-post')
        axs[0].set_ylabel('Spike')
        axs[0].set_title('Spike Train')
        axs[0].set_yticks([0, 1])

        # 下段：LFP
        axs[1].plot(nstime, lfp)
        axs[1].set_ylabel('LFP (μV)')
        axs[1].set_xlabel('Relative time (s)')
        axs[1].set_title('LFP Signal')

        plt.tight_layout()
        plt.show()
        plt.savefig("parallel.png")

        data = np.concatenate([spike_binary,lfp], axis=1)
        pd.DataFrame(data,columns=["Spike","LFP"]).to_csv("parallel.csv",index=False)
    
    data = pd.read_csv("parallel.csv").values
    print(data.shape)
    theta_hat, log_likelihood, comp_time = run(data)
    
    print("Log likelihood", log_likelihood)
    # print("AIC=", -2*log_likelihood + K * 2)
    # print("PIC=", -2*log_likelihood + K * log(len(data)))

    # df = []
    # # for channel in tqdm(range(96)):
    # for channel in [0]:
    #     lfp = lfp_all.iloc[:,[channel]].to_numpy()
    #     #If standard scaling...
    #     from sklearn.preprocessing import StandardScaler
    #     scaler = StandardScaler()
    #     lfp = scaler.fit_transform(lfp)
    #     #If minmax scaling ...
    #     # lfp = (lfp - lfp.min(axis=0)) / (lfp.max(axis=0) - lfp.min(axis=0))

    #     theta_hat, log_likelihood, comp_time = run(lfp)
    #     K = 3
    #     print("Log likelihood", log_likelihood)
    #     print("AIC=", -2*log_likelihood + K * 2)
    #     print("PIC=", -2*log_likelihood + K * log(len(lfp)))
    #     df.append(list(theta_hat.T))