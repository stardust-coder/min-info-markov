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

#Data
def sample_plot(data):
    '''
    Input: (steps,dim)
    '''
    df = pd.DataFrame(data)
    df.plot(figsize=(15,5))
    plt.savefig("sample_plot.png")

def load_ecog():
    file = "Ji20180308S1c_ECoG_raw.csv"
    raw = np.loadtxt(f"data/{file}", delimiter=',')
    raw = raw[:500,[0]] # first electrode
    min_ = np.min(raw)
    max_ = np.max(raw)
    raw = (raw-min_)/(max_-min_)
    return raw

def simulate_VAR(dim,order=1,steps=500):
    #Model parameters
    phi = [0.5*np.identity(dim) for _ in range(order)]
    # phi[0][0][0] = 0.5
    # phi[0][1][0] = 0.1
    # phi[0][0][1] = 0.1
    # phi[0][1][1] = 0.5
    # phi = [0.5*np.identity(dim),0.3*np.identity(dim)]
    # phi = [0.5*np.identity(dim),0.3*np.identity(dim),0.1*np.identity(dim)]

    for item in phi:
        assert item.shape == (dim,dim)
    assert len(phi) == order

    sigma = np.identity(dim)*0.5 ### noise variance
    assert sigma.shape == (dim,dim)

    mean = np.zeros((dim))
    assert mean.shape == (dim,)

    ### Stationarity check
    check_stationarity = False
    if check_stationarity:
        coeffs = [1] + [-phi_.item() for phi_ in phi]  # 特性方程式：1 - φ₁z - φ₂z² - φ₃z³ = 0
        coeffs.reverse()
        roots = np.roots(coeffs)
        is_stationary = np.all(np.abs(roots) > 1)
        print("Is stationary?", is_stationary)
    ### Data generation
    var_data = []
    for _ in range(order):
        var_data.append(np.zeros((1,dim))) # initial value
    for _ in range(order,steps+order):
        v = np.zeros((dim,1))
        for k in range(1,order+1):
            v += phi[k-1]@var_data[-k].T #A^k x_{t-k}            
        v += np.random.multivariate_normal(mean, sigma, 1).T
        var_data.append(v.T)
    var_data = np.array(var_data[order:])
    
    assert (dim,order) in [(1,1),(1,2),(1,3),(2,1)] #AR(1),AR(2),AR(3),VAR(1)
    Theta = [phi[k].T@np.linalg.inv(sigma) for k in range(order)]
    Theta = np.concatenate(Theta)

    print("True Parameter (estimation target):", Theta.flatten())
    return var_data[:,0,:], Theta.flatten()

def MLE(Y, order):
    from statsmodels.tsa.api import VAR, ARIMA
    if Y.shape[1] == 1:
        model = ARIMA(Y, order=(order, 0, 0), trend="n") #AR(d)
        results = model.fit()
    
    else:
        model = VAR(Y)
        results = model.fit(maxlags=order, ic=None, trend="n")
    print(results.summary())
    return results
    

def run():
    # raw = load_ecog()
    raw, true_parameter = simulate_VAR(dim=1,order=1,steps=10000)
    # sample_plot(raw)

    #model parameters
    dim = 1
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
            xt = df[:,t,-1]
            xtj = df[:,t,-1-j]
            tmp_.append(np.kron(xt,xtj))   #dependence modeling
        return np.concatenate(tmp_).reshape((dim*dim*order,1))
        
    def func_h(df, dim, order):
        h_all = np.zeros((dim*dim*order,1))
        for t in range(df.shape[1]):
            h_all += func_h_t(df,t,dim,order)
        return h_all

    def func_h_einsum(df, dim, order):
        xts = df[:, :, -1].T  # shape: (T, dim)
        h_all = np.zeros((dim * dim * order, 1))
        
        for j in range(1, order + 1):
            xtj = df[:, :, -1 - j].T  # shape: (T, dim)
            kron_all = np.einsum('ti,tj->tij', xts, xtj)  # shape: (T, dim, dim)
            summed = kron_all.sum(axis=0).reshape(-1, 1)  # shape: (dim*dim, 1)
            h_all[(j - 1) * dim * dim : j * dim * dim, 0] = summed.ravel()
        return h_all

    def func_h_matrix(df, dim, order):
        T = df.shape[0]
        h_all = np.zeros((dim * dim * order, 1))
        for j in range(1, order + 1):
            xt = df[:, :, -1].T
            xtj = df[:, :, -1 - j].T
            # 各時点のベクトル外積の総和：einsumでテンソル生成せずに行列積にする
            h_j = xt.T @ xtj  # shape: (dim, dim)
            h_all[(j - 1) * dim * dim : j * dim * dim, 0] = h_j.ravel()   
        return h_all


    from besag import LogisticRegression
    def besag_PMLE(df,raw):
        n = len(raw)
        X = np.zeros((int((n-2)*(n-3)/2),dim*dim*order))
        base_h = func_h_matrix(df, dim, order)  # ← 固定値として1回だけ呼ぶ

        for i, (s, t) in enumerate(tqdm(combinations(range(order+1,n-order+1),2))):
            # print(i,"/", int(n*(n-1)/2))
            raw_tmp = copy.deepcopy(raw)
            raw_tmp[s-1],raw_tmp[t-1] = raw_tmp[t-1].copy(),raw_tmp[s-1].copy()
            df_tmp = raw_to_dfs(raw_tmp)
            x_ = base_h-func_h_matrix(df_tmp,dim,order) # time bottleneck            
            X[i] = x_.reshape(dim*dim*order,)

        y = np.ones(int((n-2)*(n-3)/2))
        print("Start Fitting ...")
        start_fit = time()
        clf = LogisticRegression(eta=1,n_iter=500).fit(X, y)
        end_fit = time()
        print(f"Optimization took {end_fit-start_fit} seconds.")
        return clf.w, end_fit-start_fit

    def besag_PMLE_online_SGD(df, raw, eta=0.01, n_iter=10000):
        from random import sample
        n = len(raw)
        base_h = func_h_matrix(df, dim, order)
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
            x_ = base_h - func_h_matrix(df_tmp, dim, order)
            x = x_.reshape(-1)

            # y = 1 always
            pred = 1 / (1 + np.exp(-np.dot(w, x)))
            error = 1 - pred
            step = eta * error
            w += step * x  # SGD step

            if it % 1000 == 0:
                print(f"Iter {it}, pred: {pred:.4f}")
            
            # if np.linalg.norm(step) < 1e-5:
            #     print(f"Optimization ended with {it} steps.")
            #     break
        end_fit = time()
        return w, end_fit-start_fit


    def besag_PMLE_parallel(df,raw):
        from joblib import Parallel, delayed
        n = len(raw)
        X = np.zeros((int((n-2)*(n-3)/2),dim*dim*order))
        
        def calc(v):
            s,t = v[0],v[1]
            raw_tmp = copy.deepcopy(raw)
            raw_tmp[s-1],raw_tmp[t-1] = raw_tmp[t-1],raw_tmp[s-1]
            df_tmp = raw_to_dfs(raw_tmp)    
            x_ = func_h(df,dim,order)-func_h(df_tmp,dim,order)
            return x_.T
        
        scores = Parallel(n_jobs=-1)(delayed(calc)(j) for j in combinations(range(2,n),2)) #use joblib.
        X = np.concatenate(scores)
        y = np.ones(X.shape[0])
        print("Start Fitting ...")
        start_fit = time()
        clf = LogisticRegression(eta=1,n_iter=500).fit(X, y)
        end_fit = time()
        print(f"Optimization took {end_fit-start_fit} seconds.")
        return clf.w, end_fit-start_fit

    def besag_PMLE_chen(df,raw):
        n = len(raw)-2*order
        X = np.zeros((int(n/2),dim*dim*order))
        base_h = func_h_matrix(df, dim, order)  # ← 固定値として1回だけ呼ぶ
        index_list_prep = [x+order+1 for x in list(range(n))]
        random.shuffle(index_list_prep)
        index_list = [item for item in zip(index_list_prep[:int(n/2)], index_list_prep[int(n/2):])]
        for i, (s, t) in enumerate(tqdm(index_list)):
            raw_tmp = copy.deepcopy(raw)
            raw_tmp[s-1],raw_tmp[t-1] = raw_tmp[t-1].copy(),raw_tmp[s-1].copy()
            df_tmp = raw_to_dfs(raw_tmp)
            x_ = base_h-func_h_matrix(df_tmp,dim,order) # time bottleneck      
            X[i] = x_.reshape(dim*dim*order,)

        y = np.ones(int(n/2))
        print("Start Fitting ...")
        start_fit = time()
        clf = LogisticRegression(eta=1,n_iter=500).fit(X, y)
        end_fit = time()
        print(f"Optimization took {end_fit-start_fit} seconds.")
        return clf.w, end_fit-start_fit

    ### MLE for AR or VAR
    # start_time = time()
    # res_mle = MLE(raw, order=order)
    # # theta_hat = res_mle.params[0] / res_mle.params[1] #AR(1) case
    # theta_hat = np.array([res_mle.params[k]/res_mle.params[-1] for k in range(res_mle.params.shape[0]-1)]) #AR(d) case
    # # theta_hat = res_mle.params.T @ np.linalg.inv(res_mle.sigma_u) #VAR(1) case
    # # theta_hat = theta_hat.flatten() #VAR(1) case
    # optimization_time = None
    # end_time = time()

    ### Besag's PMLE for any model
    df = raw_to_dfs(raw)
    start_time = time()
    theta_hat, optimization_time = besag_PMLE(df=df,raw=raw)
    # theta_hat, optimization_time = besag_PMLE_online_SGD(df=df,raw=raw)
    # theta_hat, optimization_time = besag_PMLE_chen(df=df,raw=raw)
    end_time = time()

    #Result
    print("--- 推定するパラメタ数 --- ")
    print(dim*dim*order)
    print("--- 推定値 --- ")
    print(theta_hat.T)
    
    # for simulated data
    print("--- 真値 --- ")
    print(true_parameter.T)
    print("--- L2誤差 --- ")
    l2loss = np.linalg.norm(true_parameter-theta_hat)
    print(l2loss)
    
    print("--- 所要時間 うち 勾配法 ---")
    comp_time = (end_time-start_time, optimization_time)
    print(comp_time)
    return theta_hat, l2loss, comp_time

if __name__ == "__main__":
    loss_list = []
    time_list = []
    for r in range(30):
        print(f"Run {r}")
        _, loss_, time_ = run()
        loss_list.append(loss_)
        time_list.append(time_)

    print("Average L2 error for 30 runs:", sum(loss_list)/len(loss_list))
    print("Average whole estimation time for 30 runs:",sum([t[0] for t in time_list])/len(time_list))
    print("Average time consumed for gradient descent for 30 runs:",sum([t[1] for t in time_list])/len(time_list))