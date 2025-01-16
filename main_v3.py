import matplotlib.pyplot as plt
import numpy as np
from math import exp
import random
import matplotlib.pyplot as plt
from itertools import permutations
import scipy.optimize as optimize
import functools
from time import time
from tqdm import tqdm
import pdb

def raw_to_df(rawdata,d):
    df = []
    for t in range(d,len(rawdata)):
        df.append(rawdata[t-d:t+1])
    df = np.array(df) #shape:[T-d,d+1]
    return df

def func_h(df,t,mode):
    '''
    output: K
    '''
    if mode["model"]=="AR":
        d = mode["p"]
        x = df[t]
        s= mode["sigma"]
        res = []
        for i in range(1,d+1):
            res.append((x[-1-i]*x[-1])/((s**2)))
        return np.array(res)
    
def func_h_all(df,mode):
    return sum([func_h(df,t,mode) for t in range(0,len(df))])

def exchange(df,rawdata,L,phi,mode):
    '''
    phi: AR parameter, d-dim, phi[i]がx[t-i]に対応する係数とする.
    '''
    assert df.shape[1] == mode["p"]+1
    d = mode["p"]
    n = len(df)
    df_aux = df.copy()
    rawdata_aux = rawdata.copy()

    #use later
    perm_list = []
    hstar_list = []
    s = mode["sigma"]

    l = 1
    accept = 0
    while l <= L:
        tmp = random.sample([j for j in range(2,n)],2) #変更点

        s,t = min(tmp),max(tmp)
        #si成分とti成分を入れ替える

        #df_aux_tmp = df_aux.copy()
        rawdata_aux_tmp = rawdata_aux.copy()
        
        #微妙に間違ってる
        # for i in range(1,mode["p"]+2):
        #     Z_proposal[s-i][i-1],Z_proposal[t-i][i-1] = Z_proposal[t-i][i-1],Z_proposal[s-i][i-1] #変更点
        #     df_aux_tmp[s-i][i-1],df_aux_tmp[t-i][i-1] = df_aux[t-i][i-1],df_aux[s-i][i-1] #変更点  
        
        #1,n以外が選択されるように修正
        rawdata_aux_tmp[s-1] = rawdata_aux[t-1]
        rawdata_aux_tmp[t-1] = rawdata_aux[s-1]
        df_aux_tmp = raw_to_df(rawdata_aux_tmp,d)

        #(1/2s^2) Σ_{i=1}^d x_t-i x_t

        log_rho = phi.T@(func_h_all(df_aux_tmp,mode)-func_h_all(df_aux,mode))        
        rho = np.exp(log_rho).item()
        
        u = random.uniform(0,1)
        if u <= min(1,rho):
            perm_list.append((s,t))
            hstar_list.append(func_h_all(df_aux,mode))#時間かかる

            l = l+1
            df_aux = df_aux_tmp
        accept += 1
    print("Acceptance rate:", L,"/",accept,"=",L/accept)
    return perm_list[1:], hstar_list[1:], df_aux, L/accept #perm[i] has the shape (T-d,d+1)



def exchange_var(df,rawdata,L,phi,config):
    n = df.shape[1] #T-order
    d = config["order"]
    dim = config["dim"]
    error_sigma = np.array(config["sigma"])
    error_sigma_inv = np.linalg.inv(error_sigma)
    
    df_aux = df.copy()
    rawdata_aux = rawdata.copy()

    #use later
    perm_list = []
    hstar_list = []
    
    # exchange algorithm
    l = 1
    accept = 0

    def raw_to_dfs(rawdata):
        dfs = []
        for j in range(dim):
            df = []
            for t in range(dim,len(rawdata)):
                df.append(rawdata[t-dim:t+1,j])
            dfs.append(df)        
        dfs = np.array(dfs)
        return dfs
    
    def func_h_var(df):
        h_all = np.zeros((dim*dim*d,1))
        for t in range(df.shape[1]):
            tmp = []
            for j in range(1,d+1):
                xt = df[:,t,-1]
                xtj = df[:,t,-1-j]
                tmp.append(np.kron(error_sigma_inv@xt,xtj))
            h_all += np.concatenate(tmp).reshape((dim*dim*d,1))
        return h_all

    while l <= L:
        #print("l=",l)
        tmp = random.sample([j for j in range(2,n)],2) #2~n-1 から2つ数字を選択
        s,t = min(tmp),max(tmp)
        #s成分とt成分を入れ替える
        rawdata_aux_tmp = rawdata_aux.copy()
        rawdata_aux_tmp[s-1] = rawdata_aux[t-1]
        rawdata_aux_tmp[t-1] = rawdata_aux[s-1]

        df_aux_tmp = raw_to_dfs(rawdata_aux_tmp) #(dim, T-order+1, order+1)
        
        log_rho = np.dot(phi.T,(func_h_var(df_aux_tmp)-func_h_var(df_aux)))
        rho = np.exp(log_rho).item()

        u = random.uniform(0,1)
        if u <= min(1,rho):
            perm_list.append((s,t))
            hstar_list.append(func_h_var(df_aux))#時間かかる
            l = l+1
            df_aux = df_aux_tmp.copy()
        accept += 1
    print("Acceptance rate:", L,"/",accept,"=",L/accept)
    return perm_list[1:], hstar_list[1:], df_aux, L/accept #perm[i] has the shape (T-d,d+1)



# exchange algorithm CLE
import json
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('config_file', help='specify your config json file.') 
args = parser.parse_args() 
with open(args.config_file) as f:
    config = json.load(f)


def simulated_data(config):
    if "AR" == config["model"]:
        phi = np.array(config["phi"]) # for AR(1), stationary condition is |phi1| < 1
        p = config["p"]
        assert len(phi) == p
        c = config["mean"]
        sigma = config["sigma"] # sigma^2 = 0.25
        T = config["steps"] # num of data
        mu = c/(1-np.sum(phi)) #  #c + phi1 * mu + phi2 * mu = mu =>  mu = c/(1-phi1-phi2)
        ar_data = np.zeros(T)
        for i in range(p): #t=1~p
            ar_data[i] = mu + np.random.normal(0, sigma)
        for t in range(p, T): #t=p+1~
            ar_data[t] = c + phi @ ar_data[t-p:t][::-1] + np.random.normal(0, sigma)
        ar_data = list(ar_data)
        data = ar_data
        return data, p
    
    if "VAR" == config["model"]:
        order = config["order"]
        dim = config["dim"]
        phi = config["phi"]
        phi = [np.array(x) for x in phi] #A^k, phi[0]がA^1, phi[1]がA^2... とする.
        error_sigma = np.array(config["sigma"])
        error_mean = np.array(config["mean"])
        T = config["steps"]
        var_data = []
        for i in range(order):
            var_data.append([0,0])
        for t in range(order,T+order):
            v = np.zeros((dim,1))
            for k in range(1,order+1):
                v += phi[k-1]@np.array([var_data[-k]]).T #A^k x_{t-k}            
            v += np.random.multivariate_normal(error_mean, error_sigma, 1).T
            var_data.append(v.flatten().tolist())
        return np.array(var_data[order:])

class LogisticRegression(object):
    def __init__(self, eta=0.1, n_iter=50):
        self.eta = eta
        self.n_iter = n_iter

    def fit(self, X, y):
        # X = np.insert(X, 0, 1, axis=1) #intercept
        self.w = np.ones(X.shape[1])
        m = X.shape[0]
        for _ in range(self.n_iter):
            output = X.dot(self.w)
            errors = y - self._sigmoid(output)
            self.w += self.eta / m * errors.dot(X)
        return self

    def predict(self, X):
        output = np.insert(X, 0, 1, axis=1).dot(self.w)
        return (np.floor(self._sigmoid(output) + .5)).astype(int)

    def score(self, X, y):
        return sum(self.predict(X) == y) / len(y)

    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

from itertools import combinations
import copy
def besag_PMLE(df,raw,config):
    n= len(raw)
    X = np.zeros((int((n-2)*(n-3)/2),config["p"]))
    for i,v in tqdm(enumerate(combinations(range(2,n),2))):
        s,t = v[0],v[1]
        raw_tmp = copy.deepcopy(raw)
        raw_tmp[s-1],raw_tmp[t-1] = raw_tmp[t-1],raw_tmp[s-1]
        df_tmp = raw_to_df(raw_tmp,config["p"])
        X[i] = np.sum([func_h(df,t,mode=config)-func_h(df_tmp,t,mode=config) for t in range(len(df))],axis=0) 
    y = np.ones(int((n-2)*(n-3)/2))
    clf = LogisticRegression().fit(X, y)
    return clf.w


if "AR" == config["model"]:
    result = []
    acceptance_rates = []
    computational_times = []
    rmse = []
    for _ in range(config["run"]):
        rawdata,d = simulated_data(config) #d=2 in AR(2), shape: (T,)
        df = []
        for t in range(d,len(rawdata)):
            df.append(rawdata[t-d:t+1])
        df = np.array(df) #shape:[T-d,d+1]
        n = len(df)
        assert n+d == len(rawdata)
        print("shape:","=",n,"x",d+1,",","d=",d)


        starttime = time()

        if config["use_besag"]:
            phi_hat = besag_PMLE(df,rawdata,config).T
            
        else:
            #run
            current_phi = np.array([[0.2 for _ in range(d)]]).T #Initial estimates
            cons = func_h_all(df,mode=config)
            iternum = 10000 #iteration of 
            iter = 0
            
            #while True: 
            for _ in tqdm(range(30)):
                iter += 1   

                perm, hstar_list, df_aux, acc = exchange(df,rawdata,L=iternum,phi=current_phi,mode=config) ###Exchangeするたびにh*(π)を記録してある.
                L = len(hstar_list)
                
                mu_tilde = sum(hstar_list)/L
                G = sum([np.dot((hstar_list[l]-mu_tilde).reshape(d,1),(hstar_list[l]-mu_tilde).reshape(1,d)) for l in range(0,L)])/L
                G_inv = np.linalg.inv(G)
                
                dif = np.dot(G_inv, (cons-mu_tilde))
                current_phi = current_phi + dif.reshape(d,1)

                print("Current Estimated Value", ",", "The norm of steps taken")
                print(current_phi.T, np.linalg.norm(dif))
                acceptance_rates.append(acc)
                #print(np.max(np.abs(cons-mu_tilde))/n)

                if np.linalg.norm(dif) < 1e-2:
                    break
                # if np.max(np.abs(cons-mu_tilde))/n <= 1e-5:
                #     break
            phi_hat = current_phi
            print("iteration数:", iter)

        comp_time = time()-starttime
        print("秒数:",comp_time,"(s)")
        computational_times.append(comp_time)
        #Result
        print("="*5)
        print("推定値")
        print(phi_hat)
        print("実際の値")
        print(config["phi"])
        print("="*5)
        result.append(phi_hat)
        rmse.append(np.linalg.norm(phi_hat-np.array([config["phi"]]).T)) #L2-norm

    print("推定値の集合")
    print(result)
    print("RMSE")
    print(sum(rmse)/len(rmse))
    if not config["use_besag"]:
        print("Mean of acceptance rates")
        print(sum(acceptance_rates)/len(acceptance_rates))
    print("合計実行時間")
    print(sum(computational_times)," seconds.")

if "VAR" == config["model"]:
    result = []
    acceptance_rates = []
    computational_times = []
    rmse = []
    for _ in range(config["run"]):
        rawdata = simulated_data(config) #shape: (T,dim), d: order
        order = config["order"]
        dim = config["dim"]
        
        # plt.plot(rawdata[:,0])
        # plt.plot(rawdata[:,1])

        dfs = []
        for j in range(config["dim"]):
            df = []
            for t in range(dim,len(rawdata)):
                df.append(rawdata[t-dim:t+1,j])
            dfs.append(df)        
        
        dfs = np.array(dfs) #shape:[dim, T-order,order+1]
    
        #run
        num_of_parameter = dim*dim*order
    
        current_theta = np.ones((num_of_parameter,1)) * 0.2 #Initial estimates

        def func_h_var(df):
            d = config["order"]
            h_all = np.zeros((dim*dim*order,1))
            
            for t in range(df.shape[1]):
                tmp = []
                for j in range(1,d+1):
                    xt = df[:,t,-1]
                    xtj = df[:,t,-1-j]
                    tmp.append(np.kron(np.linalg.inv(config["sigma"])@xt,xtj))  

                h_all += np.concatenate(tmp).reshape((dim*dim*order,1))
            return h_all
    
        cons = func_h_var(dfs)
        iternum = 10000 #iteration of 
        iter = 0

        starttime = time()
        #while True: 
        for _ in tqdm(range(100)):
            iter += 1   
            perm, hstar_list, df_aux, acc = exchange_var(dfs,rawdata,L=iternum,phi=current_theta,config=config) ###Exchangeするたびにh*(π)を記録してある.
            L = len(hstar_list)
            
            mu_tilde = sum(hstar_list)/L
            G = sum([np.dot((hstar_list[l]-mu_tilde).reshape(num_of_parameter,1),(hstar_list[l]-mu_tilde).reshape(1,num_of_parameter)) for l in range(0,L)])/L
            G_inv = np.linalg.inv(G)
            
            dif = np.dot(G_inv, (cons-mu_tilde))
            current_theta = current_theta + dif.reshape(num_of_parameter,1)

            print("Current Estimated Value", ",", "The norm of steps taken")
            print(current_theta.T, np.linalg.norm(dif))
            acceptance_rates.append(acc)
            #print(np.max(np.abs(cons-mu_tilde))/n)

            if np.linalg.norm(dif) < 1e-2:
                break
            # if np.max(np.abs(cons-mu_tilde))/n <= 1e-5:
            #     break
        comp_time = time()-starttime
        print("秒数:",comp_time,"(s)")
        computational_times.append(comp_time)
        phi_hat = current_theta
        print("iteration数:", iter)
        
        #Result
        print("="*5)
        print("推定するパラメタ")
        print(num_of_parameter)
        print("推定値")
        print(phi_hat)
        print("実際の値")
        print(config["phi"])
        print("="*5)
        result.append(phi_hat)
        rmse.append(np.linalg.norm(phi_hat-np.array(config["phi"]).flatten().reshape((num_of_parameter,1)))) #L2-norm

    print("推定値の集合")
    print(result)
    print("RMSE")
    print(sum(rmse)/len(rmse))
    print("Mean of acceptance rates")
    print(sum(acceptance_rates)/len(acceptance_rates))
    print("合計実行時間")
    print(sum(computational_times)," seconds.")