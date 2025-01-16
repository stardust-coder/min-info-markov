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


def h(theta,x,y):
    return theta*x*y

def permutation_distribution(theta,X):
    n = len(X)
    l = X[1:-1]
    f = [] #note: non-normalized distribution
    for i in permutations(l,len(l)):
        X_pi = [X[0]] + list(i) + [X[-1]] #X_pi[0] corresponds to identity relation
        kernel = sum([h(theta,X_pi[t-1-1],X_pi[t-1]) for t in range(2,n+1)]) - 50
        f.append(exp(kernel))
    return f


def conditional_likelihood(theta,X):
    f = permutation_distribution(theta,X)
    return f[0]/sum(f) #normalize when calculating conditional likelihood

def negative_conditional_likelihood(theta,X):
    return -conditional_likelihood(theta,X)

def CLE(X): #maximize conditional likelihood w.r.t. theta
    f = functools.partial(negative_conditional_likelihood, X=X)
    res = optimize.minimize_scalar(f, method="brent", tol=1e-5)
    return res

def func_h(df,t,mode):
    '''
    output: K
    '''
    if mode["model"]=="AR":
        assert t-mode["p"] >= 0
        x = df[t-mode["p"]]

        res_ = []
        for j in range(1,mode["p"]+1):
            res_.append([x[-1]*x[-1-j]])
        return np.array(res_)
    else:
        return None

def func_h_all(df,mode):
    return sum([func_h(df,t,mode) for t in range(mode["p"],1+len(df))])

def exchange(df,L,theta,mode):
    d = df.shape[1]
    n = len(df)
    Z = np.array([[i for i in range(1,n+1)] for _ in range(d)]).T
    df_aux = df.copy()
    perm_list = []
    perm_list.append(Z)
    #あとでつかう
    hstar_list = []
    hstar_list.append(func_h_all(df_aux,mode))

    l = 1
    accept = 0
    while l <= L:
        tmp = random.sample([j for j in range(mode["p"]+1,n)],2) #変更点

        s,t = min(tmp),max(tmp)
        #si成分とti成分を入れ替える
        Z_proposal = Z.copy()    
        df_aux_tmp = df_aux.copy()

        for i in range(1,mode["p"]+2):
            Z_proposal[s-i][i-1],Z_proposal[t-i][i-1] = Z_proposal[t-i][i-1],Z_proposal[s-i][i-1] #変更点
            df_aux_tmp[s-i][i-1],df_aux_tmp[t-i][i-1] = df_aux[t-i][i-1],df_aux[s-i][i-1] #変更点  

        
        log_rho = np.dot(theta.T,func_h_all(df_aux_tmp,mode)-func_h_all(df_aux,mode))
        rho = np.exp(log_rho).item()
        #print("rho:",rho)
        u = random.uniform(0,1)
        if u <= min(1,rho):
            perm_list.append(Z_proposal)
            hstar_list.append(func_h_all(df_aux,mode))#時間かかる
            
            Z = Z_proposal.copy()
            l = l+1
            df_aux = df_aux_tmp
        accept += 1
    print("Acceptance rate:", L,"/",accept,"=",L/accept)
    return perm_list[1:], hstar_list[1:], df_aux #perm[i] has the shape (T-d,d+1)


# exchange algorithm CLE
import json
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('config_file', help='specify your config json file.') 
args = parser.parse_args() 
with open(args.config_file) as f:
    config = json.load(f)


def simulated_data():
    if "AR" in config["model"]:
        phi = np.array(config["phi"]) # for AR(1), stationary condition is |phi1| < 1
        p = config["p"]
        assert len(phi) == p
        c = config["mean"]
        sigma = config["sigma"] # sigma^2 = 0.25
        T = config["steps"]
        mu = c/(1-np.sum(phi)) #  #c + phi1 * mu + phi2 * mu = mu =>  mu = c/(1-phi1-phi2)
        ar_data = np.zeros(T)
        for i in range(p): #t=1~p
            ar_data[i] = mu + np.random.normal(0, sigma)
        for t in range(p, T): #t=p+1~
            ar_data[t] = c + phi @ ar_data[t-p:t] + np.random.normal(0, sigma)
        ar_data = list(ar_data)
        data = ar_data
    return data, p

result = []
rmse = []
for _ in range(config["run"]):
    data,d = simulated_data() #d=2 in AR(2)
    df = []
    for t in range(d,len(data)):
        df.append(data[t-d:t+1])
    df = np.array(df) #shape:[T-d,d+1]
    n = len(df)
    assert n+d == len(data)
    print("shape:","=",n,"x",d+1,",","d=",d)

    #run
    current_theta = np.array([[0.5 for _ in range(d)]]).T #Initial estimates
    cons = func_h_all(df,mode=config)
    iternum = 10000 #iteration of 
    iter = 0

    starttime = time()
    #while True: 
    for _ in tqdm(range(100)):
        iter += 1   
        perm, hstar_list, df_aux = exchange(df,L=iternum,theta=current_theta,mode=config) ###Exchangeするたびにh*(π)を記録してある.
        L = len(hstar_list)
        
        mu_tilde = sum(hstar_list)/L
        G = sum([np.dot((hstar_list[l]-mu_tilde).reshape(d,1),(hstar_list[l]-mu_tilde).reshape(1,d)) for l in range(0,L)])/L
        G_inv = np.linalg.inv(G)
        
        dif = np.dot(G_inv, (cons-mu_tilde))
        current_theta = current_theta + dif
        print(current_theta.T, np.linalg.norm(dif))
        #print(np.max(np.abs(cons-mu_tilde))/n)
        if np.max(np.abs(cons-mu_tilde))/n <= 1e-3:
            break
    
    print("秒数:",time()-starttime,"(s)")
    theta_hat = current_theta
    print("iteration数:", iter)
    

    #AR(1)
    phi_hat = (config["sigma"]**2)*theta_hat
    print("="*5)
    print("推定値")
    print(phi_hat)
    print("実際の値")
    print(config["phi"])
    print("="*5)
    result.append(phi_hat)
    rmse.append(np.linalg.norm(phi_hat-np.array([config["phi"]]).T))


if config["p"] == 1:
    print("Result of AR(1)")
    print(result)
    plt.hist(result)
    plt.savefig("histogram.png")



print("推定値の集合")
print(result)
print("RMSE")
print(sum(rmse)/len(rmse))


