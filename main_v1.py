import numpy as np
from math import exp
import random
import matplotlib.pyplot as plt
from itertools import permutations
import scipy.optimize as optimize
import functools
#from fisher_scoring import *
from fisher_scoring_v1 import *
import pdb
from time import time
from tqdm import tqdm

#minimum information markox process modelling
def h(theta,x,y):
    return theta*x*y

def permute(X): #note: not used
    ''' X : list '''
    l = X[1:-1]
    permuted = [X[0]] + random.sample(l, len(l)) + [X[-1]]
    return permuted

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

result = []
for _ in range(30):
    ### Simulation
    # AR(1) process
    phi1 = 0.5 # stationary condition is |phi1| < 1
    c = 0 
    sigma = 0.5 # sigma^2 = 0.25
    T = 500
    mu = c / (1 - phi1) #c + phi * mu = mu =>  mu = c/(1-phi)

    ar_data = np.zeros(T)
    ar_data[0] = mu + np.random.normal(0, sigma)
    for t in range(1, T):
        ar_data[t] = c + phi1 * ar_data[t-1] + np.random.normal(0, sigma)
    ar_data = list(ar_data)
    data = ar_data
    
    #Poisson(λ)
    # tmax=200. # seconds
    # tbin=200 # number of time bins
    # meanrate=16. # Hz
    # time=np.linspace(0,tmax,tbin) # 
    # dt = tmax/tbin # sampling time
    # data = np.random.poisson(lam = meanrate, size = tbin)

    ### Check 1.
    # opt = CLE(ar_data) #計算が終わらない...
    # theta_hat = opt.x
    # phi_hat = (sigma**2)*theta_hat
    # print(f"推定値:{phi_hat}, 実際の値:{phi1}")

    ### Check2.

    df = []
    for t in range(1,len(data)):
        df.append([data[t-1],data[t]])
    df = np.array(df)
    d = df.shape[1]
    n = len(df)
    
    # df = np.array([ar_data]).T
    # d = df.shape[1]
    # n = len(df)
    
    
    print(n,d)

    #run
    current_theta = np.array([[1]]).T #初期値. テキトーに設定.
    K = len(current_theta)
    cons = func_h_all(df)
    iternum = 10000
    iter = 0

    starttime = time()
    #while True: 
    for _ in tqdm(range(100)):
        iter += 1   
        _, hstar_list, _ = exchange(df,L=iternum,theta=current_theta) ###Exchangeするたびにh*(π)を記録してある.

        L = len(hstar_list)
        mu_tilde = sum(hstar_list)/L
        G = sum([np.dot((hstar_list[l]-mu_tilde).reshape(K,1),(hstar_list[l]-mu_tilde).reshape(1,K)) for l in range(0,L)])/L
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
    phi_hat = (sigma**2)*theta_hat
    print(f"推定値:{phi_hat.item()}, 実際の値:{phi1}")
    result.append(phi_hat.item())

    #Possion(λ)


print(result)
import matplotlib.pyplot as plt
plt.hist(result)
plt.savefig("histogram.png")



