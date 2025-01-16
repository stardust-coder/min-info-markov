import numpy as np
import matplotlib.pyplot as plt
from time import time
from tqdm import tqdm
from model import func_h_ar, func_h_var
from exchange import exchange_var

#Load config
import json
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('config_file', help='specify your config json file.', type=str) 
args = parser.parse_args() 
with open(args.config_file) as f:
    config = json.load(f)
order = config["order"]
dim = config["dim"]
phi = np.array(config["phi"])
phi = [np.array(x) for x in phi] #A^k, phi[0]がA^1, phi[1]がA^2... とする.
assert len(phi) == order
sigma = np.array(config["sigma"])
mean = np.array(config["mean"])
steps = config["steps"]
num_runs = config["run"]
print("--- Experimental setting ---")
print(config)


def simulation():
    error_sigma = np.array(config["sigma"])
    error_mean = np.array(config["mean"])
    var_data = []
    for _ in range(order):
        var_data.append([0,0]) # initial value
    for _ in range(order,steps+order):
        v = np.zeros((dim,1))
        for k in range(1,order+1):
            v += phi[k-1]@np.array([var_data[-k]]).T #A^k x_{t-k}            
        v += np.random.multivariate_normal(error_mean, error_sigma, 1).T
        var_data.append(v.flatten().tolist())

    var_data = np.array(var_data[order:])
    plt.figure(figsize=(15,5))
    plt.plot(var_data[:,0])
    plt.plot(var_data[:,1])
    plt.title(config)
    plt.savefig("output/simulated-data.png")
    return var_data

def run(rawdata):    
    dfs = []
    for j in range(dim):
        df = []
        for t in range(dim,len(rawdata)):
            df.append(rawdata[t-dim:t+1,j])
        dfs.append(df)        
    dfs = np.array(dfs) #shape:[dim, T-order,order+1]
    num_of_parameter = dim*dim*order

    # Fisher scoring
    cons = func_h_var(dfs, dim=dim, order=order, sigma=sigma)
    current_theta = np.ones((num_of_parameter,1)) * 0.2 #Initial estimate

    starttime = time()
    
    
    for iter in range(10): # until converge        
        perm, hstar_list, df_aux, acc = exchange_var(dfs,rawdata,L=1000,phi=current_theta,config=config) ###Exchangeするたびにh*(π)をhstar_listに記録してある.
        L = len(hstar_list)
        mu_tilde = sum(hstar_list)/L
        G = sum([np.dot((hstar_list[l]-mu_tilde).reshape(num_of_parameter,1),(hstar_list[l]-mu_tilde).reshape(1,num_of_parameter)) for l in range(0,L)])/L
        G_inv = np.linalg.inv(G)
        
        dif = np.dot(G_inv, (cons-mu_tilde))

        #update theta estimates
        current_theta = current_theta + dif.reshape(num_of_parameter,1)

        print(f"{iter}: Current Estimated Value", ",", "The norm of steps taken")
        print(current_theta.T, np.linalg.norm(dif))

    comp_time = time()-starttime
    print("秒数:",comp_time,"(s)")
    phi_hat = current_theta
    l2_err = np.linalg.norm(phi_hat-np.array(phi).flatten().reshape((num_of_parameter,1)))
    
    return phi_hat, l2_err, comp_time
    

result = []
acceptance_rates = []
computational_times = []
rmse = []

for _ in range(num_runs):    
    data = simulation() #shape: (T,dim), d: order
    phi_hat, l2_err, comp_time = run(data)
    computational_times.append(comp_time)
    #Result
    print("--- 推定するパラメタ数 --- ")
    print(dim*dim*order)
    print("--- 推定値 --- ")
    print(phi_hat)
    print("--- 真値 --- ")
    print(phi)
    print("--- L2 error --- ")
    result.append(phi_hat)
    rmse.append(l2_err) #L2-norm

print(f"{num_runs} 回の施行結果")
print(result)
print(rmse)
print("RMSE=",sum(rmse)/len(rmse))