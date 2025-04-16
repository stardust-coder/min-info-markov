import random
import numpy as np


def exchange(rawdata,df,theta,L):
    rawdata_aux = rawdata.copy()
    df_aux = df.copy()

    #use later
    perm_list = []
    hstar_list = []
    
    # exchange algorithm
    l = 0
    trial = 0
    while l <= L: #時間かかる
        tmp = random.sample([j for j in range(order,len(rawdata)-order)],2) #2~n-1 から2つ数字を選択
        s,t = min(tmp),max(tmp)
        #s成分とt成分を入れ替える
        rawdata_aux_tmp = rawdata_aux.copy()
        rawdata_aux_tmp[s] = rawdata_aux[t]
        rawdata_aux_tmp[t] = rawdata_aux[s]

        df_aux_tmp = raw_to_dfs(rawdata_aux_tmp) #(dim, T-order+1, order+1)
        
        h1 = func_h(df_aux_tmp,dim,order)
        h2 = func_h(df_aux,dim,order)
        
        log_rho = np.dot(theta.T,(h1-h2))
        rho = np.exp(log_rho).item()

        u = random.uniform(0,1)
        if u <= min(1,rho):
            perm_list.append((s,t))
            hstar_list.append(h2) #これh2でいいのか？
            l = l+1
            df_aux = df_aux_tmp.copy()
            print(f"{l} samples accepted ... continue")
        trial += 1
    print("Acceptance rate:", L,"/",trial,"=",L/trial)
    burnin = 100
    return perm_list[burnin:], hstar_list[burnin:], df_aux  #perm[i] has the shape (T-d,d+1)


def run(rawdata):    
    dfs = raw_to_dfs(rawdata)
    num_of_parameter = dim*dim*order

    # Fisher scoring
    cons = func_h(dfs, dim=dim, order=order)
    current_theta = np.ones((num_of_parameter,1)) #Initial estimate

    starttime = time()
    for iter in range(100): # until converge        
        perm, hstar_list, df_aux  = exchange(rawdata,dfs,L=1000,theta=current_theta) ###Exchangeするたびにh*(π)をhstar_listに記録してある.
        L = len(hstar_list)
        mu_tilde = sum(hstar_list)/L
        G = sum([np.dot((hstar_list[l]-mu_tilde).reshape(num_of_parameter,1),(hstar_list[l]-mu_tilde).reshape(1,num_of_parameter)) for l in range(0,L)])/L
        G_inv = np.linalg.inv(G)
        
        dif = np.dot(G_inv, (cons-mu_tilde))
        current_theta = current_theta + dif.reshape(num_of_parameter,1) #update theta estimates
        print(f"{iter}: Current Estimated Value")
        print(current_theta.T)
        print("The norm of steps taken")
        print(np.linalg.norm(dif))
        print("Current conditional likelihood calculated from MCMC samples.")
        print(mu_tilde.sum())

        if np.linalg.norm(dif) < 1:
            break

    comp_time = time()-starttime
    print("秒数:",comp_time,"(s)")

    return current_theta, comp_time
