import numpy as np
import random
from model import func_h_var
from utils import raw_to_dfs

def exchange_var(df,rawdata,L,phi,config):
    '''この関数内での論文ではパラメタΘなので注意'''
    n = df.shape[1] #T-order
    order = config["order"]
    dim = config["dim"]
    sigma = np.array(config["sigma"])
    
    df_aux = df.copy()
    rawdata_aux = rawdata.copy()

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

        df_aux_tmp = raw_to_dfs(rawdata_aux_tmp, dim, order) #(dim, T-order+1, order+1)
        
        h1 = func_h_var(df_aux_tmp,dim,order,sigma)
        h2 = func_h_var(df_aux,dim,order,sigma)
        
        log_rho = np.dot(phi.T,(h1-h2))
        rho = np.exp(log_rho).item()

        u = random.uniform(0,1)
        if u <= min(1,rho):
            perm_list.append((s,t))
            hstar_list.append(h2)
            l = l+1
            df_aux = df_aux_tmp.copy()
            print(f"{l} samples accepted ... continue")
        trial += 1
    print("Acceptance rate:", L,"/",trial,"=",L/trial)
    return perm_list[1:], hstar_list[1:], df_aux, L/trial #perm[i] has the shape (T-d,d+1)
