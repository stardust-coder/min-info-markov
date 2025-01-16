import numpy as np
import random
import pdb

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

        for i in range(1,mode["p"]+1):
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
    return perm_list[1:], hstar_list[1:], df_aux
    #return perm_list[3000:], hstar_list[3000:], df_aux
