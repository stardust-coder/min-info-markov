import numpy as np
import random
import pdb

def func_h(df,t):
    '''
    output: K
    '''
    x = df[t-1]
    return np.array([x[0]*x[1]]).T

def func_h_all(df):
    return sum([func_h(df,t) for t in range(1,1+len(df))])

def H(df,t,theta,Z):
    #Zで置換する代わりにdf_auxを入れてる
    #h(t)を計算
    h = func_h(df,t)
    #内積を取る
    res = np.dot(theta.T,h)
    #assert res.shape == (1,1)
    return res

def exchange(df,L,theta):
    d = df.shape[1]
    n = len(df)
    Z = np.array([[i for i in range(1,n+1)] for _ in range(d)]).T
    df_aux = df.copy()
    perm_list = []
    perm_list.append(Z)
    #あとでつかう
    hstar_list = []
    hstar_list.append(func_h_all(df_aux))

    l = 1
    while l <= L:
        tmp = random.sample([j for j in range(2,n)],2) #変更点

        s,t = min(tmp),max(tmp)
        #si成分とti成分を入れ替える
        Z_proposal = Z.copy()

        
    
        Z_proposal[s-1][0],Z_proposal[t-1][0] = Z_proposal[t-1][0],Z_proposal[s-1][0] #変更点
        Z_proposal[s-2][1],Z_proposal[t-2][1] = Z_proposal[t-2][1],Z_proposal[s-2][1] #変更点
        
        df_aux_tmp = df_aux.copy()

        df_aux_tmp[s-1][0] = df_aux[t-1][0] #変更点 
        df_aux_tmp[s-2][1] = df_aux[t-2][1] #変更点 
        df_aux_tmp[t-1][0] = df_aux[s-1][0] #変更点 
        df_aux_tmp[t-2][1] = df_aux[s-2][1] #変更点 
        

        #compute conditional likelihood ratio
        #rho = (np.exp(H(df_aux_tmp,s,theta,Z_proposal))*np.exp(H(df_aux_tmp,t,theta,Z_proposal)))/(np.exp(H(df_aux,s,theta,Z))*np.exp(H(df_aux,t,theta,Z)))
        #rho = np.exp(H(df_aux_tmp,s,theta,Z_proposal)+H(df_aux_tmp,t,theta,Z_proposal)-H(df_aux,s,theta,Z)-H(df_aux,t,theta,Z))

        rho = np.exp(theta*np.dot(df_aux_tmp[:,0],df_aux_tmp[:,1]))/np.exp(theta*np.dot(df_aux[:,0],df_aux[:,1]))
        #
        # print("rho:",rho)

        u = random.uniform(0,1)
        if u <= min(1,rho):
            perm_list.append(Z_proposal)
            #hstar_list.append(sum([func_h(df_aux,t) for t in range(1,n+1)]))#時間かかる

            hstar_list.append(func_h_all(df_aux))#時間かかる
            
            Z = Z_proposal.copy()
            l = l+1
            df_aux = df_aux_tmp
            # print(df_aux.iloc[s-1,:])
            # print(df_aux.iloc[t-1,:])
            
            # print(df_aux.iloc[s-1,:])
            # print(df_aux.iloc[t-1,:])
            # print("EXCHANGE")
        
        # if l % 100 == 0:
        #     print(l)
    return perm_list[1:], hstar_list[1:], df_aux  