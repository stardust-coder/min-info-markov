import numpy as np

def func_h_ar(df,t,mode):
    '''
    output: K
    '''
    d = mode["p"]
    x = df[t]
    s= mode["sigma"]
    res = []
    for i in range(1,d+1):
        # res.append((x[-1-i]*x[-1])/((s**2))) #sigma:known = not included in the model
        res.append((x[-1-i]*x[-1]))   #sigma:unknown = included in the model
    return np.array(res)

def func_h_var(df, dim, order, sigma):
    h_all = np.zeros((dim*dim*order,1))
    for t in range(df.shape[1]):
        tmp = []
        for j in range(1,order+1):
            xt = df[:,t,-1]
            xtj = df[:,t,-1-j]
            # tmp.append(np.kron(np.linalg.inv(sigma)@xt,xtj))   #sigma:known = not included in the model
            tmp.append(np.kron(xt,xtj))   #sigma:unknown = included in the model
        h_all += np.concatenate(tmp).reshape((dim*dim*order,1))
    return h_all