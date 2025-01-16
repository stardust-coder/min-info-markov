from itertools import combinations
import copy
import numpy as np
from tqdm import tqdm
from utils import raw_to_df
from model import func_h

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

if __name__ == "__main__":
    pass