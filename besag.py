
import numpy as np
from tqdm import tqdm

class LogisticRegression(object):
    def __init__(self, eta=0.1, n_iter=50):
        self.eta = eta
        self.n_iter = n_iter

    def fit(self, X, y, verbose=False):
        self.w = np.zeros(X.shape[1])
        m = X.shape[0]
        prev_loss = np.inf

        for it in range(self.n_iter):
            p = self._sigmoid(X @ self.w)
            errors = y - p

            # 学習率減衰
            eta_t = self.eta #/ (1 + 0.05 * it)

            # 重み更新
            step = eta_t / m * (X.T @ errors)
            self.w += step

            # ロス計算
            loss = -np.mean(y * np.log(p + 1e-15) + (1 - y) * np.log(1 - p + 1e-15))
            if verbose:
                print(f"Iter {it+1}: loss={loss:.6f}")

            # シンプルな収束判定
            if abs(prev_loss - loss) < 1e-6:
                if verbose:
                    print(f"Converged at iteration {it+1}")
                grad = X.T @ (1 - 1 / (1 + np.exp(-(X @ self.w))))
                print("grad norm =", np.linalg.norm(grad))
                print("max abs grad =", np.max(np.abs(grad)))   
                break
            prev_loss = loss

        return self
    
    def fit_add(self, X, y, verbose=False):
        m = X.shape[0]
        prev_loss = np.inf
        for it_ in range(self.n_iter):
            if verbose:
                print(self.w.T)
            output = X.dot(self.w)
            p = self._sigmoid(output)
            errors = y - p
            step = self.eta / m * errors.dot(X)
            self.w += step

            loss = -np.mean(y * np.log(p + 1e-15) + (1 - y) * np.log(1 - p + 1e-15))
            if verbose:
                print(f"Iter {it_}: loss={loss:.6f}, step_norm={np.linalg.norm(step):.6f}")

            # ロスの変化で収束判定
            if it_ > 10 and abs(prev_loss - loss) < 1e-5:
                print(f"Converged at iteration {it_+1}")
                break
            prev_loss = loss
        print(f"Optimization ended with full {it_+1} steps.")
        return self

    def predict(self, X):
        output = np.insert(X, 0, 1, axis=1).dot(self.w)
        return (np.floor(self._sigmoid(output) + .5)).astype(int)

    def score(self, X, y):
        return sum(self.predict(X) == y) / len(y)

    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    def _log_loss(self, X, y):
        """Logistic loss function (negative log-likelihood)"""
        z = X.dot(self.w)
        p = self._sigmoid(z)
        # 安定化（log(0)対策）
        p = np.clip(p, 1e-10, 1 - 1e-10)
        loss = -np.sum(y * np.log(p) + (1 - y) * np.log(1 - p))
        return loss


def besag_PMLE_parallel(df,raw):
    from joblib import Parallel, delayed
    n = len(raw)
    X = np.zeros((int((n-2)*(n-3)/2),dim*dim*order))
    
    def calc(v):
        s,t = v[0],v[1]
        raw_tmp = copy.deepcopy(raw)
        raw_tmp[s-1],raw_tmp[t-1] = raw_tmp[t-1],raw_tmp[s-1]
        df_tmp = raw_to_dfs(raw_tmp)    
        x_ = func_h(df,dim,order)-func_h(df_tmp,dim,order)
        return x_.T
    
    scores = Parallel(n_jobs=-1)(delayed(calc)(j) for j in combinations(range(2,n),2)) #use joblib.
    X = np.concatenate(scores)
    y = np.ones(X.shape[0])
    print("Start Fitting ...")
    start_fit = time()
    clf = LogisticRegression(eta=1,n_iter=500).fit(X, y)
    end_fit = time()
    print(f"Optimization took {end_fit-start_fit} seconds.")
    return clf.w, end_fit-start_fit

def besag_PMLE_chen(df,raw):
    n = len(raw)-2*order
    X = np.zeros((int(n/2),dim*dim*order))
    base_h = func_h_matrix(df, dim, order)  # ← 固定値として1回だけ呼ぶ
    index_list_prep = [x+order+1 for x in list(range(n))]
    random.shuffle(index_list_prep)
    index_list = [item for item in zip(index_list_prep[:int(n/2)], index_list_prep[int(n/2):])]
    for i, (s, t) in enumerate(tqdm(index_list)):
        raw_tmp = copy.deepcopy(raw)
        raw_tmp[s-1],raw_tmp[t-1] = raw_tmp[t-1].copy(),raw_tmp[s-1].copy()
        df_tmp = raw_to_dfs(raw_tmp)
        x_ = base_h-func_h_matrix(df_tmp,dim,order) # time bottleneck      
        X[i] = x_.reshape(dim*dim*order,)

    y = np.ones(int(n/2))
    print("Start Fitting ...")
    start_fit = time()
    clf = LogisticRegression(eta=1,n_iter=500).fit(X, y)
    end_fit = time()
    print(f"Optimization took {end_fit-start_fit} seconds.")
    return clf.w, end_fit-start_fit


if __name__ == "__main__":
    pass