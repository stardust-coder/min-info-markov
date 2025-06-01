
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




if __name__ == "__main__":
    pass