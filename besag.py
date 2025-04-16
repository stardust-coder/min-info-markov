
import numpy as np
from tqdm import tqdm

class LogisticRegression(object):
    def __init__(self, eta=0.1, n_iter=50):
        self.eta = eta
        self.n_iter = n_iter

    def fit(self, X, y, verbose=False):
        # X = np.insert(X, 0, 1, axis=1) #intercept
        self.w = np.ones(X.shape[1])
        m = X.shape[0]
        for it_ in range(self.n_iter):
            if verbose:
                print(self.w.T)
            output = X.dot(self.w)
            errors = y - self._sigmoid(output)
            step = self.eta / m * errors.dot(X)
            self.w += step
            if np.linalg.norm(step) < 1e-5:
                print(f"Optimization ended with {it_} steps.")
                break
        return self

    def predict(self, X):
        output = np.insert(X, 0, 1, axis=1).dot(self.w)
        return (np.floor(self._sigmoid(output) + .5)).astype(int)

    def score(self, X, y):
        return sum(self.predict(X) == y) / len(y)

    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))




if __name__ == "__main__":
    pass