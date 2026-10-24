"GaussianSketch implementation"

import numpy as np

class GaussianSketch:
    def __init__(self, n, d, rng):
        self.n, self.d = n, d
        self.Theta = rng.standard_normal((d, n)) / np.sqrt(d)
        self._ThetaC = None
        self._C_ref = None

    @staticmethod
    def _mul(T, X):
        if np.iscomplexobj(X):
            return T @ X.real + 1j * (T @ X.imag)
        return T @ X

    def apply_columns(self, X):
        return np.asfortranarray(self._mul(self.Theta, X))

    def __call__(self, x):
        return self.apply_columns(np.asarray(x)[:, None])[:, 0]

    def prepare_C(self, C_perm):
        if self._C_ref is not C_perm:
            self._ThetaC = np.asfortranarray((C_perm.T @ self.Theta.T).T)
            self._C_ref = C_perm

    def apply_columns_C(self, X, C_perm):
        self.prepare_C(C_perm)
        return np.asfortranarray(self._mul(self._ThetaC, X))