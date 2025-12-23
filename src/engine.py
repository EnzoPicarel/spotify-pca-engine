import numpy as np
from typing import Tuple

class CustomPCA:
    def __init__(self, n_components: int = 2):
        self.n_components = n_components
        self.mean = None
        self.std = None
        self.eigenvalues = None
        self.eigenvectors = None
        self.explained_variance_ratio = None
        self.feature_names = None

    def fit(self, X: np.ndarray, feature_names: list = None) -> 'CustomPCA':
        self.feature_names = feature_names
        
        # 1. z-score norm
        self.mean = np.mean(X, axis=0)
        self.std = np.std(X, axis=0)
        
        # safety: avoid div by zero
        self.std[self.std == 0] = 1.0 
        
        X_std = (X - self.mean) / self.std

        # 2. cov matrix (1/n * Z.T * Z)
        n = X_std.shape[0]
        corr_matrix = (1 / n) * np.dot(X_std.T, X_std)

        # 3. eigen decomp
        eig_vals, eig_vecs = np.linalg.eig(corr_matrix)

        # 4. sort desc
        idx = np.argsort(eig_vals)[::-1]
        self.eigenvalues = eig_vals[idx]
        self.eigenvectors = eig_vecs[:, idx]

        # 5. explained variance
        self.explained_variance_ratio = self.eigenvalues / np.sum(self.eigenvalues)

        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        # project data
        X_std = (X - self.mean) / self.std
        proj_mat = self.eigenvectors[:, :self.n_components]
        return np.dot(X_std, proj_mat)

    def get_components_stats(self) -> np.ndarray:
        # corr = sqrt(lambda) * eigenvector
        correlations = np.zeros((len(self.feature_names), self.n_components))
        for k in range(self.n_components):
            correlations[:, k] = np.sqrt(self.eigenvalues[k]) * self.eigenvectors[:, k]
        return correlations