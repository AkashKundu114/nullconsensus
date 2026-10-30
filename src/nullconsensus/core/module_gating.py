"""
Correlation Module-Aware Feature Gating.
Identifies co-regulated gene modules from landmark anchor projections,
ensuring base ensemble models sample coordinated biological pathways.
"""

from typing import List

import numpy as np


def detect_feature_modules(
    X_pool: np.ndarray,
    n_modules: int = 10,
    random_state: int = 42,
) -> List[np.ndarray]:
    """
    Groups active candidate features into coherent co-expression modules
    using landmark covariance approximation.

    Parameters:
        X_pool: Data restricted to active feature candidate pool (N, D_pool).
        n_modules: Desired number of feature modules.
        random_state: Random state for deterministic clustering.

    Returns:
        modules: List of index arrays, each containing feature indices belonging to that module.
    """
    N, D_pool = X_pool.shape
    if D_pool <= n_modules:
        return [np.array([i]) for i in range(D_pool)]

    # Compute fast anchor/subsample feature correlation
    sub_size = min(N, 200)
    rng = np.random.default_rng(random_state)
    sub_idx = rng.choice(N, size=sub_size, replace=False)
    X_sub = X_pool[sub_idx]

    # Standardize features
    means = np.mean(X_sub, axis=0, keepdims=True)
    stds = np.std(X_sub, axis=0, keepdims=True)
    stds[stds < 1e-8] = 1.0
    X_std = (X_sub - means) / stds

    # Feature covariance / correlation matrix (D_pool x D_pool)
    corr_mat = (X_std.T @ X_std) / max(1, sub_size - 1)
    np.fill_diagonal(corr_mat, 1.0)
    # Absolute correlation for co-regulation
    affinity = np.abs(corr_mat)

    # Fast spectral or k-means grouping on feature correlation embedding
    deg = np.sum(affinity, axis=1)
    deg_inv_sqrt = 1.0 / np.sqrt(np.fmax(deg, 1e-12))
    norm_aff = deg_inv_sqrt[:, None] * affinity * deg_inv_sqrt[None, :]

    evals, evecs = np.linalg.eigh(norm_aff)
    # Top eigenvectors of normalized affinity
    n_components = min(n_modules, D_pool)
    feat_emb = evecs[:, -n_components:]

    # Group features by maximum projection or sign
    from sklearn.cluster import MiniBatchKMeans

    mbk = MiniBatchKMeans(
        n_clusters=n_modules,
        batch_size=min(512, D_pool),
        n_init=3,
        random_state=random_state,
    )
    module_labels = mbk.fit_predict(feat_emb)

    modules = []
    for m in range(n_modules):
        m_idx = np.where(module_labels == m)[0]
        if len(m_idx) > 0:
            modules.append(m_idx)

    return modules
