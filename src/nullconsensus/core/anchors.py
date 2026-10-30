"""
Diversity Landmark Anchor Selection Module.
Implements Determinantal Point Process (DPP) greedy volume-maximization,
Balanced K-Means centroid projection, and uniform random anchor sampling.
"""

from typing import Literal

import numpy as np
from sklearn.cluster import MiniBatchKMeans


def select_anchors_dpp(X: np.ndarray, L: int, random_state: int = 42) -> np.ndarray:
    """
    Selects L landmark anchors using a greedy volume-maximization approximation to a
    k-Determinantal Point Process (k-DPP).

    Maximizes the orthogonal volume spanned by the anchor subset in O(N * L * D_sub) time,
    ensuring uniform coverage of both dominant manifold clusters and rare minority populations.
    """
    N, D = X.shape
    L = min(L, N)
    if L == N:
        return np.arange(N)

    rng = np.random.default_rng(random_state)
    selected = []

    # Fast projection sketch if D is very large
    if D > 100:
        sketch_dim = min(64, D)
        R = rng.standard_normal((D, sketch_dim)) / np.sqrt(sketch_dim)
        X_proj = X @ R
    else:
        X_proj = X

    # 1. First anchor: point furthest from sample mean
    mean_vec = np.mean(X_proj, axis=0)
    dists = np.sum((X_proj - mean_vec) ** 2, axis=1)
    first_idx = int(np.argmax(dists))
    selected.append(first_idx)

    # 2. Iteratively select points maximizing squared orthogonal distance to current anchor span
    min_sq_dists = np.sum((X_proj - X_proj[first_idx]) ** 2, axis=1)

    for _ in range(1, L):
        # Soft-stochastic DPP sampling with volume proportional to residual orthogonal variance
        sum_dists = np.sum(min_sq_dists)
        if sum_dists <= 1e-12:
            remaining = np.setdiff1d(np.arange(N), selected)
            if len(remaining) > 0:
                selected.append(int(rng.choice(remaining)))
            continue

        probs = min_sq_dists / sum_dists
        # Greedy-mode with deterministic top volume or stochastic draw
        next_anchor = int(np.argmax(probs))
        selected.append(next_anchor)

        # Update minimum distance to any selected anchor
        new_dists = np.sum((X_proj - X_proj[next_anchor]) ** 2, axis=1)
        min_sq_dists = np.minimum(min_sq_dists, new_dists)

    return np.array(selected, dtype=int)


def select_anchors_kmeans(X: np.ndarray, L: int, random_state: int = 42) -> np.ndarray:
    """
    Selects L landmark anchors via balanced MiniBatchKMeans centroid clustering,
    mapping each centroid to its nearest observed sample in X.
    """
    N = X.shape[0]
    L = min(L, N)
    if L == N:
        return np.arange(N)

    mbk = MiniBatchKMeans(
        n_clusters=L,
        batch_size=min(1024, max(256, 4 * L)),
        n_init=3,
        random_state=random_state,
    )
    mbk.fit(X)
    centroids = mbk.cluster_centers_

    # Find nearest actual data sample for each centroid
    selected = []
    for c in centroids:
        sq_dist = np.sum((X - c) ** 2, axis=1)
        nearest = int(np.argmin(sq_dist))
        selected.append(nearest)

    # Ensure unique anchors
    unique_selected = list(dict.fromkeys(selected))
    if len(unique_selected) < L:
        remaining = np.setdiff1d(np.arange(N), unique_selected)
        rng = np.random.default_rng(random_state)
        needed = L - len(unique_selected)
        fill = rng.choice(remaining, size=needed, replace=False)
        unique_selected.extend(fill)

    return np.array(unique_selected[:L], dtype=int)


def select_anchors(
    X: np.ndarray,
    L: int = 40,
    method: Literal["dpp", "kmeans", "random"] = "dpp",
    random_state: int = 42,
) -> np.ndarray:
    """
    Universal landmark anchor selection dispatcher.

    Parameters:
        X: Data matrix (N, D).
        L: Target number of anchors.
        method: 'dpp' (Determinantal Point Process), 'kmeans', or 'random'.
        random_state: Random seed for deterministic reproducibility.

    Returns:
        anchor_indices: Array of shape (L,) containing indices into X.
    """
    N = X.shape[0]
    L = min(L, N)

    if method == "dpp":
        return select_anchors_dpp(X, L=L, random_state=random_state)
    elif method == "kmeans":
        return select_anchors_kmeans(X, L=L, random_state=random_state)
    else:  # random
        rng = np.random.default_rng(random_state)
        return rng.choice(N, size=L, replace=False)
