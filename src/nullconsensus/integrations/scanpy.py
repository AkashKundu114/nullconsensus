"""
Scanpy and AnnData Native Integration Module for NullConsensus.
Provides the standard scanpy-style API: nullconsensus.tl.cluster(adata).
"""

from typing import Any

import numpy as np
import scipy.sparse as sp


def cluster(
    adata: Any,
    n_clusters: int = 8,
    key_added: str = "nullconsensus",
    obsm_key: str = "X_nullconsensus",
    use_rep: str | None = None,
    mode: str = "auto",
    anchor_method: str = "dpp",
    n_anchors: int = 40,
    n_base_models: int = 25,
    device: str = "auto",
    copy: bool = False,
    random_state: int = 42,
    **kwargs: Any,
) -> Any:
    """
    Performs AutoNullConsensus consensus clustering on an AnnData object.

    Follows standard Scanpy API conventions (e.g. sc.tl.leiden):
    Writes cluster categorical labels to adata.obs[key_added] and
    spectral coordinates to adata.obsm[obsm_key].

    Parameters:
        adata: AnnData object.
        n_clusters: Target number of cell-type clusters. Default: 8.
        key_added: adata.obs key where cluster labels are stored. Default: 'nullconsensus'.
        obsm_key: adata.obsm key where spectral embedding is stored. Default: 'X_nullconsensus'.
        use_rep: Representation in adata.obsm to use (e.g., 'X_pca'). If None, uses adata.X.
        mode: Gating mode ('auto', 'hurdlegate', 'uniform', 'nullgate'). Default: 'auto'.
        anchor_method: Landmark anchor selection ('dpp', 'kmeans', 'random'). Default: 'dpp'.
        n_anchors: Number of landmark anchors L. Default: 40.
        n_base_models: Number of ensemble base partitions M. Default: 25.
        device: Computational device ('auto', 'cpu', 'cuda'). Default: 'auto'.
        copy: If True, returns a modified copy of adata. If False, updates in place.
        random_state: Random state for deterministic reproducibility.

    Returns:
        AnnData if copy=True, else None.
    """
    import pandas as pd

    from nullconsensus.api import AutoNullConsensus

    adata_obj = adata.copy() if copy else adata

    # 1. Extract data representation
    if use_rep is not None:
        if use_rep not in adata_obj.obsm:
            raise KeyError(f"Representation '{use_rep}' not found in adata.obsm.")
        X = adata_obj.obsm[use_rep]
    else:
        X = adata_obj.X

    if sp.issparse(X):
        X = X.toarray()
    X = np.asarray(X, dtype=float)

    # 2. Fit AutoNullConsensus
    model = AutoNullConsensus(
        n_clusters=n_clusters,
        n_base_models=n_base_models,
        n_anchors=n_anchors,
        mode=mode,
        anchor_method=anchor_method,
        device=device,
        random_state=random_state,
        **kwargs,
    )
    labels = model.fit_predict(X)

    # 3. Store results in AnnData following Scanpy conventions
    adata_obj.obs[key_added] = pd.Categorical(labels.astype(str))
    if model.embedding_ is not None:
        adata_obj.obsm[obsm_key] = model.embedding_

    # Store provenance metadata in uns
    adata_obj.uns[key_added] = {
        "params": {
            "n_clusters": n_clusters,
            "mode": model.mode_,
            "anchor_method": model.anchor_method,
            "n_anchors": model.n_anchors,
            "n_base_models": model.n_base_models,
            "device": model.device,
            "random_state": random_state,
        },
        "profile": {
            "zero_fraction": model.profile_.zero_fraction if model.profile_ else None,
            "d_over_n": model.profile_.d_over_n if model.profile_ else None,
            "intrinsic_dim": model.profile_.intrinsic_dim if model.profile_ else None,
            "diagnostic_summary": model.profile_.diagnostic_summary if model.profile_ else None,
        },
    }

    return adata_obj if copy else None
