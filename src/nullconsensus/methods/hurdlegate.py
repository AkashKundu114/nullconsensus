"""
Hurdle-Bimodality Gating (HurdleGate) for zero-inflated count data (scRNA-seq / sparse text).
Candidate 2 Count-Aware Module for NullConsensus.
"""

import numpy as np
import scipy.stats as stats
from typing import Optional


class HurdleGateFeatureSelector:
    """
    Two-part Hurdle Decomposition Feature Selector (Candidate 2: HurdleGate).
    
    Specifically engineered for zero-inflated high-dimensional modalities (e.g. scRNA-seq,
    metagenomics, sparse document representations) where extreme dropout spikes at zero
    distort global skewness and kurtosis moments, producing false-positive bimodality.
    
    Decomposition:
    Part 1: Evaluates continuous multimodality and Fano dispersion on the positive support:
            x_d^+ = {x_{id} | x_{id} > 0}.
    Part 2: Evaluates Bernoulli dropout dispersion and zero-inflation excess:
            DropoutExcess_d = p_0 - exp(-mean(x_d)).
            
    Parameters:
        min_positive_samples: Minimum number of expressing cells/samples required to evaluate
                              positive support bimodality (default: 5).
        random_state: Random state seed.
    """
    def __init__(self, min_positive_samples: int = 5, random_state: int = 42):
        self.min_positive_samples = min_positive_samples
        self.random_state = random_state
        
        # Fitted attributes
        self.hurdle_scores_: Optional[np.ndarray] = None
        self.sampling_probs_: Optional[np.ndarray] = None
        self.positive_bc_: Optional[np.ndarray] = None
        self.dropout_entropy_: Optional[np.ndarray] = None
        self.dropout_excess_: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray, temperature: float = 1.0) -> "HurdleGateFeatureSelector":
        """
        Fits two-part hurdle model and generates saliency weights and sampling probabilities.
        """
        X = np.asarray(X, dtype=float)
        N, D = X.shape
        
        self.positive_bc_ = np.zeros(D)
        self.dropout_entropy_ = np.zeros(D)
        self.dropout_excess_ = np.zeros(D)
        scores = np.zeros(D)
        
        for d in range(D):
            col = X[:, d]
            pos_mask = (col > 0.0)
            n_pos = int(np.sum(pos_mask))
            dropout_frac = 1.0 - (n_pos / N)
            mu = float(np.mean(col))
            
            # Part 1: Bernoulli dropout entropy and zero-inflation excess
            # Binary entropy is maximized at p=0.5 (ideal marker gene separating two groups)
            if 0.001 < dropout_frac < 0.999:
                entropy = -(
                    dropout_frac * np.log2(dropout_frac) +
                    (1.0 - dropout_frac) * np.log2(1.0 - dropout_frac)
                )
            else:
                entropy = 0.0
                
            excess = max(0.0, dropout_frac - np.exp(-mu))
            self.dropout_entropy_[d] = entropy
            self.dropout_excess_[d] = excess
            
            # Part 2: Positive support continuous multimodality & dispersion
            if n_pos >= self.min_positive_samples:
                pos_vals = col[pos_mask]
                var_pos = float(np.var(pos_vals, ddof=1))
                mean_pos = float(np.mean(pos_vals))
                fano_pos = var_pos / (mean_pos + 1e-4) if mean_pos > 0 else 1.0
                
                if var_pos > 1e-12:
                    skew_pos = float(stats.skew(pos_vals, bias=False))
                    kurt_pos = float(stats.kurtosis(pos_vals, bias=False, fisher=True))
                    denom = kurt_pos + 3.0
                    bc_pos = (skew_pos ** 2 + 1.0) / denom if denom > 0.0 else 0.556
                else:
                    # Constant positive expression (e.g. discrete on/off marker)
                    bc_pos = 0.556
            else:
                fano_pos = 1.0
                bc_pos = 0.0
                
            self.positive_bc_[d] = np.clip(bc_pos, 0.0, 1.0)
            
            # Combined Hurdle Saliency
            # S = Positive_Dispersion * Positive_BC + (Entropy + Excess)
            scores[d] = (fano_pos * self.positive_bc_[d]) + (entropy + excess)

        self.hurdle_scores_ = np.maximum(0.0, scores)
        
        # Softmax sampling probabilities
        t = max(1e-3, float(temperature))
        scaled_s = self.hurdle_scores_ / t
        exp_s = np.exp(scaled_s - np.max(scaled_s))
        sum_exp = np.sum(exp_s)
        if sum_exp > 0:
            self.sampling_probs_ = exp_s / sum_exp
        else:
            self.sampling_probs_ = np.ones(D) / D
            
        return self
