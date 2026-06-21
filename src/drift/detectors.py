"""Statistical drift detectors.

- ``psi_score``        — Population Stability Index for a numerical feature (quantile bins).
- ``categorical_psi``  — PSI over category frequencies for a categorical feature.
- ``ks_test``          — two-sample Kolmogorov–Smirnov test for a numerical feature.

PSI rule of thumb: < 0.1 no shift · 0.1–0.2 moderate · > 0.2 significant drift.
KS rule: p-value < 0.05 → distributions differ (drift).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

# Floor applied to bin/category proportions so log/ratio terms stay finite.
_EPSILON = 1e-6


def _psi_from_proportions(expected: np.ndarray, actual: np.ndarray) -> float:
    """PSI given two aligned proportion vectors. PSI = Σ (a-e)·ln(a/e)."""
    expected = np.clip(expected, _EPSILON, None)
    actual = np.clip(actual, _EPSILON, None)
    return float(np.sum((actual - expected) * np.log(actual / expected)))


def psi_score(expected, actual, bins: int = 10) -> float:
    """PSI for a *numerical* feature using quantile bin edges from ``expected``.

    Returns 0.0 when ``expected`` is constant (no meaningful bins).
    """
    expected = np.asarray(expected, dtype=float)
    actual = np.asarray(actual, dtype=float)
    if expected.size == 0 or actual.size == 0:
        return 0.0

    # Quantile edges from the reference, deduped, with open outer edges so any
    # out-of-range current values still fall into the end bins.
    edges = np.unique(np.quantile(expected, np.linspace(0, 1, bins + 1)))
    if edges.size < 2:
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf

    exp_counts, _ = np.histogram(expected, bins=edges)
    act_counts, _ = np.histogram(actual, bins=edges)
    exp_prop = exp_counts / exp_counts.sum()
    act_prop = act_counts / act_counts.sum()
    return _psi_from_proportions(exp_prop, act_prop)


def categorical_psi(expected, actual) -> float:
    """PSI for a *categorical* feature over the union of observed categories."""
    expected = pd.Series(expected).astype("object")
    actual = pd.Series(actual).astype("object")
    if expected.empty or actual.empty:
        return 0.0

    exp_prop = expected.value_counts(normalize=True)
    act_prop = actual.value_counts(normalize=True)
    categories = exp_prop.index.union(act_prop.index)

    exp_vec = exp_prop.reindex(categories, fill_value=0.0).to_numpy()
    act_vec = act_prop.reindex(categories, fill_value=0.0).to_numpy()
    return _psi_from_proportions(exp_vec, act_vec)


def ks_test(reference, current) -> tuple[float, float]:
    """Two-sample KS test. Returns (statistic, p_value).

    p_value < threshold (default 0.05) ⇒ the two samples differ ⇒ drift.
    """
    reference = np.asarray(reference, dtype=float)
    current = np.asarray(current, dtype=float)
    if reference.size == 0 or current.size == 0:
        return 0.0, 1.0
    result = ks_2samp(reference, current)
    return float(result.statistic), float(result.pvalue)
