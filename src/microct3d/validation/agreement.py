"""Agreement between two methods: Bland-Altman and Lin's concordance coefficient."""
from __future__ import annotations

import numpy as np


def lin_ccc(a, b) -> float:
    """Lin's concordance correlation coefficient. Unlike Pearson's r it penalises
    bias and differences of scale."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 3:
        return float("nan")
    cov = ((a - a.mean()) * (b - b.mean())).mean()
    den = a.var() + b.var() + (a.mean() - b.mean()) ** 2
    return 2 * cov / den if den else float("nan")


def bland_altman(reference, test) -> dict:
    """Bland-Altman in PERCENT of the pair mean: bias, 95 % limits of agreement
    (bias ± 1.96 SD) and the 95 % confidence interval of the bias.

    With few pairs a paired test flags biases of 0.01 %; what matters is the size of
    the bias and of the limits, not the p-value.
    """
    from scipy import stats
    a, b = np.asarray(reference, float), np.asarray(test, float)
    mean = (a + b) / 2.0
    with np.errstate(divide="ignore", invalid="ignore"):
        d = np.where(mean != 0, (b - a) / mean * 100.0, 0.0)
    n = len(d)
    bias = float(d.mean())
    sd = float(d.std(ddof=1)) if n > 1 else 0.0
    half = float(stats.t.ppf(0.975, n - 1) * sd / np.sqrt(n)) if n > 1 and sd > 0 else 0.0
    return {"n": n, "bias_pct": bias, "loa_low_pct": bias - 1.96 * sd,
            "loa_high_pct": bias + 1.96 * sd, "bias_ci_low_pct": bias - half,
            "bias_ci_high_pct": bias + half, "ccc": lin_ccc(a, b)}
