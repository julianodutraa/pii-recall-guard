"""Distribution-free uncertainty for column tagging.

Three tools, all finite-sample valid under exchangeability of calibration
and test columns:

1. Split conformal prediction sets with the LAC score s(x, y) = 1 - p_y(x)
   (Sadinle, Lei and Wasserman 2019). Marginal guarantee:
   P(y_test in C(x_test)) >= 1 - alpha.
2. Mondrian (class-conditional) conformal: one quantile per true class, which
   upgrades the guarantee to P(y in C(x) | y = k) >= 1 - alpha for every k.
3. Conformal risk control (Angelopoulos, Bates, Fisch, Lei and Schuster 2022)
   for a single binary decision, "quarantine this column as PII", with a
   bounded monotone loss. Here the loss is a missed PII column, and the
   guarantee is E[FNR on PII columns] <= alpha.
4. A high-probability variant in the spirit of risk-controlling prediction
   sets (Bates, Angelopoulos, Lei, Malik and Jordan 2021): pick lambda so
   that an exact Clopper-Pearson upper bound on the miss rate stays below
   alpha, which gives P(FNR > alpha) <= delta instead of a bound in
   expectation only.
"""
from __future__ import annotations

import math

import numpy as np


def conformal_quantile(scores: np.ndarray, alpha: float) -> float:
    """The ceil((n+1)(1-alpha))/n empirical quantile, +inf if it does not exist."""
    n = len(scores)
    k = math.ceil((n + 1) * (1 - alpha))
    if k > n:
        return float("inf")
    return float(np.sort(scores)[k - 1])


def lac_threshold(p_cal: np.ndarray, y_cal: np.ndarray, alpha: float) -> float:
    scores = 1.0 - p_cal[np.arange(len(y_cal)), y_cal]
    return conformal_quantile(scores, alpha)


def lac_sets(p: np.ndarray, qhat: float) -> np.ndarray:
    """Boolean matrix (n, K): class k is in the set iff 1 - p_k <= qhat."""
    return (1.0 - p) <= qhat


def mondrian_thresholds(p_cal: np.ndarray, y_cal: np.ndarray, alpha: float,
                        n_classes: int) -> np.ndarray:
    q = np.empty(n_classes)
    for k in range(n_classes):
        mask = y_cal == k
        q[k] = conformal_quantile(1.0 - p_cal[mask, k], alpha)
    return q


def mondrian_sets(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    return (1.0 - p) <= q[None, :]


def coverage(sets: np.ndarray, y: np.ndarray) -> float:
    return float(sets[np.arange(len(y)), y].mean())


def per_class_coverage(sets: np.ndarray, y: np.ndarray, n_classes: int) -> np.ndarray:
    hit = sets[np.arange(len(y)), y]
    return np.array([hit[y == k].mean() if np.any(y == k) else np.nan
                     for k in range(n_classes)])


def crc_threshold(p_pii_cal: np.ndarray, is_pii_cal: np.ndarray, alpha: float,
                  grid: np.ndarray | None = None) -> float:
    """Largest lambda such that flagging p_pii >= lambda controls FNR at alpha.

    Restricting to the n PII calibration columns, the loss of column i at
    threshold lambda is L_i(lambda) = 1{p_pii_i < lambda}, which is bounded
    by B = 1 and non-decreasing in lambda. CRC picks
        lambda_hat = sup { lambda : n/(n+1) * Rhat_n(lambda) + 1/(n+1) <= alpha }
    and guarantees E[L_test(lambda_hat)] <= alpha. If no lambda qualifies
    (alpha too small for n), lambda = 0 flags everything.
    """
    s = p_pii_cal[is_pii_cal]
    n = len(s)
    if grid is None:
        grid = np.unique(np.concatenate([[0.0], s, [1.0]]))
    best = 0.0
    for lam in np.sort(grid):
        risk = np.mean(s < lam)
        if n / (n + 1) * risk + 1 / (n + 1) <= alpha:
            best = float(lam)
        else:
            break  # risk is monotone in lambda
    return best


def fnr_fpr(p_pii: np.ndarray, is_pii: np.ndarray, lam: float) -> tuple[float, float]:
    flag = p_pii >= lam
    fnr = float(np.mean(~flag[is_pii])) if is_pii.any() else float("nan")
    fpr = float(np.mean(flag[~is_pii])) if (~is_pii).any() else float("nan")
    return fnr, fpr


def _binom_cdf(k: int, n: int, p: float) -> float:
    if p <= 0.0:
        return 1.0
    if p >= 1.0:
        return 1.0 if k >= n else 0.0
    lp, lq = math.log(p), math.log1p(-p)
    terms = [math.lgamma(n + 1) - math.lgamma(i + 1) - math.lgamma(n - i + 1) + i * lp + (n - i) * lq
             for i in range(k + 1)]
    m = max(terms)
    return min(1.0, math.exp(m) * sum(math.exp(t - m) for t in terms))


def clopper_pearson_upper(k: int, n: int, delta: float) -> float:
    """One-sided exact upper confidence bound for a binomial proportion."""
    if k >= n:
        return 1.0
    lo, hi = k / n, 1.0
    for _ in range(60):
        mid = (lo + hi) / 2
        if _binom_cdf(k, n, mid) > delta:
            lo = mid
        else:
            hi = mid
    return hi


def ucb_threshold(p_pii_cal: np.ndarray, is_pii_cal: np.ndarray, alpha: float,
                  delta: float = 0.1) -> float:
    """Largest lambda whose Clopper-Pearson upper bound on FNR is <= alpha.

    With a single nested family scanned from small to large lambda and a
    fixed stopping rule, this is fixed-sequence testing, so no multiplicity
    correction is needed: P(FNR_true(lambda_hat) > alpha) <= delta.
    """
    s = np.sort(p_pii_cal[is_pii_cal])
    n = len(s)
    grid = np.unique(np.concatenate([[0.0], s, [1.0]]))
    best = 0.0
    for lam in grid:
        k = int(np.searchsorted(s, lam, side="left"))  # count of s < lam
        if clopper_pearson_upper(k, n, delta) <= alpha:
            best = float(lam)
        else:
            break
    return best
