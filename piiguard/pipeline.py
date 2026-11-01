"""End to end tagging pipeline: classifier, conformal sets, escalation, CRC gate."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import conformal as C
from .features import featurize_many
from .model import SoftmaxRegression
from .synth import CLASS_INDEX, CLASSES, IS_PII, Column

PII_MASK = np.array(IS_PII)


def labels(cols: list[Column]) -> np.ndarray:
    return np.array([CLASS_INDEX[c.label] for c in cols])


def pii_truth(cols: list[Column]) -> np.ndarray:
    return np.array([c.is_pii for c in cols])


def p_pii(p: np.ndarray) -> np.ndarray:
    """Probability mass on PII classes."""
    return p[:, PII_MASK].sum(axis=1)


def straddles(sets: np.ndarray) -> np.ndarray:
    """A set is ambiguous for governance if it mixes PII and non-PII classes."""
    return sets[:, PII_MASK].any(axis=1) & sets[:, ~PII_MASK].any(axis=1)


@dataclass
class Guard:
    model: SoftmaxRegression
    q_mondrian: np.ndarray
    lam: float
    escalate_rows: int
    alpha_sets: float
    alpha_fnr: float

    def proba(self, cols: list[Column]) -> np.ndarray:
        return self.model.predict_proba(featurize_many(cols))

    def sets(self, p: np.ndarray) -> np.ndarray:
        return C.mondrian_sets(p, self.q_mondrian)

    def stage2_scores(self, cols: list[Column]) -> tuple[np.ndarray, np.ndarray]:
        """p_pii after escalating ambiguous columns to a larger sample."""
        p = self.proba(cols)
        esc = straddles(self.sets(p))
        score = p_pii(p)
        if esc.any():
            bigger = [cols[i].resample(max(self.escalate_rows, 4 * len(cols[i].values)))
                      for i in np.flatnonzero(esc)]
            score[esc] = p_pii(self.proba(bigger))
        return score, esc

    def decide(self, cols: list[Column]) -> dict:
        score, esc = self.stage2_scores(cols)
        return {"p_pii": score, "escalated": esc, "quarantine": score >= self.lam}


def fit_guard(train: list[Column], cal_sets: list[Column], cal_risk: list[Column],
              alpha_sets: float = 0.05, alpha_fnr: float = 0.05,
              escalate_rows: int = 60, seed: int = 0) -> Guard:
    """Fit model on `train`, Mondrian thresholds on `cal_sets`, CRC lambda on `cal_risk`.

    The two calibration sets are disjoint on purpose: the escalation rule is a
    function of the Mondrian thresholds, so the CRC step must see data that
    did not choose that rule, otherwise exchangeability between the risk
    calibration columns and future columns is broken.
    """
    model = SoftmaxRegression(len(CLASSES), seed=seed).fit(featurize_many(train), labels(train))
    p_cs = model.predict_proba(featurize_many(cal_sets))
    q = C.mondrian_thresholds(p_cs, labels(cal_sets), alpha_sets, len(CLASSES))
    g = Guard(model, q, 0.0, escalate_rows, alpha_sets, alpha_fnr)
    score, _ = g.stage2_scores(cal_risk)
    g.lam = C.crc_threshold(score, pii_truth(cal_risk), alpha_fnr)
    return g
