import math
import random

import numpy as np
import pytest

from piiguard import conformal as C
from piiguard.features import DIM, featurize, shape
from piiguard.pipeline import fit_guard, pii_truth, straddles
from piiguard.synth import (CLASSES, PII_CLASSES, cpf_check_digits, has_visible_leak,
                            luhn_valid, make_column, make_dataset)


# synthetic data ---------------------------------------------------------------

def test_card_numbers_pass_luhn():
    col = make_column("card_number", 7, n_rows=50)
    assert all(luhn_valid(v) for v in col.values if v.replace(" ", "").isdigit())


def test_national_ids_pass_checksum():
    col = make_column("national_id", 11, n_rows=50)
    for v in col.values:
        d = "".join(ch for ch in v if ch.isdigit())
        if len(d) == 11:
            assert cpf_check_digits(d[:9]) == d[9:]


def test_resample_keeps_prefix():
    col = make_column("free_text_pii", 123)
    big = col.resample(80)
    assert big.values[:len(col.values)] == col.values
    assert len(big.values) == 80


def test_dataset_is_deterministic_and_balanced():
    a, b = make_dataset(3, 5), make_dataset(3, 5)
    assert [c.values for c in a] == [c.values for c in b]
    assert sorted({c.label for c in a}) == sorted(CLASSES)
    assert len(a) == 3 * len(CLASSES)


def test_leak_detector():
    assert has_visible_leak(["ok", "send the receipt to ana.lima@gmail.com"])
    assert has_visible_leak(["please call me at (11) 98877-1234"])
    assert not has_visible_leak(["batch 4411 reprocessed after schema change"])


# features ---------------------------------------------------------------------

def test_shape_and_feature_dimension():
    assert shape("Ana-42") == "Aaa-99"
    v = featurize(["ana@x.com", "", "bob@y.org"])
    assert v.shape == (DIM,)
    assert np.allclose(v, featurize(["ana@x.com", "", "bob@y.org"]))
    assert v[-14] == pytest.approx(1 / 3)  # blank rate


# conformal --------------------------------------------------------------------

def test_conformal_quantile_rank():
    scores = np.arange(1, 20) / 20.0  # n = 19
    # ceil(20 * 0.9) = 18th smallest
    assert C.conformal_quantile(scores, 0.1) == pytest.approx(18 / 20)
    # alpha too small for n: ceil(20 * 0.99) = 20 > 19
    assert C.conformal_quantile(scores, 0.01) == math.inf


def test_split_conformal_marginal_coverage_on_average():
    rng = np.random.default_rng(0)
    covs = []
    for _ in range(400):
        p = rng.dirichlet(np.ones(5) * 0.5, size=300)
        y = np.array([rng.choice(5, p=row) for row in p])  # calibrated scores
        q = C.lac_threshold(p[:100], y[:100], 0.1)
        covs.append(C.coverage(C.lac_sets(p[100:], q), y[100:]))
    assert np.mean(covs) >= 0.89


def test_mondrian_infeasible_alpha_gives_full_sets():
    p = np.full((10, 3), 1 / 3)
    y = np.array([0, 1, 2] * 3 + [0])
    q = C.mondrian_thresholds(p, y, 0.05, 3)  # 3 or 4 points per class
    assert np.all(np.isinf(q))
    assert C.mondrian_sets(p, q).all()


def test_crc_threshold_satisfies_bound_and_is_maximal():
    rng = np.random.default_rng(1)
    s = rng.uniform(size=400)
    is_pii = np.ones(400, dtype=bool)
    lam = C.crc_threshold(s, is_pii, 0.05)
    n = 400
    risk = np.mean(s < lam)
    assert n / (n + 1) * risk + 1 / (n + 1) <= 0.05
    bigger = np.min(s[s > lam])
    assert n / (n + 1) * np.mean(s < bigger + 1e-12) + 1 / (n + 1) > 0.05


def test_crc_controls_expected_fnr():
    rng = np.random.default_rng(2)
    fnrs = []
    for _ in range(500):
        s = rng.beta(5, 1, size=600)
        lam = C.crc_threshold(s[:200], np.ones(200, bool), 0.1)
        fnrs.append(np.mean(s[200:] < lam))
    assert np.mean(fnrs) <= 0.1 + 0.005


def test_clopper_pearson_zero_successes_closed_form():
    assert C.clopper_pearson_upper(0, 100, 0.1) == pytest.approx(1 - 0.1 ** (1 / 100), rel=1e-6)


def test_ucb_is_more_conservative_than_crc():
    rng = np.random.default_rng(3)
    s = rng.beta(5, 1, size=500)
    m = np.ones(500, bool)
    assert C.ucb_threshold(s, m, 0.1, 0.1) <= C.crc_threshold(s, m, 0.1)


# end to end -------------------------------------------------------------------

@pytest.fixture(scope="module")
def small_guard():
    return fit_guard(make_dataset(25, 1), make_dataset(25, 2), make_dataset(25, 3),
                     alpha_sets=0.1, alpha_fnr=0.1)


def test_guard_quarantines_obvious_pii(small_guard):
    cols = [make_column(c, 1000 + i) for i, c in enumerate(["email", "card_number", "phone"])]
    out = small_guard.decide(cols)
    assert out["quarantine"].all()


def test_guard_end_to_end_recall(small_guard):
    test = make_dataset(20, 4)
    out = small_guard.decide(test)
    truth = pii_truth(test)
    fnr = np.mean(~out["quarantine"][truth])
    assert fnr <= 0.2  # loose: alpha 0.1 in expectation, small samples
    assert np.mean(out["quarantine"][~truth]) <= 0.2


def test_straddle_definition():
    k = len(CLASSES)
    s = np.zeros((2, k), bool)
    s[0, CLASSES.index("free_text")] = True
    s[0, CLASSES.index("free_text_pii")] = True
    s[1, CLASSES.index("email")] = True
    assert straddles(s).tolist() == [True, False]
    assert len(PII_CLASSES) == 9
