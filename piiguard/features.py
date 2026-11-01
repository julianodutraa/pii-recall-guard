"""Column featurizer: hashed character n-grams of values and of value shapes,
plus a small set of distributional statistics.

A column is represented by the mean of its per-value feature vectors, which
makes the representation invariant to sample size and row order. Hashing uses
zlib.crc32 so features are stable across interpreter runs (Python's built-in
`hash` is salted per process).
"""
from __future__ import annotations

import re
import zlib

import numpy as np

N_HASH = 1024
_DATE_ISO = re.compile(r"^(\d{4})-(\d{2})-(\d{2})$")
_DATE_BR = re.compile(r"^(\d{2})/(\d{2})/(\d{4})$")
_EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+")
_PHONEISH = re.compile(r"\(?\+?\d[\d ()]{6,}[-.]\d{4}")
N_STATS = 14
DIM = 2 * N_HASH + N_STATS


def shape(value: str) -> str:
    """Map characters to classes: digit to 9, upper to A, lower to a."""
    out = []
    for ch in value:
        if ch.isdigit():
            out.append("9")
        elif ch.isalpha():
            out.append("A" if ch.isupper() else "a")
        else:
            out.append(ch)
    # collapse runs so that 'aaaaaa' and 'aaa' share most n-grams
    return re.sub(r"(.)\1{2,}", r"\1\1\1", "".join(out))


def _ngrams(s: str, ns=(1, 2, 3)):
    s = f"^{s}$"
    for n in ns:
        for i in range(len(s) - n + 1):
            yield s[i:i + n]


def _hash(token: str, salt: str) -> int:
    return zlib.crc32((salt + token).encode("utf-8")) % N_HASH


def _year(value: str):
    m = _DATE_ISO.match(value)
    if m:
        return int(m.group(1))
    m = _DATE_BR.match(value)
    if m:
        return int(m.group(3))
    return None


def featurize(values: list[str]) -> np.ndarray:
    """Return a DIM-dimensional vector for one column sample."""
    raw = np.zeros(N_HASH)
    shp = np.zeros(N_HASH)
    non_empty = [v for v in values if v.strip()]
    for v in non_empty:
        grams = list(_ngrams(v.lower()))
        for g in grams:
            raw[_hash(g, "r")] += 1.0 / len(grams)
        sgrams = list(_ngrams(shape(v)))
        for g in sgrams:
            shp[_hash(g, "s")] += 1.0 / len(sgrams)
    k = max(len(non_empty), 1)
    raw /= k
    shp /= k

    lens = np.array([len(v) for v in non_empty] or [0], dtype=float)
    joined = "".join(non_empty) or " "
    years = [y for y in (_year(v) for v in non_empty) if y is not None]
    yarr = np.array(years, dtype=float) if years else np.array([2000.0])
    distinct = len(set(non_empty)) / k
    stats = np.array([
        1.0 - len(non_empty) / max(len(values), 1),          # null or blank rate
        np.log1p(lens.mean()) / 4.0,
        np.log1p(lens.std()) / 4.0,
        sum(c.isdigit() for c in joined) / len(joined),
        sum(c.isalpha() for c in joined) / len(joined),
        joined.count(" ") / len(joined),
        sum("@" in v for v in non_empty) / k,
        sum(bool(_EMAIL.search(v)) for v in non_empty) / k,
        sum(bool(_PHONEISH.search(v)) for v in non_empty) / k,
        len(years) / k,
        (yarr.mean() - 1980.0) / 40.0,
        yarr.std() / 40.0,
        (yarr.min() - 1980.0) / 40.0,
        distinct,
    ])
    return np.concatenate([raw, shp, stats])


def featurize_many(columns) -> np.ndarray:
    return np.stack([featurize(c.values) for c in columns])
