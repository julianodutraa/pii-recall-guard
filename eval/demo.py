"""Small demo: fit the guard and explain its decision on a handful of columns.

Usage:  python -m eval.demo
"""
from __future__ import annotations

import numpy as np

from piiguard.pipeline import fit_guard, p_pii
from piiguard.synth import CLASSES, make_dataset

SEED = 20261101


def main() -> None:
    guard = fit_guard(make_dataset(60, SEED + 1), make_dataset(40, SEED + 2),
                      make_dataset(50, SEED + 3), alpha_sets=0.05, alpha_fnr=0.05)
    test = make_dataset(100, SEED + 4)
    p = guard.proba(test)
    sets = guard.sets(p)
    out = guard.decide(test)
    print(f"lambda_hat = {guard.lam:.4f}  (quarantine if P(PII) >= lambda_hat, target FNR 5%)\n")
    shown = set()
    picks = []
    for i, c in enumerate(test):  # one escalated column per label, plus a few clear ones
        if out["escalated"][i] and c.label not in shown:
            picks.append(i); shown.add(c.label)
    for lab in ("email", "national_id", "order_id", "birth_date"):
        picks.append(next(i for i, c in enumerate(test) if c.label == lab))
    for i in picks:
        c = test[i]
        preview = [v for v in c.values if v][:2]
        members = [CLASSES[k] for k in np.flatnonzero(sets[i])]
        print(f"column #{i}  truth={c.label}  rows={len(c.values)}")
        print(f"  sample     : {preview}")
        print(f"  set (95%)  : {members}")
        print(f"  P(PII)     : stage1={p_pii(p[i:i+1])[0]:.3f}  final={out['p_pii'][i]:.3f}"
              f"  escalated={bool(out['escalated'][i])}")
        print(f"  decision   : {'QUARANTINE' if out['quarantine'][i] else 'pass'}\n")


if __name__ == "__main__":
    main()
