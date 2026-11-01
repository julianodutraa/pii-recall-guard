"""Full, reproducible evaluation. Writes eval/results.json.

Usage:  python -m eval.run_eval
"""
from __future__ import annotations

import json
import random
import sys
from collections import Counter
from pathlib import Path

import numpy as np

from piiguard import conformal as C
from piiguard.pipeline import fit_guard, labels, p_pii, pii_truth, straddles
from piiguard.synth import CLASSES, CLASS_INDEX, PII_CLASSES, has_visible_leak, make_dataset

HERE = Path(__file__).parent
SEED = 20261101
N_TRAIN, N_CAL_SETS, N_CAL_RISK, N_TEST = 60, 40, 50, 100
ALPHAS = [0.10, 0.05, 0.02]
REPEATS = 300


def r4(x):
    return None if x is None else round(float(x), 4)


def main() -> dict:
    train = make_dataset(N_TRAIN, SEED + 1)
    cal_sets = make_dataset(N_CAL_SETS, SEED + 2)
    cal_risk = make_dataset(N_CAL_RISK, SEED + 3)
    test = make_dataset(N_TEST, SEED + 4)
    shifted = make_dataset(N_TEST, SEED + 5, leak_range=(0.02, 0.08))
    res: dict = {"seed": SEED, "sizes": {
        "classes": len(CLASSES), "train": len(train), "cal_sets": len(cal_sets),
        "cal_risk": len(cal_risk), "test": len(test), "shifted_test": len(shifted)}}

    guard = fit_guard(train, cal_sets, cal_risk, alpha_sets=0.05, alpha_fnr=0.05)
    model = guard.model

    # 1. point classifier
    p_cs, p_cr, p_te = guard.proba(cal_sets), guard.proba(cal_risk), guard.proba(test)
    y_cs, y_te = labels(cal_sets), labels(test)
    pred = p_te.argmax(1)
    errs = Counter((CLASSES[a], CLASSES[b]) for a, b in zip(y_te, pred) if a != b)
    pii_te = pii_truth(test)
    argmax_pii = np.array([CLASSES[k] in PII_CLASSES for k in pred])
    res["classifier"] = {
        "test_accuracy": r4((pred == y_te).mean()),
        "top_errors": [[a, b, n] for (a, b), n in errs.most_common(8)],
        "argmax_rule_fnr": r4(np.mean(~argmax_pii[pii_te])),
        "argmax_rule_fpr": r4(np.mean(argmax_pii[~pii_te])),
        "argmax_rule_fnr_free_text_pii": r4(np.mean(
            ~argmax_pii[y_te == CLASS_INDEX["free_text_pii"]])),
    }

    # 2. marginal versus Mondrian prediction sets
    sets_res = []
    for a in ALPHAS:
        q = C.lac_threshold(p_cs, y_cs, a)
        S = C.lac_sets(p_te, q)
        qm = C.mondrian_thresholds(p_cs, y_cs, a, len(CLASSES))
        M = C.mondrian_sets(p_te, qm)
        pcm, pcmm = C.per_class_coverage(S, y_te, len(CLASSES)), C.per_class_coverage(M, y_te, len(CLASSES))
        k = CLASS_INDEX["free_text_pii"]
        sets_res.append({
            "alpha": a,
            "marginal": {"coverage": r4(C.coverage(S, y_te)), "mean_size": r4(S.sum(1).mean()),
                         "empty_rate": r4((S.sum(1) == 0).mean()),
                         "worst_class": CLASSES[int(np.nanargmin(pcm))],
                         "worst_class_coverage": r4(np.nanmin(pcm)),
                         "free_text_pii_coverage": r4(pcm[k])},
            "mondrian": {"coverage": r4(C.coverage(M, y_te)), "mean_size": r4(M.sum(1).mean()),
                         "empty_rate": r4((M.sum(1) == 0).mean()),
                         "worst_class": CLASSES[int(np.nanargmin(pcmm))],
                         "worst_class_coverage": r4(np.nanmin(pcmm)),
                         "free_text_pii_coverage": r4(pcmm[k]),
                         "straddle_rate": r4(straddles(M).mean())},
        })
    res["prediction_sets"] = sets_res

    # 3. why free text is hard: is the leak even visible in the sample?
    ftp = [c for c in test if c.label == "free_text_pii"]
    res["free_text_visibility"] = {
        "columns": len(ftp),
        "visible_in_original_sample": r4(np.mean([has_visible_leak(c.values) for c in ftp])),
        "visible_after_60_rows": r4(np.mean([has_visible_leak(c.resample(max(60, 4 * len(c.values))).values)
                                             for c in ftp])),
        "mean_original_rows": r4(np.mean([len(c.values) for c in ftp])),
    }

    # 4. conformal risk control on the quarantine decision, one stage and two stage
    s1_cr, s1_te = p_pii(p_cr), p_pii(p_te)
    s2_cr, esc_cr = guard.stage2_scores(cal_risk)
    s2_te, esc_te = guard.stage2_scores(test)
    pii_cr = pii_truth(cal_risk)
    crc = []
    for a in ALPHAS:
        row = {"alpha": a}
        for name, sc, st in (("one_stage", s1_cr, s1_te), ("two_stage", s2_cr, s2_te)):
            lam = C.crc_threshold(sc, pii_cr, a)
            fnr, fpr = C.fnr_fpr(st, pii_te, lam)
            ft = np.array([c.label == "free_text" for c in test])
            row[name] = {"lambda": r4(lam), "test_fnr": r4(fnr), "test_fpr": r4(fpr),
                         "free_text_quarantined": r4(np.mean(st[ft] >= lam))}
        crc.append(row)
    res["risk_control"] = crc
    res["escalation"] = {"test_escalated_rate": r4(esc_te.mean()),
                         "escalated_classes": Counter(CLASSES[k] for k in y_te[esc_te]).most_common(8)}

    # 5. repeated random re-splits of (cal_risk + test): is E[FNR] <= alpha really honored?
    pool_s1 = np.concatenate([s1_cr, s1_te]); pool_s2 = np.concatenate([s2_cr, s2_te])
    pool_pii = np.concatenate([pii_cr, pii_te])
    rng = np.random.default_rng(SEED)
    n_cal = len(cal_risk)
    rep = []
    for a in ALPHAS:
        out = {"alpha": a}
        for name, pool in (("one_stage", pool_s1), ("two_stage", pool_s2)):
            fnrs, fprs = [], []
            for _ in range(REPEATS):
                perm = rng.permutation(len(pool))
                ci, ti = perm[:n_cal], perm[n_cal:]
                lam = C.crc_threshold(pool[ci], pool_pii[ci], a)
                f, p = C.fnr_fpr(pool[ti], pool_pii[ti], lam)
                fnrs.append(f); fprs.append(p)
            out[name] = {"mean_fnr": r4(np.mean(fnrs)), "p90_fnr": r4(np.quantile(fnrs, 0.9)),
                         "share_runs_fnr_above_alpha": r4(np.mean(np.array(fnrs) > a)),
                         "mean_fpr": r4(np.mean(fprs))}
        fnrs, fprs = [], []
        for _ in range(REPEATS):
            perm = rng.permutation(len(pool_s2))
            ci, ti = perm[:n_cal], perm[n_cal:]
            lam = C.ucb_threshold(pool_s2[ci], pool_pii[ci], a, delta=0.1)
            f, p = C.fnr_fpr(pool_s2[ti], pool_pii[ti], lam)
            fnrs.append(f); fprs.append(p)
        out["two_stage_ucb_delta_0.1"] = {
            "mean_fnr": r4(np.mean(fnrs)), "p90_fnr": r4(np.quantile(fnrs, 0.9)),
            "share_runs_fnr_above_alpha": r4(np.mean(np.array(fnrs) > a)),
            "mean_fpr": r4(np.mean(fprs))}
        rep.append(out)
    res["repeated_splits"] = {"repeats": REPEATS, "rows": rep}

    # 6. distribution shift: leaks become rarer in production than in calibration
    s2_sh, _ = guard.stage2_scores(shifted)
    pii_sh = pii_truth(shifted)
    fnr_sh, fpr_sh = C.fnr_fpr(s2_sh, pii_sh, guard.lam)
    y_sh = labels(shifted)
    k = CLASS_INDEX["free_text_pii"]
    res["shift"] = {"leak_range_calibration": [0.08, 0.35], "leak_range_production": [0.02, 0.08],
                    "alpha": 0.05, "lambda": r4(guard.lam),
                    "fnr": r4(fnr_sh), "fpr": r4(fpr_sh),
                    "free_text_pii_recall": r4(np.mean(s2_sh[y_sh == k] >= guard.lam))}

    # 7. LLM second opinion on escalated columns (labels cached from a blind run)
    batch_path, labels_path = HERE / "llm_batch.json", HERE / "llm_labels.json"
    rr = random.Random(SEED)
    esc_idx = [int(i) for i in np.flatnonzero(esc_te)]
    rest_idx = [int(i) for i in np.flatnonzero(~esc_te)]
    rr.shuffle(esc_idx); rr.shuffle(rest_idx)
    chosen = sorted(esc_idx[:50] + rest_idx[:50])
    escalated_ids = {f"c{i:04d}" for i in esc_idx[:50]}
    batch = [{"id": f"c{i:04d}", "values": test[i].values} for i in chosen]
    batch_path.write_text("[\n" + ",\n".join(json.dumps(b, ensure_ascii=False) for b in batch) + "\n]\n")
    if labels_path.exists():
        llm = json.loads(labels_path.read_text())["labels"]
        truth = {f"c{i:04d}": test[i].label for i in chosen}
        clf = {f"c{i:04d}": CLASSES[int(pred[i])] for i in chosen}
        s2map = {f"c{i:04d}": bool(s2_te[i] >= guard.lam) for i in chosen}
        is_p = lambda lab: lab in PII_CLASSES
        res["llm_second_opinion"] = {}
        for stratum, ids in (("all", [b["id"] for b in batch if b["id"] in llm]),
                             ("escalated", [b["id"] for b in batch if b["id"] in llm and b["id"] in escalated_ids]),
                             ("not_escalated", [b["id"] for b in batch if b["id"] in llm and b["id"] not in escalated_ids])):
            res["llm_second_opinion"][stratum] = {
                "columns": len(ids),
                "llm_class_accuracy": r4(np.mean([llm[i] == truth[i] for i in ids])),
                "classifier_class_accuracy": r4(np.mean([clf[i] == truth[i] for i in ids])),
                "llm_pii_fnr": r4(np.mean([not is_p(llm[i]) for i in ids if is_p(truth[i])])),
                "llm_pii_fpr": r4(np.mean([is_p(llm[i]) for i in ids if not is_p(truth[i])])),
                "crc_gate_pii_fnr": r4(np.mean([not s2map[i] for i in ids if is_p(truth[i])])),
                "crc_gate_pii_fpr": r4(np.mean([s2map[i] for i in ids if not is_p(truth[i])])),
                "llm_misses_with_no_visible_leak": int(sum(
                    1 for i in ids if is_p(truth[i]) and not is_p(llm[i])
                    and not has_visible_leak(test[int(i[1:])].values))),
                "llm_misses_total": int(sum(1 for i in ids if is_p(truth[i]) and not is_p(llm[i]))),
                "pii_columns": int(sum(is_p(truth[i]) for i in ids)),
            }
    out_path = HERE / "results.json"
    out_path.write_text(json.dumps(res, indent=2))
    return res


if __name__ == "__main__":
    json.dump(main(), sys.stdout, indent=2)
