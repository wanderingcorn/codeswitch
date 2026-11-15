"""Shared evaluation: switch PR-AUC/P/R/F1, duration macro-F1 / per-class recall, joint acc, per-head CE."""
import json

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, f1_score, precision_recall_curve, recall_score

from . import config as C

DUR_CLASSES = [1, 2, 3]


def head_outputs(sw_logits, du_logits, sw_gold, du_gold):
    """Tensors (N,2), (N,4), (N,), (N,) -> numpy dict of probabilities, hard predictions and per-sample CE.
    Duration prediction and duration CE only use classes 1..3 (class 0 = no switch is excluded)."""
    sw_logits, du_logits = sw_logits.float().cpu(), du_logits.float().cpu()
    sw_gold, du_gold = sw_gold.cpu(), du_gold.cpu()
    sw_ce = F.cross_entropy(sw_logits, sw_gold, reduction="none")          # unweighted, comparable across runs
    du_ce = torch.full_like(sw_ce, float("nan"))
    m = du_gold > 0                                                         # only real switches
    if m.any():
        du_ce[m] = F.cross_entropy(du_logits[m][:, 1:], du_gold[m] - 1, reduction="none")
    return {"sw_prob": sw_logits.softmax(-1)[:, 1].numpy(),
            "sw_pred": sw_logits.argmax(-1).numpy(),
            "du_pred": (du_logits[:, 1:].argmax(-1) + 1).numpy(),
            "sw_ce": sw_ce.numpy(), "du_ce": du_ce.numpy()}


def best_f1_threshold(sw_gold, sw_prob):
    """Threshold maximising switch (positive-class) F1. Returns (thr, f1). Predict switch if prob >= thr."""
    p, r, t = precision_recall_curve(sw_gold, sw_prob)
    f = 2 * p[:-1] * r[:-1] / np.maximum(p[:-1] + r[:-1], 1e-12)
    i = int(np.argmax(f))
    return float(t[i]), float(f[i])


def score(sw_pred, sw_gold, du_pred, du_gold, sw_prob=None, sw_ce=None, du_ce=None):
    """sw_prob / sw_ce / du_ce are optional (baselines B0 have no probabilities)."""
    sw_pred, sw_gold, du_pred, du_gold = map(np.asarray, (sw_pred, sw_gold, du_pred, du_gold))
    tp = int(((sw_pred == 1) & (sw_gold == 1)).sum())
    fp = int(((sw_pred == 1) & (sw_gold == 0)).sum())
    fn = int(((sw_pred == 0) & (sw_gold == 1)).sum())
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * p * r / (p + r) if p + r else 0.0

    nan = float("nan")
    has_both = 0 < sw_gold.sum() < len(sw_gold)
    pr_auc = float(average_precision_score(sw_gold, sw_prob)) if (sw_prob is not None and has_both) else nan

    f1max, thr = nan, nan
    if sw_prob is not None and has_both:
        thr, f1max = best_f1_threshold(sw_gold, sw_prob)

    m = du_gold > 0                                                         # duration: only gold switch positions
    if m.any():
        mac = f1_score(du_gold[m], du_pred[m], labels=DUR_CLASSES, average="macro", zero_division=0)
        rec = recall_score(du_gold[m], du_pred[m], labels=DUR_CLASSES, average=None, zero_division=0)
        acc = float((du_pred[m] == du_gold[m]).mean())
        mae = float(np.abs(du_pred[m] - du_gold[m]).mean())
    else:
        mac, rec, acc, mae = nan, [nan] * 3, nan, nan

    # joint: gold switch -> need switch predicted AND duration class right; gold no-switch -> need no-switch predicted
    joint = float(np.where(sw_gold == 1, (sw_pred == 1) & (du_pred == du_gold), sw_pred == 0).mean())

    out = {  # primary
        "switch_PR_AUC": round(pr_auc, 3), "dur_macro_F1": round(float(mac), 3),
        # auxiliary
        "switch_P": round(p, 3), "switch_R": round(r, 3), "switch_F1": round(f1, 3),
        "dur_R1": round(float(rec[0]), 3), "dur_R2": round(float(rec[1]), 3), "dur_R3": round(float(rec[2]), 3),
        "switch_F1max": round(f1max, 3), "switch_thr": round(thr, 4),
        "joint_acc": round(joint, 3), "dur_acc": round(acc, 3), "dur_MAE": round(mae, 3)}
    if sw_ce is not None:
        out["val_ce_sw"] = round(float(np.mean(sw_ce)), 4)
    if du_ce is not None:
        out["val_ce_dur"] = round(float(np.nanmean(du_ce)), 4) if m.any() else nan
    return out


def score_by_pair(o, sw_gold, du_gold, pairs, thr=None):
    """o = head_outputs(...) dict. Return {"ALL": ..., "<pair>": ...}.
    thr: if given, switch prediction = (prob >= thr) instead of argmax (0.5)."""
    sw_gold, du_gold, pairs = map(np.asarray, (sw_gold, du_gold, pairs))
    if thr is not None:
        o = dict(o, sw_pred=(o["sw_prob"] >= thr).astype(int))

    def sub(mask):
        return score(o["sw_pred"][mask], sw_gold[mask], o["du_pred"][mask], du_gold[mask],
                     sw_prob=o["sw_prob"][mask], sw_ce=o["sw_ce"][mask], du_ce=o["du_ce"][mask])

    out = {"ALL": sub(np.ones(len(pairs), dtype=bool))}
    for tag in C.PAIR_TAGS:
        m = pairs == tag
        if m.any():
            out[tag] = sub(m)
    return out


def print_scores(prefix, res):
    for k, v in res.items():
        print(f"{prefix} [{k:6}]", v)


def monitor_line(res):
    """Two-head monitor on the pooled set: big head (switch) PR-AUC / Anticipatory F1, small head (duration) Acc / Macro-F1."""
    a = res["ALL"]
    return (f"  [monitor] SW: PR-AUC {a['switch_PR_AUC']:.3f} | Antic-F1 {a['switch_F1']:.3f} "
            f"(max {a['switch_F1max']:.3f} @thr {a['switch_thr']:.3f}) || "
            f"DUR: Acc {a['dur_acc']:.3f} | MacroF1 {a['dur_macro_F1']:.3f}")


def pair_report(res):
    """Per-pair Anticipatory F1 (switch F1) and duration accuracy + mean / std across pairs (sample std, ddof=1)."""
    df = pd.DataFrame({k: {"Anticipatory_F1": v["switch_F1"], "dur_acc": v["dur_acc"]}
                       for k, v in res.items() if k != "ALL"}).T
    df.loc["MEAN"] = df.mean()
    df.loc["STD"] = df.iloc[:-1].std(ddof=1)
    if "ALL" in res:
        df.loc["POOLED"] = [res["ALL"]["switch_F1"], res["ALL"]["dur_acc"]]
    return df.round(4)


def results_table(results, metric="switch_F1"):
    """results: {model_name: {pair_or_ALL: score_dict}} -> DataFrame (rows = pairs, cols = models)."""
    df = pd.DataFrame({name: {k: v.get(metric, float("nan")) for k, v in res.items()} for name, res in results.items()})
    return df.reindex(["ALL"] + C.PAIR_TAGS)


def save_results(results, path="results.json"):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
