"""Naive baselines, trained JOINTLY on all selected language pairs (same setup as XLM-R / Qwen) and
evaluated on the pooled set ("ALL") and per language pair.

B0  prior:  always / never predict a switch; duration = most common class
B1  lookup: P(switch | current language, run length, trailing punctuation); duration = mode per language
B2  char n-gram TF-IDF + logistic regression on the last 6 characters (one model for all pairs)
"""
from collections import Counter, defaultdict

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from . import config as C
from .metrics import score


def _feat(e):                                              # (current language, run length so far, trailing punctuation)
    return (e["cur"], min(e["run"], 6), e["trail"])


def _gold(ev):
    return np.array([e["switch"] for e in ev]), np.array([e["duration"] for e in ev])


def _report(ev, sw_pred, du_pred, sw_prob=None):
    """{"ALL": ..., "<pair>": ...} like the neural models."""
    sw_g, du_g = _gold(ev)
    sw_pred, du_pred = np.asarray(sw_pred), np.asarray(du_pred)
    pairs = np.array([e["pair"] for e in ev])
    sw_prob = None if sw_prob is None else np.asarray(sw_prob)

    def sub(m):
        return score(sw_pred[m], sw_g[m], du_pred[m], du_g[m], sw_prob=None if sw_prob is None else sw_prob[m])

    out = {"ALL": sub(np.ones(len(ev), dtype=bool))}
    for tag in C.PAIR_TAGS:
        m = pairs == tag
        if m.any():
            out[tag] = sub(m)
    return out


def _du_mode(tr):
    return Counter(e["duration"] for e in tr if e["duration"] > 0).most_common(1)[0][0]


def _b0(tr, ev):
    n, mode = len(ev), _du_mode(tr)
    return (_report(ev, [1] * n, [mode] * n), _report(ev, [0] * n, [mode] * n))


def _b1(tr, ev, a=2.0):
    prior = float(np.mean([e["switch"] for e in tr]))      # global base rate
    mode = _du_mode(tr)
    sw_tab = defaultdict(lambda: [0, 0])                   # feats -> [no-switch count, switch count]
    du_tab = defaultdict(Counter)                          # current language -> duration counts (switch=1 only)
    for e, key in zip(tr, map(_feat, tr)):
        sw_tab[key][e["switch"]] += 1
        if e["duration"] > 0:
            du_tab[key[0]][e["duration"]] += 1
    sw_pred, du_pred, probs = [], [], []
    for e, key in zip(ev, map(_feat, ev)):
        n0, n1 = sw_tab.get(key, (0, 0))
        p = (n1 + a * prior) / (n0 + n1 + a)               # smooth toward the global prior
        probs.append(p)
        sw_pred.append(int(p > prior))                     # threshold at the base rate
        c = du_tab.get(key[0])
        du_pred.append(c.most_common(1)[0][0] if c else mode)
    return _report(ev, sw_pred, du_pred, sw_prob=probs)


def _b2(tr, ev, k=6):
    tail = lambda t: t[-k:]                                # current unit + the punctuation just typed
    vec = TfidfVectorizer(analyzer="char", ngram_range=(1, 3))
    X_tr = vec.fit_transform([tail(e["text"]) for e in tr])
    X_ev = vec.transform([tail(e["text"]) for e in ev])
    clf_sw = LogisticRegression(class_weight="balanced", max_iter=1000).fit(X_tr, [e["switch"] for e in tr])
    has_dur = np.array([e["duration"] > 0 for e in tr])
    clf_du = LogisticRegression(max_iter=1000).fit(X_tr[has_dur], [e["duration"] for e in tr if e["duration"] > 0])
    return _report(ev, clf_sw.predict(X_ev), clf_du.predict(X_ev), sw_prob=clf_sw.predict_proba(X_ev)[:, 1])


def run_baselines(train_raw, eval_raw):
    """Fit on the pooled train set, evaluate on eval_raw. Return {baseline_name: {"ALL"/pair: score_dict}}."""
    b0a, b0n = _b0(train_raw, eval_raw)
    return {"B0 always-switch": b0a, "B0 never-switch": b0n,
            "B1 lookup": _b1(train_raw, eval_raw), "B2 char-ngram LR": _b2(train_raw, eval_raw)}
