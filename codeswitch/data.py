"""Corpus loading, train/val split by conversation, and per-prefix example generation."""
import ast
import json
import random
import re
from multiprocessing import get_context

import numpy as np
import pandas as pd

from . import config as C
from .lid import find_units, langs_of, LANG_NAME, PIECE_RE


# ---------- Loading ----------
def parse_conv(s):
    """Parse the data_generation_result string into a list of sentences; None on failure."""
    if not isinstance(s, str):
        return None
    for loader in (ast.literal_eval, json.loads):
        try:
            x = loader(s)
        except (ValueError, SyntaxError, TypeError, MemoryError, RecursionError):
            continue
        if isinstance(x, (list, tuple)):
            return [t for t in x if isinstance(t, str)]
    return None


def clean(t):
    return re.sub(r"\s+", " ", t).strip()


def load_records(path=C.DATA_PATH, verbose=True):
    """Return one record per sentence: {"pair", "conv", "text", "is_test"}.
    Test conversations are fixed per pair by TEST_SEED + N_TEST_CONV, chosen from ALL parseable
    conversations BEFORE any subsampling, so they do not depend on SEED, PAIR_SEL or MAX_CONV_PER_PAIR."""
    df = pd.read_csv(path)
    if verbose:
        print(df[["first_language", "second_language"]].value_counts().to_string())

    records, bad_rows = [], []
    for (l1, l2), tag in C.PAIRS.items():
        sub = df[(df["first_language"] == l1) & (df["second_language"] == l2)]
        if len(sub) == 0:
            raise ValueError(f"No {l1}-{l2} rows in the CSV; check PAIRS against the values printed above")
        parsed = {}
        for row_id, s in zip(sub.index, sub["data_generation_result"]):
            sents = parse_conv(s)
            if sents is None:
                bad_rows.append((tag, row_id, s))
            else:
                parsed[row_id] = sents
        ids = sorted(parsed)
        if len(ids) < C.N_TEST_CONV:
            print(f"!! {tag}: only {len(ids)} parseable conversations < N_TEST_CONV={C.N_TEST_CONV}")
        test_ids = set(random.Random(f"{C.TEST_SEED}-{tag}").sample(ids, min(C.N_TEST_CONV, len(ids))))
        pool = [i for i in ids if i not in test_ids]
        if C.MAX_CONV_PER_PAIR is not None and len(pool) > C.MAX_CONV_PER_PAIR:
            pool = sorted(random.Random(C.SEED).sample(pool, C.MAX_CONV_PER_PAIR))
        for row_id in sorted(test_ids) + pool:
            for t in map(clean, parsed[row_id]):
                if len(find_units(t)) >= C.MIN_UNITS:
                    records.append({"pair": tag, "conv": f"{tag}#{row_id}", "text": t,
                                    "is_test": row_id in test_ids})

    with open(C.BAD_PATH, "w", encoding="utf-8") as f:
        for tag, row_id, s in bad_rows:
            f.write(f"=== pair={tag} | row_id={row_id} | type={type(s).__name__} ===\n{s}\n\n")
    if verbose:
        print(f"\n{len(records)} sentences; {len(bad_rows)} conversations failed to parse -> {C.BAD_PATH}")
    return records


def corpus_report(records, examples_per_pair=2):
    """Per-pair sanity check. If L1 units are almost absent, LID does not fit that pair
    (e.g. romanized Hindi, or the language column does not match the text)."""
    for tag in C.PAIR_TAGS:
        rs = [r for r in records if r["pair"] == tag]
        lg = [langs_of(find_units(r["text"]), tag) for r in rs]
        n_l1 = sum(l.count(0) for l in lg); n_en = sum(l.count(1) for l in lg)
        mixed = np.mean([len(set(l)) > 1 for l in lg])
        print(f"{tag:7} sentences {len(rs):6d} | L1 units {n_l1:8d} | EN units {n_en:8d} | sentences with a switch {mixed:.1%}")
        if n_l1 < C.L1_MIN_SHARE * (n_l1 + n_en):
            print(f"  !! {tag}: L1 share below threshold; check for romanized text or a wrong language column")
    for tag in C.PAIR_TAGS:
        for r in [r for r in records if r["pair"] == tag][:examples_per_pair]:
            print(tag, "|", r["text"])


# ---------- Split ----------
def split_records(records):
    """Return (train, val, test). Test = the fixed conversations flagged in load_records.
    The remaining conversations are split into train/val by conversation (VAL_FRAC per pair)."""
    test = [r for r in records if r["is_test"]]
    rest = [r for r in records if not r["is_test"]]
    rng = random.Random(C.SEED)
    val_convs = set()
    for tag in C.PAIR_TAGS:
        cs = sorted({r["conv"] for r in rest if r["pair"] == tag})
        rng.shuffle(cs)
        val_convs |= set(cs[:max(1, int(len(cs) * C.VAL_FRAC))])
    train = [r for r in rest if r["conv"] not in val_convs]
    val = [r for r in rest if r["conv"] in val_convs]
    return train, val, test


def label_stats(recs, examples, name=""):
    """Gold class counts per pair (+ALL): conversations, sentences, switch [0,1], duration [0..N_DUR-1]. Prints and returns a dict."""
    out = {}
    for tag in ["ALL"] + C.PAIR_TAGS:
        rs = recs if tag == "ALL" else [r for r in recs if r["pair"] == tag]
        es = examples if tag == "ALL" else [e for e in examples if e["pair"] == tag]
        sw = np.array([e["switch"] for e in es], dtype=int)
        du = np.array([e["duration"] for e in es], dtype=int)
        out[tag] = {"conversations": len({r["conv"] for r in rs}), "sentences": len(rs), "examples": len(es),
                    "switch": np.bincount(sw, minlength=2).tolist(),
                    "duration": np.bincount(du, minlength=C.N_DUR).tolist()}
        print(f"[{name:5}] {tag:7} conv {out[tag]['conversations']:5d} | sent {len(rs):6d} | ex {len(es):7d} | "
              f"switch(0/1) {out[tag]['switch']} | dur(0/1/2/3) {out[tag]['duration']}")
    return out


# ---------- Examples ----------
def make_examples(text, pair=None):
    """One example per stopping point (after each unit, and after each punctuation mark / number)."""
    units = find_units(text)
    n = len(units)
    langs = langs_of(units, pair)

    # run_len[j] = number of consecutive same-language units starting at unit j
    run_len = [1] * n
    for j in range(n - 2, -1, -1):
        if langs[j] == langs[j + 1]:
            run_len[j] = run_len[j + 1] + 1

    examples = []
    for i in range(n - 1):
        switch = int(langs[i] != langs[i + 1])
        # On a switch, unit i+1 starts the new segment, so run_len[i+1] is its length
        duration = C.dur_class(run_len[i + 1]) if switch else 0

        gap_start = units[i].end()
        gap = text[gap_start:units[i + 1].start()]                # punctuation / digits / spaces between units
        stops = [gap_start] + [gap_start + m.end() for m in PIECE_RE.finditer(gap)]
        for end in stops:
            examples.append({
                "text": text[:end],
                "pair": pair,
                "switch": switch,
                "duration": duration,
                "last_lang": LANG_NAME[langs[i]],                 # LID labels, for inspection only
                "next_lang": LANG_NAME[langs[i + 1]],
            })
    return examples


def pmap(fn, items, chunksize=256, min_items=1000):
    """Fork process pool; falls back to serial for 1 core / small inputs. fn must be a module-level function."""
    items = list(items)
    if C.NUM_WORKERS <= 1 or len(items) < min_items:
        return list(map(fn, items))
    with get_context("fork").Pool(C.NUM_WORKERS) as p:
        return p.map(fn, items, chunksize=chunksize)


def _mk(r):
    return make_examples(r["text"], r["pair"])


def build_examples(recs):
    return [e for es in pmap(_mk, recs) for e in es]


def example_report(train_raw, val_raw):
    sw = np.array([e["switch"] for e in train_raw])
    du = np.array([e["duration"] for e in train_raw])
    print(f"train: {len(train_raw)} examples, switch=1 share {sw.mean():.1%}")
    print("duration class counts:", np.bincount(du, minlength=C.N_DUR).tolist())
    print(f"val:   {len(val_raw)} examples")
    for tag in C.PAIR_TAGS:
        tr = [e for e in train_raw if e["pair"] == tag]; va = [e for e in val_raw if e["pair"] == tag]
        print(f"  {tag:7} train {len(tr):7d} (switch=1 {np.mean([e['switch'] for e in tr]):.1%}) | val {len(va):6d}")
