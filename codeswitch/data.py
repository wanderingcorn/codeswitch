"""Corpus loading, cached-LID lookup, train/val split by conversation, and per-prefix example generation."""
import ast
import json
import random
import re
from multiprocessing import get_context

import numpy as np
import pandas as pd

from . import config as C
from . import lid
from .lid import find_units, find_stops, LANG_NAME

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


def read_pairs(path=C.DATA_PATH, verbose=True):
    """Return (df, {pair: {row_id: [raw sentences]}}, bad_rows) for the selected pairs."""
    df = pd.read_csv(path)
    if verbose:
        print(df[["first_language", "second_language"]].value_counts().to_string())
    parsed, bad_rows = {}, []
    for (l1, l2), tag in C.PAIRS.items():
        sub = df[(df["first_language"] == l1) & (df["second_language"] == l2)]
        if len(sub) == 0:
            raise ValueError(f"No {l1}-{l2} rows in the CSV; check PAIRS against the values printed above")
        parsed[tag] = {}
        for row_id, s in zip(sub.index, sub["data_generation_result"]):
            sents = parse_conv(s)
            if sents is None:
                bad_rows.append((tag, row_id, s))
            else:
                parsed[tag][row_id] = sents
    return df, parsed, bad_rows


def select_convs(parsed):
    """[(pair, row_id, is_test)]. Test conversations are fixed per pair by TEST_SEED + N_TEST_CONV, chosen from ALL parseable
    conversations BEFORE any subsampling, so they do not depend on SEED, PAIR_SEL or MAX_CONV_PER_PAIR."""
    sel = []
    for tag in C.PAIR_TAGS:
        ids = sorted(parsed[tag])
        if len(ids) < C.N_TEST_CONV:
            print(f"!! {tag}: only {len(ids)} parseable conversations < N_TEST_CONV={C.N_TEST_CONV}")
        test_ids = set(random.Random(f"{C.TEST_SEED}-{tag}").sample(ids, min(C.N_TEST_CONV, len(ids))))
        pool = [i for i in ids if i not in test_ids]
        if C.MAX_CONV_PER_PAIR is not None and len(pool) > C.MAX_CONV_PER_PAIR:
            pool = sorted(random.Random(C.SEED).sample(pool, C.MAX_CONV_PER_PAIR))
        sel += [(tag, i, True) for i in sorted(test_ids)] + [(tag, i, False) for i in pool]
    return sel


def _sentences(raw):
    """(index in the conversation, cleaned text, n_units) for every sentence that has a prediction point."""
    for idx, t in enumerate(map(clean, raw)):
        n = len(find_units(t))
        if n >= C.MIN_UNITS:
            yield idx, t, n


def annotate_selected(path=C.DATA_PATH):
    """Run LID (MixLID / bhasha) on the sentences the experiment will use and cache the labels on disk.
    C.ANN_ALL=True annotates every parseable conversation instead (use once on HPC; later runs with any
    MAX_CONV_PER_PAIR / SEED are then fully covered). Resumable: cached sentences are skipped."""
    _, parsed, _ = read_pairs(path, verbose=False)
    keep = {tag: None for tag in C.PAIR_TAGS} if C.ANN_ALL else {}
    if not C.ANN_ALL:
        for tag, row_id, _ in select_convs(parsed):
            keep.setdefault(tag, []).append(row_id)
    for tag in C.PAIR_TAGS:
        rows = sorted(parsed[tag]) if C.ANN_ALL else keep.get(tag, [])
        items = [(f"{rid}:{idx}", t) for rid in rows for idx, t, _ in _sentences(parsed[tag][rid])]
        lid.annotate_items(tag, items)


def load_records(path=C.DATA_PATH, verbose=True):
    """Return one record per sentence: {"pair", "conv", "text", "langs", "is_test"} (langs = cached unit labels).
    Raises if some selected sentence has no cached annotation (run `run.py --model annotate` first)."""
    _, parsed, bad_rows = read_pairs(path, verbose)
    ann = {tag: lid.load_ann(tag) for tag in C.PAIR_TAGS}
    records, missing = [], 0
    for tag, row_id, is_test in select_convs(parsed):
        for idx, t, n in _sentences(parsed[tag][row_id]):
            langs = lid.lookup(ann[tag], row_id, idx, n)
            if langs is None:
                missing += 1
                continue
            records.append({"pair": tag, "conv": f"{tag}#{row_id}", "text": t, "langs": langs, "is_test": is_test})
    if missing:
        raise RuntimeError(f"{missing} sentences have no cached LID annotation in {C.ANN_DIR}/ "
                           f"(run: python run.py --model annotate --name ann --set <same PAIR_SEL / MAX_CONV_PER_PAIR>)")
    with open(C.BAD_PATH, "w", encoding="utf-8") as f:
        for tag, row_id, s in bad_rows:
            f.write(f"=== pair={tag} | row_id={row_id} | type={type(s).__name__} ===\n{s}\n\n")
    if verbose:
        print(f"\n{len(records)} sentences; {len(bad_rows)} conversations failed to parse -> {C.BAD_PATH}")
    return records


def load_convs(conv_ids, path=C.DATA_PATH):
    """Complete conversations for the demo. Each: {"conv", "pair", "first_language", "cs_type", "turns": [{"text", "langs"|None}]}.
    A turn is one element of the conversation list; langs is None when the turn is too short or not annotated."""
    df = pd.read_csv(path)
    if C.CS_TYPE_COL not in df.columns:
        raise KeyError(f"column {C.CS_TYPE_COL!r} not in the CSV; columns: {list(df.columns)}")
    anns, out = {}, []
    for cid in conv_ids:
        tag, rid = cid.split("#")
        row = df.loc[int(rid)]
        anns.setdefault(tag, lid.load_ann(tag))
        raw = parse_conv(row["data_generation_result"]) or []
        turns = []
        for idx, t in enumerate(map(clean, raw)):
            n = len(find_units(t))
            turns.append({"text": t, "langs": lid.lookup(anns[tag], rid, idx, n) if n >= C.MIN_UNITS else None})
        out.append({"conv": cid, "pair": tag, "first_language": row["first_language"],
                    "cs_type": row[C.CS_TYPE_COL], "turns": turns})
    return out


def corpus_report(records, examples_per_pair=2):
    """Per-pair sanity check of the cached LID labels (L1 share, switch rate)."""
    for tag in C.PAIR_TAGS:
        rs = [r for r in records if r["pair"] == tag]
        n_l1 = sum(r["langs"].count(0) for r in rs); n_en = sum(r["langs"].count(1) for r in rs)
        mixed = np.mean([len(set(r["langs"])) > 1 for r in rs])
        print(f"{tag:7} sentences {len(rs):6d} | L1 units {n_l1:8d} | EN units {n_en:8d} | sentences with a switch {mixed:.1%}")
        if n_l1 < C.L1_MIN_SHARE * (n_l1 + n_en):
            print(f"  !! {tag}: L1 share below threshold; check the LID labels or the language column")
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
def make_examples(text, langs, pair=None):
    """One example per stopping point (after each unit, and after each punctuation mark / number).
    langs = cached per-unit labels of the FULL sentence. Besides the gold labels, each example carries the
    features of the visible prefix used by the lookup baseline: cur (last unit's label), run (units in the
    current language so far), trail (punctuation / digits after the last unit)."""
    units = find_units(text)
    n = len(units)
    assert len(langs) == n, "cached labels do not match the unit segmentation"

    run_len = [1] * n                      # run_len[j]  = consecutive same-language units starting at j
    for j in range(n - 2, -1, -1):
        if langs[j] == langs[j + 1]:
            run_len[j] = run_len[j + 1] + 1
    run_sofar = [1] * n                    # run_sofar[j] = consecutive same-language units ending at j
    for j in range(1, n):
        if langs[j] == langs[j - 1]:
            run_sofar[j] = run_sofar[j - 1] + 1

    examples = []
    for s in find_stops(text):
        i = s.ref                                   # 非 unit stop 沿用 ref unit 的 CS / DUR 标签
        if i + 1 >= n:
            continue                                # 最后一个 unit 及其后的标点没有"下一个 unit"
        switch = int(langs[i] != langs[i + 1])
        duration = C.dur_class(run_len[i + 1]) if switch else 0
        examples.append({
            "text": text[:s.end], "pair": pair, "switch": switch, "duration": duration,
            "cur": langs[i], "run": run_sofar[i], "trail": int(not s.is_unit),
            "last_lang": LANG_NAME[langs[i]], "next_lang": LANG_NAME[langs[i + 1]],
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
    return make_examples(r["text"], r["langs"], r["pair"])


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
