"""统计 test 对话上正确 / 错误的 switch 预测。
用法: python -m codeswitch.eval_switch out/NAME [--n 200] [--thr 0.42]
输出: 终端汇总 (ALL / 每个 pair / 每个 cs_type) + <run>/switch_preds.csv (每个 stop 一行)
定义: TP = 正确预测的 switch | FP = 误报 (false alarm) | FN = 漏报 | TN = 正确的 no-switch
Usage： python -m codeswitch.eval_switch out/NAME            # HPC：全部 test 对话
        python -m codeswitch.eval_switch out/all_qwen_8k_dur02 --n 50     # MacBook：快速测试
        python -m codeswitch.eval_switch out/NAME --thr 0.4  # 可选：覆盖保存的阈值
"""
import argparse
import csv
import json
import os
from collections import defaultdict

import pandas as pd
import torch

from . import config as C
from .data import load_convs, make_examples
from .store import load_run
from .utils import DEVICE


@torch.no_grad()
def probs_qwen(model, tok, text, langs):
    """一次前向得到整句所有 stop 的 P(switch); 与 make_examples 顺序一一对应 (被截断的尾部不返回)。"""
    from .qwen import build_sample
    s = build_sample(text, langs, tok)
    ids = torch.tensor([s["input_ids"][:C.QWEN_MAX_LEN]], device=DEVICE)
    lab = torch.tensor(s["switch_labels"][:C.QWEN_MAX_LEN])
    out = model(input_ids=ids, attention_mask=torch.ones_like(ids))
    p = out["switch_logits"][0].softmax(-1)[:, 1].cpu()
    return p[lab != C.IGNORE].tolist()


@torch.no_grad()
def probs_xlmr(model, tok, exs, bs=64):
    res = []
    for i in range(0, len(exs), bs):
        enc = tok([e["text"].rstrip() + tok.mask_token for e in exs[i:i + bs]], truncation=True,
                  max_length=C.MAX_LEN, padding=True, return_tensors="pt").to(DEVICE)
        out = model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"])
        res += out["switch_logits"].softmax(-1)[:, 1].cpu().tolist()
    return res


def summarize(rows, key):
    g = defaultdict(lambda: dict(TP=0, FP=0, FN=0, TN=0))
    stats = {}
    for r in rows:
        g[r[key]][r["outcome"]] += 1
    print(f"\n--- by {key} ---")
    print(f"{key:<14}{'TP':>8}{'FP':>8}{'FN':>8}{'TN':>9}{'P':>7}{'R':>7}{'F1':>7}")
    for k, c in sorted(g.items(), key=lambda x: str(x[0])):
        p = c["TP"] / max(1, c["TP"] + c["FP"]); r_ = c["TP"] / max(1, c["TP"] + c["FN"])
        f = 2 * p * r_ / max(1e-12, p + r_)
        stats[str(k)] = {**c, "P": round(p, 4), "R": round(r_, 4), "F1": round(f, 4)}
        print(f"{str(k):<14}{c['TP']:>8}{c['FP']:>8}{c['FN']:>8}{c['TN']:>9}{p:>7.3f}{r_:>7.3f}{f:>7.3f}")
    return stats


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--n", type=int, default=None, help="每次只用前 n 个 test 对话 (MacBook 快速测试)")
    ap.add_argument("--thr", type=float, default=None, help="默认用保存的阈值")
    a = ap.parse_args(argv)

    kind, model, tok, thr, meta = load_run(a.run)
    thr = thr if a.thr is None else a.thr
    C.PAIR_SEL = meta["pairs"]; C.apply_pair_sel()
    with open(os.path.join(a.run, "test_convs.json")) as f:
        ids = json.load(f)
    ids = ids[:a.n] if a.n else ids
    print(f"{kind} | {len(ids)} test conversations | thr={thr:.4f} | device={DEVICE}")

    df, anns, rows = pd.read_csv(C.DATA_PATH), {}, []
    for i in range(0, len(ids), 200):
        for c in load_convs(ids[i:i + 200], df=df, anns=anns):
            for t, turn in enumerate(c["turns"]):
                if turn["langs"] is None:
                    continue
                exs = make_examples(turn["text"], turn["langs"], c["pair"])
                if not exs:
                    continue
                ps = probs_qwen(model, tok, turn["text"], turn["langs"]) if kind == "qwen" else probs_xlmr(model, tok, exs)
                for e, p in zip(exs, ps):
                    pred = int(p >= thr)
                    outcome = {(1, 1): "TP", (1, 0): "FP", (0, 1): "FN", (0, 0): "TN"}[(pred, e["switch"])]
                    rows.append({"conv": c["conv"], "turn": t, "pair": c["pair"], "cs_type": c["cs_type"],
                                 "prefix_tail": e["text"][-40:], "gold": e["switch"], "p_switch": round(p, 4),
                                 "pred": pred, "outcome": outcome})
        print(f"  {min(i + 200, len(ids))}/{len(ids)} conversations", flush=True)

    for r in rows:
        r["ALL"] = "ALL"
    counts = {k: summarize(rows, k) for k in ("ALL", "pair", "cs_type")}
    counts["thr"] = thr
    with open(os.path.join(a.run, "switch_counts.json"), "w", encoding="utf-8") as f:
        json.dump(counts, f, ensure_ascii=False, indent=2)

    path = os.path.join(a.run, "switch_preds.csv")
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=[k for k in rows[0] if k != "ALL"], extrasaction="ignore")
        w.writeheader(); w.writerows(rows)
    print(f"\nsaved {len(rows)} rows -> {path}  + switch_counts.json (filter outcome == 'FP' / 'TP' to inspect)")


if __name__ == "__main__":
    main()
