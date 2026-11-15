"""Run one experiment: python run.py --name NAME --model xlmr|qwen|baselines [--set KEY=VALUE ...]
Results go to out/NAME/{results,history,config}.json"""
import argparse, ast, json, os, time

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")   # avoid tokenizer deadlocks after fork

ap = argparse.ArgumentParser()
ap.add_argument("--model", choices=["xlmr", "qwen", "baselines"], required=True)
ap.add_argument("--name", required=True)
ap.add_argument("--set", nargs="*", default=[])      # e.g. LR=2e-5 EPOCHS=3
args = ap.parse_args()

# Override config BEFORE importing the other modules (some defaults bind at import time)
from codeswitch import config as C
for kv in args.set:
    k, v = kv.split("=", 1)
    assert hasattr(C, k), f"unknown config key: {k}"
    setattr(C, k, ast.literal_eval(v))
C.apply_pair_sel()
print("PAIR_TAGS:", C.PAIR_TAGS, "| NUM_WORKERS:", C.NUM_WORKERS, flush=True)
import torch; torch.set_num_threads(C.NUM_WORKERS)

from codeswitch.utils import set_seed
from codeswitch import data

out = f"out/{args.name}"
os.makedirs(out, exist_ok=True)
set_seed(C.SEED)
t0 = time.time()

records = data.load_records(verbose=False)
train_recs, val_recs, test_recs = data.split_records(records)   # test is fixed per pair (TEST_SEED, N_TEST_CONV)
tr, va, te = (data.build_examples(r) for r in (train_recs, val_recs, test_recs))
split_stats = {"train": data.label_stats(train_recs, tr, "train"),
               "val":   data.label_stats(val_recs, va, "val"),
               "test":  data.label_stats(test_recs, te, "test")}
json.dump(split_stats, open(f"{out}/split_stats.json", "w"), indent=2)
json.dump(sorted({r["conv"] for r in test_recs}), open(f"{out}/test_convs.json", "w"))   # to verify the test set is identical across runs
hist = []

if args.model == "baselines":
    from codeswitch import baselines
    res = baselines.run_baselines(tr, va)
    test_res = baselines.run_baselines(tr, te)   # same train set, evaluated on the fixed test set
elif args.model == "xlmr":
    from codeswitch import xlmr
    tok, model, tdl, vdl = xlmr.build(tr, va)
    hist = xlmr.train(model, tdl, vdl)       # restores best epoch (by SW_PR-AUC + DUR_MacroF1)
    mod, test_dl = xlmr, xlmr.make_eval_dl(te, tok)
    res = xlmr.evaluate(model, vdl)
else:
    from codeswitch import qwen
    tok, model, tdl, vdl, _ = qwen.build(train_recs, val_recs)
    hist = qwen.train(model, tdl, vdl)       # restores best epoch (by SW_PR-AUC + DUR_MacroF1)
    mod, test_dl = qwen, qwen.make_eval_dl(test_recs, tok)
    res = qwen.evaluate(model, vdl)

if args.model == "baselines":
    from codeswitch.metrics import pair_report
    json.dump(test_res, open(f"{out}/test_results.json", "w"), indent=2, default=float)
    for name, r in test_res.items():
        rep = pair_report(r)
        print(f"\nTEST {name}:\n{rep}")
        rep.to_csv(f"{out}/test_by_pair_{name.split()[0]}_{name.split()[1]}.csv")
else:
    from codeswitch.metrics import pair_report, print_scores
    from codeswitch.plots import plot_history
    # threshold maximising switch F1, searched on VAL (pooled), then applied unchanged to TEST
    thr = res["ALL"]["switch_thr"]
    print(f"val: best switch-F1 {res['ALL']['switch_F1max']:.3f} @ thr {thr:.4f}")
    test_res = mod.evaluate(model, test_dl, thr=thr)
    test_res["ALL"]["threshold"] = thr
    print_scores("test", test_res)
    rep = pair_report(test_res)
    print(f"\nTEST per language pair (thr={thr:.4f}):\n{rep}")
    rep.to_csv(f"{out}/test_by_pair.csv")
    json.dump(test_res, open(f"{out}/test_results.json", "w"), indent=2, default=float)
    plot_history(hist, f"{out}/curves.png")

cfg = {k: getattr(C, k) for k in dir(C)
       if k.isupper() and not k.startswith("_") and k not in ("PAIRS", "ALL_PAIRS")}
cfg.update(model=args.model, overrides=args.set, minutes=round((time.time() - t0) / 60, 1))
json.dump(res,  open(f"{out}/results.json", "w"), indent=2, default=float)
json.dump(hist, open(f"{out}/history.json", "w"), indent=2, default=float)
json.dump(cfg,  open(f"{out}/config.json",  "w"), indent=2, default=str)
print(f"done: {out} ({cfg['minutes']} min)")
