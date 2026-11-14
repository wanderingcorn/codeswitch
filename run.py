"""Run one experiment: python run.py --name NAME --model xlmr|qwen|baselines [--set KEY=VALUE ...]
Results go to out/NAME/{results,history,config}.json"""
import argparse, ast, json, os, time

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
print("PAIR_TAGS:", C.PAIR_TAGS, flush=True)

from codeswitch.utils import set_seed
from codeswitch import data

out = f"out/{args.name}"
os.makedirs(out, exist_ok=True)
set_seed(C.SEED)
t0 = time.time()

records = data.load_records(verbose=False)
train_recs, val_recs = data.split_records(records)
hist = []

if args.model == "baselines":
    from codeswitch import baselines
    res = baselines.run_baselines(data.build_examples(train_recs), data.build_examples(val_recs))
elif args.model == "xlmr":
    from codeswitch import xlmr
    tr, va = data.build_examples(train_recs), data.build_examples(val_recs)
    tok, model, tdl, vdl = xlmr.build(tr, va)
    hist = xlmr.train(model, tdl, vdl)       # restores best epoch
    res = xlmr.evaluate(model, vdl)
else:
    from codeswitch import qwen
    tok, model, tdl, vdl, _ = qwen.build(train_recs, val_recs)
    hist = qwen.train(model, tdl, vdl)       # restores best epoch
    res = qwen.evaluate(model, vdl)

cfg = {k: getattr(C, k) for k in dir(C)
       if k.isupper() and not k.startswith("_") and k not in ("PAIRS", "ALL_PAIRS")}
cfg.update(model=args.model, overrides=args.set, minutes=round((time.time() - t0) / 60, 1))
json.dump(res,  open(f"{out}/results.json", "w"), indent=2, default=float)
json.dump(hist, open(f"{out}/history.json", "w"), indent=2, default=float)
json.dump(cfg,  open(f"{out}/config.json",  "w"), indent=2, default=str)
print(f"done: {out} ({cfg['minutes']} min)")
