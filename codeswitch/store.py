"""Save / load a trained run: trainable weights (best epoch), tokenizer, switch threshold and architecture settings.
Layout: <out>/model/{weights.pt, tokenizer/, meta.json}. Frozen weights (Qwen base / embedding) are re-read from the hub cache."""
import json
import os

import torch
from transformers import AutoTokenizer

from . import config as C

ARCH_KEYS = ["XLMR_NAME", "MAX_LEN", "QWEN_NAME", "QWEN_MAX_LEN", "USE_LORA",
             "LORA_R", "LORA_ALPHA", "LORA_DROPOUT", "LORA_TARGETS", "N_DUR"]


def save_run(out, kind, model, tok, thr):
    d = os.path.join(out, "model")
    os.makedirs(d, exist_ok=True)
    state = {n: p.detach().cpu() for n, p in model.named_parameters() if p.requires_grad}
    torch.save(state, os.path.join(d, "weights.pt"))
    tok.save_pretrained(os.path.join(d, "tokenizer"))
    meta = {"kind": kind, "thr": float(thr), "pairs": list(C.PAIR_TAGS), **{k: getattr(C, k) for k in ARCH_KEYS}}
    with open(os.path.join(d, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"saved model -> {d} (threshold {thr:.4f})")


def load_run(out):
    """Return (kind, model, tokenizer, thr, meta); model is on utils.DEVICE in eval mode."""
    from .utils import DEVICE
    d = os.path.join(out, "model")
    with open(os.path.join(d, "meta.json")) as f:
        meta = json.load(f)
    for k in ARCH_KEYS:                                   # architecture settings are module globals
        setattr(C, k, meta[k])
    C.GRAD_CKPT = False
    tok = AutoTokenizer.from_pretrained(os.path.join(d, "tokenizer"))
    sw_w, du_w = torch.ones(2), torch.ones(3)             # loss weights are irrelevant at inference
    if meta["kind"] == "xlmr":
        from . import xlmr
        tok.truncation_side = "left"
        model = xlmr.SwitchDurModel(C.XLMR_NAME, C.N_DUR, sw_w, du_w, tok.mask_token_id)
    else:
        from . import qwen
        model = qwen.QwenSwitchDur(C.QWEN_NAME, C.N_DUR, sw_w, du_w)
    r = model.load_state_dict(torch.load(os.path.join(d, "weights.pt"), map_location="cpu"), strict=False)
    assert not r.unexpected_keys, r.unexpected_keys
    return meta["kind"], model.to(DEVICE).eval(), tok, meta["thr"], meta
