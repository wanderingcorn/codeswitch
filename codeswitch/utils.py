"""Device selection, seeding, memory monitoring, class weights."""
import gc
import os
import random

import numpy as np
import torch
import torch.nn.functional as F

from . import config as C

try:
    import psutil
except ImportError:
    psutil = None  # pip install psutil


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


DEVICE = get_device()
# Abort training when MPS allocation passes this, instead of freezing the machine
MEM_LIMIT_GB = (torch.mps.recommended_max_memory() / 1e9 * 0.9) if DEVICE.type == "mps" else 22


def amp_enabled():
    return bool(getattr(C, "USE_AMP", False)) and DEVICE.type == "cuda"


def autocast():
    """fp16 autocast on CUDA (V100: no bf16); no-op elsewhere. Wrap only the backbone forward."""
    return torch.autocast("cuda", dtype=torch.float16, enabled=amp_enabled())


def set_seed(seed=C.SEED):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)


def _gb(x):
    return x / 1e9


def mem(tag=""):
    s = f"[mem {tag}]"
    if DEVICE.type == "mps":
        s += f" MPS alloc {_gb(torch.mps.current_allocated_memory()):.1f} GB | driver {_gb(torch.mps.driver_allocated_memory()):.1f} GB"
    elif DEVICE.type == "cuda":
        s += f" CUDA alloc {_gb(torch.cuda.memory_allocated()):.1f} GB"
    if psutil:
        s += f" | RSS {_gb(psutil.Process(os.getpid()).memory_info().rss):.1f} GB"
    print(s)


def check_mem():
    if DEVICE.type == "mps" and _gb(torch.mps.current_allocated_memory()) > MEM_LIMIT_GB:
        raise RuntimeError(f"MPS allocated > {MEM_LIMIT_GB:.1f} GB: call free() or lower the batch size")


def free():
    gc.collect()
    if DEVICE.type == "mps":
        torch.mps.empty_cache()
    elif DEVICE.type == "cuda":
        torch.cuda.empty_cache()


def sqrt_balanced_weights(labels):
    """Square-root-damped balanced class weights for a binary label array."""
    labels = np.asarray(labels)
    counts = np.bincount(labels, minlength=2).clip(min=1)
    return (torch.tensor(len(labels) / (2 * counts), dtype=torch.float)) ** 0.5


def dur_weights(labels):
    """sqrt-balanced weights over duration classes 1..3, computed on switch=1 positions only (labels > 0)."""
    y = np.asarray(labels)
    y = y[y > 0]
    counts = np.bincount(y - 1, minlength=3).clip(min=1)
    return (torch.tensor(len(y) / (3 * counts), dtype=torch.float)) ** 0.5


def duration_loss(dur_logits, dur_labels, weights):
    """Duration loss, learned only where a switch occurs. dur_logits (..., N_DUR), dur_labels in {0,1,2,3, IGNORE}.
    Targets: switch=1 -> class-1 (0..2); everything else (no switch, no label) -> IGNORE.
    CE (class-weighted, label-smoothed) on logits[:, 1:] + EMD on the cumulative distribution (ordinal)."""
    lg = dur_logits.reshape(-1, dur_logits.size(-1))[:, 1:]
    y = dur_labels.reshape(-1)
    tgt = torch.where(y > 0, y - 1, torch.full_like(y, C.IGNORE))
    m = tgt != C.IGNORE
    if not m.any():
        return lg.sum() * 0.0
    ce = F.cross_entropy(lg, tgt, weight=weights, ignore_index=C.IGNORE, label_smoothing=C.DUR_SMOOTH)
    cdf_p = lg[m].softmax(-1).cumsum(-1)
    cdf_y = (torch.arange(lg.size(-1), device=lg.device)[None] >= tgt[m][:, None]).float()
    emd = ((cdf_p - cdf_y) ** 2).sum(-1).mean()
    return C.LAMBDA_DUR * (ce + C.LAMBDA_EMD * emd)   # LAMBDA_DUR: DUR vs SW; LAMBDA_EMD: EMD vs CE inside DUR
