"""Model-agnostic training loop shared by XLM-R and Qwen3."""
import torch
from transformers import get_linear_schedule_with_warmup

from . import config as C
from .metrics import print_scores
from .utils import DEVICE, amp_enabled, check_mem, free, mem

HEAD_PREFIX = ("switch_head", "dur_head")


def select_score(res):
    """Primary selection score on the pooled val set: mean of switch PR-AUC and duration macro-F1 (higher is better)."""
    return 0.5 * (res["ALL"]["switch_PR_AUC"] + res["ALL"]["dur_macro_F1"])


def fit(model, train_dl, val_dl, evaluate_fn, epochs, lr, head_lr,
        log_every=100, keep_best=False, weight_decay=0.01, tag="model"):
    """Train with separate LRs for the backbone and the two heads.

    The batch must contain `pair_ids` (used only for evaluation, popped before the forward pass).
    Parameters with requires_grad=False (e.g. a frozen embedding) are skipped automatically.
    Returns the per-epoch validation results.
    """
    body = [p for n, p in model.named_parameters() if p.requires_grad and not n.startswith(HEAD_PREFIX)]
    head = [p for n, p in model.named_parameters() if p.requires_grad and n.startswith(HEAD_PREFIX)]
    opt = torch.optim.AdamW([{"params": body, "lr": lr}, {"params": head, "lr": head_lr}],
                            weight_decay=weight_decay)
    total_steps = epochs * len(train_dl)
    sched = get_linear_schedule_with_warmup(opt, int(0.1 * total_steps), total_steps)

    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled())   # no-op when AMP is off
    history, best, best_state = [], float("-inf"), None
    for epoch in range(epochs):
        model.train()
        total, tot_sw, tot_du = 0.0, 0.0, 0.0
        for step, batch in enumerate(train_dl):
            batch = {k: v.to(DEVICE) for k, v in batch.items()}
            batch.pop("pair_ids")
            out = model(**batch)
            loss = out["loss"]
            scaler.scale(loss).backward()
            scaler.unscale_(opt)                  # clip on true (unscaled) gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update(); sched.step(); opt.zero_grad()
            total += loss.item(); tot_sw += out["loss_sw"].item(); tot_du += out["loss_du"].item()
            if step % log_every == 0:
                check_mem()
                if step == 0 and epoch == 0:
                    mem(f"{tag} step 0")
        n = len(train_dl)
        print(f"epoch {epoch + 1}: train loss {total / n:.4f} (sw {tot_sw / n:.4f} | dur {tot_du / n:.4f})")
        res = evaluate_fn(model, val_dl)
        print_scores("  val", res)
        history.append(res)
        if keep_best and select_score(res) > best:
            best = select_score(res)
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            print("  ^ best so far")
        free(); mem(f"{tag} epoch {epoch + 1}")

    if keep_best and best_state is not None:
        model.load_state_dict(best_state)
    del opt, sched, scaler, best_state
    free()
    return history
