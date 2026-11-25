"""Model-agnostic training loop shared by XLM-R and Qwen3."""
import math
import torch
from transformers import get_linear_schedule_with_warmup

from . import config as C
from .metrics import monitor_line
from .utils import DEVICE, amp_enabled, check_mem, free, mem

HEAD_PREFIX = ("switch_head", "dur_head")


def select_score(res):
    """Primary selection score on the pooled val set: mean of switch PR-AUC and duration macro-F1 (higher is better)."""
    return 0.5 * (res["ALL"]["switch_PR_AUC"] + res["ALL"]["dur_macro_F1"])


def eval_steps(n_steps, epoch):
    """1-indexed steps within an epoch at which to validate (scales with dataset size; always includes the epoch end)."""
    fracs = C.EVAL_FRACS_FIRST if epoch == 0 else C.EVAL_FRACS_LATER
    steps = {min(n_steps, max(1, round(n_steps * f))) for f in fracs}
    steps.add(n_steps)
    return steps



def fit(model, train_dl, val_dl, evaluate_fn, epochs, lr, head_lr,
        log_every=100, keep_best=False, weight_decay=0.01, tag="model"):
    body = [p for n, p in model.named_parameters() if p.requires_grad and not n.startswith(HEAD_PREFIX)]
    head = [p for n, p in model.named_parameters() if p.requires_grad and n.startswith(HEAD_PREFIX)]
    opt = torch.optim.AdamW([{"params": body, "lr": lr}, {"params": head, "lr": head_lr}],
                            weight_decay=weight_decay)
    n = len(train_dl)
    total_steps = epochs * n
    sched = get_linear_schedule_with_warmup(opt, int(0.1 * total_steps), total_steps)

    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled())
    history, best, best_state, gstep = [], float("-inf"), None, 0
    nbad, stop = 0, False
    for epoch in range(epochs):
        if stop: break
        model.train()
        eval_at = eval_steps(n, epoch)
        win = [0.0, 0.0, 0.0, 0]
        gmax, ninf = 0.0, 0
        for step, batch in enumerate(train_dl):
            batch = {k: v.to(DEVICE) for k, v in batch.items()}
            batch.pop("pair_ids")
            out = model(**batch)
            loss = out["loss"]
            if not torch.isfinite(loss):
                nbad += 1
                opt.zero_grad()
                print(f"  !! non-finite loss at step {step} (skipped {nbad})")
                if nbad > 20:
                    stop = True; break
            else:
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0).item()
                if math.isfinite(gn):
                    gmax = max(gmax, gn)
                else:
                    ninf += 1
                scaler.step(opt); scaler.update(); sched.step(); opt.zero_grad()
                gstep += 1
                win[0] += loss.item(); win[1] += out["loss_sw"].item(); win[2] += out["loss_du"].item(); win[3] += 1
                if step % log_every == 0:
                    check_mem()
                    if step == 0 and epoch == 0:
                        mem(f"{tag} step 0")
            if step + 1 in eval_at:
                k = max(win[3], 1)
                print(f"epoch {epoch + 1} step {step + 1}/{n} (ep {gstep / n:.3f}): "
                      f"train loss {win[0] / k:.4f} (sw {win[1] / k:.4f} | dur {win[2] / k:.4f}) "
                      f"| max grad norm {gmax:.2f} (inf steps {ninf})")
                gmax, ninf = 0.0, 0
                try:
                    res = evaluate_fn(model, val_dl)
                except (ValueError, FloatingPointError) as e:
                    print(f"  !! validation failed ({e}); stopping, restoring best")
                    stop = True; break
                model.train()
                print(monitor_line(res))
                history.append({"step": gstep, "epoch": round(gstep / n, 3), "res": res,
                                "train": {"loss": win[0] / k, "loss_sw": win[1] / k, "loss_dur": win[2] / k}})
                if keep_best and select_score(res) > best:
                    best = select_score(res)
                    best_state = {nm: p.detach().cpu().clone() for nm, p in model.named_parameters() if p.requires_grad}
                    print("  ^ best so far")
                win = [0.0, 0.0, 0.0, 0]
                free(); mem(f"{tag} step {gstep}")

    if keep_best and best_state is not None:
        model.load_state_dict(best_state, strict=False)
    del opt, sched, scaler, best_state
    free()
    return history