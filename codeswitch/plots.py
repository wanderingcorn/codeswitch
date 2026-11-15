"""Training curves: loss convergence (Switch vs Duration) + the 4 monitored metrics."""
import matplotlib
matplotlib.use("Agg")                      # headless (HPC)
import matplotlib.pyplot as plt


def plot_history(hist, path="curves.png"):
    ep = [h["epoch"] for h in hist]
    A = lambda k: [h["res"]["ALL"][k] for h in hist]
    T = lambda k: [h["train"][k] for h in hist]
    fig, ax = plt.subplots(2, 3, figsize=(16, 7), sharex=True)
    ax[1, 2].axis("off")

    for a, title, tr, va in ((ax[0, 0], "Total loss", "loss", "val_loss"),
                             (ax[0, 1], "Switch loss", "loss_sw", "val_loss_sw"),
                             (ax[0, 2], "Duration loss", "loss_dur", "val_loss_dur")):
        a.plot(ep, T(tr), "o-", label="train (window avg)")
        a.plot(ep, A(va), "s-", label="val")
        a.set_title(title); a.set_ylabel("loss"); a.legend(); a.grid(alpha=.3)

    ax[1, 0].plot(ep, A("switch_PR_AUC"), "o-", label="PR-AUC")
    ax[1, 0].plot(ep, A("switch_F1"), "s-", label="Anticipatory F1 (argmax)")
    ax[1, 0].plot(ep, A("switch_F1max"), "^--", label="Anticipatory F1 (max over thr)")
    ax[1, 0].set_title("Switch head (val)")
    ax[1, 1].plot(ep, A("dur_acc"), "o-", label="Accuracy")
    ax[1, 1].plot(ep, A("dur_macro_F1"), "s-", label="Macro-F1")
    ax[1, 1].set_title("Duration head (val)")
    for a in ax[1]:
        a.set_xlabel("epoch"); a.legend(); a.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)
