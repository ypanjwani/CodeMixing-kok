"""Paper figures from results/runs/*.json and data/processed/stats.json -> results/figures/*.png"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from lid import LABELS, ROOT

RUNS = ROOT / "results" / "runs"
FIG = ROOT / "results" / "figures"

# reference categorical palette (fixed order), text inks, surface
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
INK, INK2, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#8a8984", "#e4e3df", "#fcfcfb"
BLUE_SEQ = ["#fcfcfb", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "axes.titlesize": 12, "axes.titleweight": "bold", "font.size": 10,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.8, "axes.axisbelow": True, "figure.dpi": 150,
})


def load(name):
    p = RUNS / f"{name}.json"
    return json.loads(p.read_text()) if p.exists() else None


def bar_labels(ax, bars, fmt="{:.3f}", horizontal=True):
    for b in bars:
        if not np.isfinite(b.get_width() if horizontal else b.get_height()):
            continue
        if horizontal:
            ax.text(b.get_width(), b.get_y() + b.get_height() / 2, " " + fmt.format(b.get_width()),
                    va="center", ha="left", color=INK2, fontsize=9)
        else:
            ax.text(b.get_x() + b.get_width() / 2, b.get_height(), fmt.format(b.get_height()),
                    va="bottom", ha="center", color=INK2, fontsize=8)


def hbar(names, values, title, xlabel, fname, highlight=0, fmt="{:.3f}", xlim=None):
    fig, ax = plt.subplots(figsize=(7, 0.5 * len(names) + 1.2))
    colors = [SERIES[0] if highlight is None or i == highlight else "#a9a8a3" for i in range(len(names))]
    bars = ax.barh(names[::-1], values[::-1], color=colors[::-1], height=0.6, edgecolor=SURFACE, linewidth=2)
    bar_labels(ax, bars, fmt)
    ax.set_title(title, loc="left")
    ax.set_xlabel(xlabel)
    ax.grid(axis="y", visible=False)
    if xlim:
        ax.set_xlim(*xlim)
    fig.tight_layout()
    fig.savefig(FIG / fname, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    stats = json.loads((ROOT / "data" / "processed" / "stats.json").read_text(encoding="utf-8"))
    pairs = list(stats["pairs"])

    # 1. dataset statistics: two panels instead of a dual axis
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4))
    real = [stats["pairs"][p]["real"] for p in pairs]
    syn = [stats["pairs"][p]["synthetic"] for p in pairs]
    x = np.arange(len(pairs))
    a1.bar(x, real, color=SERIES[0], width=0.6, label="Real", edgecolor=SURFACE, linewidth=2)
    a1.bar(x, syn, bottom=real, color=SERIES[1], width=0.6, label="Synthetic", edgecolor=SURFACE, linewidth=2)
    for i, (r, s) in enumerate(zip(real, syn)):
        a1.text(i, r + s, f"{r + s:,}", ha="center", va="bottom", fontsize=8, color=INK2)
    a1.set_xticks(x, pairs, rotation=20, ha="right")
    a1.set_title("Sentences per language pair", loc="left")
    a1.set_ylabel("Sentences")
    a1.legend(frameon=False)
    a1.grid(axis="x", visible=False)
    avg = [stats["pairs"][p]["avg_words"] for p in pairs]
    b = a2.bar(x, avg, color=SERIES[0], width=0.6)
    bar_labels(a2, b, "{:.1f}", horizontal=False)
    a2.set_xticks(x, pairs, rotation=20, ha="right")
    a2.set_title("Average words per sentence", loc="left")
    a2.set_ylabel("Words")
    a2.grid(axis="x", visible=False)
    fig.tight_layout()
    fig.savefig(FIG / "fig_dataset.png", bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)

    # 2. confusion matrix of the proposed model (seed 42)
    main_run = load("proposed_seed42")
    cm = np.array(main_run["confusion_matrix"])
    fig, ax = plt.subplots(figsize=(6.2, 5.2))
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("blue", BLUE_SEQ)
    im = ax.imshow(np.log10(cm + 1), cmap=cmap)
    for i in range(len(LABELS)):
        for j in range(len(LABELS)):
            ax.text(j, i, f"{cm[i, j]:,}", ha="center", va="center", fontsize=9,
                    color="white" if np.log10(cm[i, j] + 1) > np.log10(cm.max() + 1) * 0.6 else INK)
    ax.set_xticks(range(len(LABELS)), LABELS)
    ax.set_yticks(range(len(LABELS)), LABELS)
    ax.set_xlabel("Predicted language")
    ax.set_ylabel("Actual language")
    ax.grid(False)
    ax.set_title(f"Token-level confusion matrix (test, {cm.sum():,} tokens)", loc="left")
    cb = fig.colorbar(im, ax=ax, fraction=0.046)
    cb.set_label("log10(tokens + 1)")
    fig.tight_layout()
    fig.savefig(FIG / "fig_confusion_matrix.png", bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)

    # 3. model comparison (accuracy, macro-F1) and training time
    models = [("Proposed CNN-BiLSTM-Adam", "proposed_seed42"), ("Baseline CNN", "baseline_cnn"),
              ("Baseline RNN (BiLSTM)", "baseline_bilstm"), ("Transformer (scratch)", "baseline_transformer"),
              ("mBERT (fine-tuned)", "pretrained_mbert"), ("XLM-R (fine-tuned)", "pretrained_xlmr")]
    models = [(n, load(k)) for n, k in models if load(k)]
    names = [n for n, _ in models]
    fig, ax = plt.subplots(figsize=(9, 4.2))
    x = np.arange(len(names))
    w = 0.38
    for k, (metric, label) in enumerate((("accuracy", "Accuracy"), ("f1_macro", "Macro F1"))):
        bars = ax.bar(x + (k - 0.5) * w, [m[metric] for _, m in models], w, color=SERIES[k],
                      label=label, edgecolor=SURFACE, linewidth=2)
        bar_labels(ax, bars, "{:.3f}", horizontal=False)
    lo = min(min(m["accuracy"], m["f1_macro"]) for _, m in models)
    ax.set_ylim(max(0, lo - 0.05), 1.0)
    ax.set_xticks(x, names, rotation=15, ha="right")
    ax.set_title("Model comparison on the test set (token level)", loc="left")
    ax.legend(frameon=False, ncol=2, loc="upper right")
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    fig.savefig(FIG / "fig_model_comparison.png", bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)

    hbar(names, [m["train_time_s"] for _, m in models], "Training time", "Seconds (RTX 3050 laptop GPU)",
         "fig_training_time.png", fmt="{:.0f}s")
    hbar(names, [m["auc_roc"] for _, m in models], "AUC-ROC (one-vs-rest, macro)", "AUC-ROC",
         "fig_auc_roc.png", xlim=(min(m["auc_roc"] for _, m in models) - 0.01, 1.0))

    # 4. optimizer comparison
    opts = [("Adam", "proposed_seed42"), ("AdamW", "optimizer_adamw"), ("RMSprop", "optimizer_rmsprop"),
            ("SGD", "optimizer_sgd")]
    opts = [(n, load(k)) for n, k in opts if load(k)]
    hbar([n for n, _ in opts], [m["accuracy"] for _, m in opts], "Optimizer comparison (CNN-BiLSTM)",
         "Test accuracy", "fig_optimizers.png",
         xlim=(min(m["accuracy"] for _, m in opts) - 0.03, 1.0))

    # 5. ablation
    abl = [("Full model", "proposed_seed42"), ("Without CNN", "baseline_bilstm"),
           ("Without BiLSTM", "baseline_cnn"), ("Without Adam (SGD)", "optimizer_sgd")]
    abl = [(n, load(k)) for n, k in abl if load(k)]
    hbar([n for n, _ in abl], [m["accuracy"] for _, m in abl], "Ablation study", "Test accuracy",
         "fig_ablation.png", xlim=(min(m["accuracy"] for _, m in abl) - 0.03, 1.0))

    # 6. learning rate x batch size grid
    lrs, bss = (1e-5, 1e-4, 1e-3), (32, 64, 128)
    fig, ax = plt.subplots(figsize=(7.5, 4))
    x = np.arange(len(lrs))
    w = 0.26
    for k, bs in enumerate(bss):
        vals = [(load(f"grid_lr{lr:g}_bs{bs}") or {}).get("accuracy", np.nan) for lr in lrs]
        bars = ax.bar(x + (k - 1) * w, vals, w, color=SERIES[k], label=f"batch {bs}",
                      edgecolor=SURFACE, linewidth=2)
        bar_labels(ax, bars, "{:.3f}", horizontal=False)
    ax.set_xticks(x, [f"{lr:g}" for lr in lrs])
    ax.set_xlabel("Learning rate")
    ax.set_ylabel("Test accuracy")
    ax.set_ylim(0, 1.05)
    ax.set_title("Accuracy vs learning rate and batch size (max 20 epochs)", loc="left")
    ax.legend(frameon=False, ncol=3, loc="upper left")
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    fig.savefig(FIG / "fig_lr_batch.png", bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)

    # 7. learning curves
    fig, ax = plt.subplots(figsize=(7.5, 4))
    for k, (n, m) in enumerate(models[:4]):
        h = m["history"]
        ax.plot([e["epoch"] for e in h], [e["val_acc"] for e in h], color=SERIES[k], lw=2, label=n)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation accuracy")
    ax.set_title("Validation accuracy per epoch", loc="left")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIG / "fig_learning_curves.png", bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)

    # 8. per-pair accuracy of the proposed model
    pp = main_run["per_pair_accuracy"]
    hbar(list(pp), list(pp.values()), "Proposed model: accuracy per language pair", "Test accuracy",
         "fig_per_pair.png", highlight=None, xlim=(min(pp.values()) - 0.05, 1.0))
    print("figures written to", FIG)


if __name__ == "__main__":
    main()
