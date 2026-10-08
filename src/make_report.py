"""Write results/REPORT.md. Every number is read from results/runs/*.json or data/processed/stats.json."""
import json

import numpy as np

from lid import LABELS, ROOT

RUNS = ROOT / "results" / "runs"
OUT = ROOT / "results" / "REPORT.md"

PAPER = {  # numbers claimed in the paper, for side-by-side comparison only
    "proposed": dict(accuracy=0.926, f1=0.925, precision=0.926, recall=0.924, auc_roc=0.930),
    "baseline_cnn": dict(accuracy=0.870, f1=0.870, auc_roc=0.880),
    "baseline_bilstm": dict(accuracy=0.860, f1=0.860, auc_roc=0.860),
    "baseline_transformer": dict(accuracy=0.900, f1=0.900, auc_roc=0.910),
    "pretrained_mbert": dict(accuracy=0.900, f1=0.900),
    "pretrained_xlmr": dict(accuracy=0.910, f1=0.910),
    "optimizer_sgd": dict(accuracy=0.882), "optimizer_rmsprop": dict(accuracy=0.901),
    "optimizer_adamw": dict(accuracy=0.915),
}


def load(name):
    p = RUNS / f"{name}.json"
    return json.loads(p.read_text()) if p.exists() else None


def f(x, d=3):
    return "–" if x is None else f"{x:.{d}f}"


def row(*cells):
    return "| " + " | ".join(str(c) for c in cells) + " |"


def table(header, rows):
    return "\n".join([row(*header), row(*["---"] * len(header))] + [row(*r) for r in rows])


def build():
    stats = json.loads((ROOT / "data" / "processed" / "stats.json").read_text(encoding="utf-8"))
    R = {p.stem: json.loads(p.read_text()) for p in RUNS.glob("*.json")}
    seeds = sorted((R[k] for k in R if k.startswith("proposed_seed")), key=lambda r: r["seed"])
    main = R["proposed_seed42"]
    T = {}

    # dataset
    T["dataset"] = table(
        ["Language pair", "Sentences", "Real", "Synthetic", "Avg. words", "Mixing ratio", "Paper (sentences)"],
        [[p, f"{d['samples']:,}", f"{d['real']:,}", f"{d['synthetic']:,}", d["avg_words"], d["mixing_ratio"],
          f"{d['paper_target_samples']:,}"] for p, d in stats["pairs"].items()]
        + [["**Total**", f"**{stats['total_samples']:,}**", "", "", "", "", "**48,397**"]])
    T["splits"] = table(["Split", "Sentences"], [[k, f"{v:,}"] for k, v in stats["splits"].items()])

    # main + seeds
    keys = ["accuracy", "f1", "precision", "recall"]
    srows = [[s["seed"]] + [f(s[k]) for k in keys] + [f(s["f1_macro"]), f(s["mcc"]), f(s["auc_roc"])] for s in seeds]
    mean = {k: np.mean([s[k] for s in seeds]) for k in keys + ["f1_macro", "mcc", "auc_roc"]}
    std = {k: np.std([s[k] for s in seeds], ddof=1) for k in mean}
    srows.append(["**Mean ± SD**"] + [f"**{mean[k]:.3f} ± {std[k]:.3f}**" for k in keys + ["f1_macro", "mcc", "auc_roc"]])
    T["seeds"] = table(["Seed", "Accuracy", "F1 (w)", "Precision (w)", "Recall (w)", "Macro F1", "MCC", "AUC-ROC"], srows)

    # model comparison
    models = [("Proposed CNN-BiLSTM-Adam", "proposed_seed42", "proposed"),
              ("Baseline CNN", "baseline_cnn", "baseline_cnn"),
              ("Baseline RNN (BiLSTM)", "baseline_bilstm", "baseline_bilstm"),
              ("Transformer (from scratch)", "baseline_transformer", "baseline_transformer"),
              ("mBERT (fine-tuned)", "pretrained_mbert", "pretrained_mbert"),
              ("XLM-R (fine-tuned)", "pretrained_xlmr", "pretrained_xlmr")]
    T["models"] = table(
        ["Method", "Accuracy", "F1 (w)", "Precision (w)", "Recall (w)", "Macro F1", "AUC-ROC", "Train time (s)",
         "Params", "Paper acc."],
        [[n, f(R[k]["accuracy"]), f(R[k]["f1"]), f(R[k]["precision"]), f(R[k]["recall"]), f(R[k]["f1_macro"]),
          f(R[k]["auc_roc"]), f"{R[k]['train_time_s']:.0f}", f"{R[k]['params'] / 1e6:.1f}M",
          f(PAPER[p].get("accuracy"))] for n, k, p in models if k in R])

    # optimizers
    opts = [("SGD (momentum 0.9)", "optimizer_sgd"), ("RMSprop", "optimizer_rmsprop"),
            ("AdamW", "optimizer_adamw"), ("Adam (proposed)", "proposed_seed42")]
    T["optimizers"] = table(["Optimizer", "Accuracy", "F1 (w)", "Precision (w)", "Recall (w)", "Epochs run",
                             "Paper acc."],
                            [[n, f(R[k]["accuracy"]), f(R[k]["f1"]), f(R[k]["precision"]), f(R[k]["recall"]),
                              R[k]["epochs_run"], f(PAPER.get(k, PAPER["proposed"]).get("accuracy"))]
                             for n, k in opts if k in R])

    # ablation
    abl = [("Full model (CNN + BiLSTM + Adam)", "proposed_seed42"), ("Without CNN", "baseline_bilstm"),
           ("Without BiLSTM", "baseline_cnn"), ("Without Adam (SGD)", "optimizer_sgd")]
    T["ablation"] = table(["Variant", "Accuracy", "F1 (w)", "Precision (w)", "Recall (w)", "Δ accuracy"],
                          [[n, f(R[k]["accuracy"]), f(R[k]["f1"]), f(R[k]["precision"]), f(R[k]["recall"]),
                            f"{R[k]['accuracy'] - main['accuracy']:+.3f}"] for n, k in abl if k in R])

    # grid
    grid = []
    for lr in (1e-5, 1e-4, 1e-3):
        cells = [f"{lr:g}"]
        for bs in (32, 64, 128):
            r = R.get(f"grid_lr{lr:g}_bs{bs}")
            cells.append(f"{f(r['accuracy'])} ({r['epochs_run']} ep)" if r else "–")
        grid.append(cells)
    T["grid"] = table(["Learning rate", "Batch 32", "Batch 64", "Batch 128"], grid)

    # per label / per pair / confusion
    T["per_label"] = table(["Language", "Precision", "Recall", "F1", "Test tokens"],
                           [[l, f(v["precision"]), f(v["recall"]), f(v["f1"]), f"{v['support']:,}"]
                            for l, v in main["per_label"].items()])
    T["per_pair"] = table(["Language pair", "Accuracy"], [[p, f(a)] for p, a in main["per_pair_accuracy"].items()])
    cm = main["confusion_matrix"]
    T["confusion"] = table(["Actual \\ Predicted"] + LABELS,
                           [[f"**{LABELS[i]}**"] + [f"{v:,}" for v in cm[i]] for i in range(len(LABELS))])
    return stats, R, seeds, mean, std, T


def pct(x):
    return f"{100 * x:.1f}%"


def write():
    stats, R, seeds, mean, std, T = build()
    main = R["proposed_seed42"]
    acc = {k: v["accuracy"] for k, v in R.items()}
    model_keys = [k for k in ("proposed_seed42", "baseline_cnn", "baseline_bilstm", "baseline_transformer",
                              "pretrained_mbert", "pretrained_xlmr") if k in R]
    best_model = max(model_keys, key=lambda k: acc[k])
    pp = main["per_pair_accuracy"]
    worst_pair, best_pair = min(pp, key=pp.get), max(pp, key=pp.get)
    cm = np.array(main["confusion_matrix"])
    off = cm - np.diag(np.diag(cm))
    ei, ej = np.unravel_index(off.argmax(), off.shape)
    pretrained_done = all(k in R for k in ("pretrained_mbert", "pretrained_xlmr"))
    grid_done = [k for k in R if k.startswith("grid_")]
    grid_best = max(grid_done, key=lambda k: acc[k]) if grid_done else None
    adaptive = [acc["proposed_seed42"], acc["optimizer_adamw"], acc["optimizer_rmsprop"]]
    cnn_hurts = acc["baseline_bilstm"] > mean["accuracy"]

    md = f"""# Results report: CNN-BiLSTM-Adam for token-level code-mixed language identification

All numbers below come from real training runs in this repository (`results/runs/*.json`) and were
written into this report by `src/make_report.py`. Nothing was tuned toward the numbers in the paper.

## 1. Summary

- **Proposed CNN-BiLSTM-Adam:** **{pct(mean['accuracy'])} ± {100 * std['accuracy']:.1f}** token-level test accuracy
  over 5 random seeds (weighted F1 {mean['f1']:.3f}, precision {mean['precision']:.3f}, recall {mean['recall']:.3f},
  AUC-ROC {mean['auc_roc']:.3f}). The paper reports 92.6% accuracy, F1 0.925, precision 0.926 and recall 0.924, so our
  reproduction is within about {abs(100 * (mean['accuracy'] - 0.926)):.1f} points of the headline number.
- **Best model in our runs:** `{best_model}` at {pct(acc[best_model])}.
"""
    if cnn_hurts:
        md += ("- **The BiLSTM-only model beat the CNN-BiLSTM**, so the CNN layer did not help on this data "
               "(see section 5).\n")
    md += f"""- **Adam vs SGD:** Adam ({pct(acc['proposed_seed42'])}) beats SGD ({pct(acc['optimizer_sgd'])}) by a wide
  margin at the same learning rate. Adam, AdamW ({pct(acc['optimizer_adamw'])}) and RMSprop
  ({pct(acc['optimizer_rmsprop'])}) end up close together.
- **Language pairs:** easiest is {best_pair} ({pct(pp[best_pair])}); hardest is {worst_pair} ({pct(pp[worst_pair])}).

## 2. Data

The authors' annotated corpus is not public, so the dataset was rebuilt from the sources listed in the paper's Table 1.
**Synthetic parts are stated as synthetic, and must be described that way in the paper.**

{T['dataset']}

- **Real data:**
  - LinCE Hindi-English (gold token labels, social media).
  - The authors' English-Kokborok corpus (`Code mixing and Code switching NEW DATASET 2023.docx`, 6,986 unique
    sentences).
- **English-Kokborok labels are silver labels**, made automatically by dictionary lookup against the
  English-Kokborok-Bengali Dictionary (Binoy Debbarma; 11k English headwords and 12.5k Kokborok words), with
  context rules for words found in both lists. They have not been checked by a native speaker.
- **Synthetic data:** sentences from FLORES-200 and MCS-350 (single-language) were mixed by inserting 1–3-word spans
  of one language into a sentence of the other, at the paper's code-mixing ratios. Kokborok-Bengali sentences were
  made by replacing the English spans of real Kokborok sentences with romanized Bengali spans.
- Hindi and Bengali were **romanized** (Latin script, as on social media); native scripts would make the task trivial.
- **No leakage:** every source sentence was assigned to train/val/test (80/10/10) *before* mixing, so no source
  sentence appears in more than one split.

{T['splits']}

Test set: {stats['test_tokens']:,} tokens.

![Dataset](figures/fig_dataset.png)

## 3. Method

Same as the paper's Table 5: embedding 300 (trainable) → Conv1D 128 filters, kernel 3, ReLU → max-pooling →
BiLSTM 128 → dropout 0.5 → softmax over 5 languages per token. Training: Adam (lr 0.001, β1 0.9, β2 0.999, ε 1e-8),
batch 64, up to 50 epochs, early stopping on validation loss (patience 5), cross-entropy loss.

Deviations from the paper, and why:
1. **Max-pooling uses stride 1** (kernel 3, padding 1). The paper's pooling over the whole sequence would leave one
   vector per sentence, which makes per-token labels impossible.
2. **PyTorch instead of Keras.** TensorFlow wasn't installed; the architecture is the same.
3. **GPU:** NVIDIA RTX 3050 Laptop (6 GB). Training times are not comparable to the paper's hardware.
4. **SGD in the optimizer comparison uses momentum 0.9.** As in the paper, every optimizer uses the same learning
   rate (0.001).

## 4. Main result and stability across seeds (paper Table 10)

{T['seeds']}

Across 5 seeds the accuracy spread is ±{100 * std['accuracy']:.1f} points. The paper reports ±0.2, which is tighter
than what we observed.

![Learning curves](figures/fig_learning_curves.png)

## 5. Comparison with baselines (paper Table 6, Fig. 11)

{T['models']}

![Model comparison](figures/fig_model_comparison.png)
![AUC-ROC](figures/fig_auc_roc.png)
![Training time](figures/fig_training_time.png)

**Explanation.**
- The **CNN alone** sees only a 3-word window, so it cannot use wider sentence context to decide language
  at switch points. It is the weakest model ({pct(acc['baseline_cnn'])}).
- **BiLSTM-based models** see the whole sentence in both directions and do much better.
- **The BiLSTM without a CNN front-end scored {pct(acc['baseline_bilstm'])}**, which is
  {"higher" if cnn_hurts else "lower"} than the CNN-BiLSTM's {pct(mean['accuracy'])} mean. The likely reason is that
  the convolution plus max-pooling mixes each word's features with its neighbours'. That blurs the signal exactly at
  language-switch points, where neighbouring words belong to different languages.
"""
    if cnn_hurts:
        md += '- **This contradicts the paper\'s ablation** ("Without CNN: 0.880") and should be reported honestly.\n'
    md += f"""- The **Transformer trained from scratch** ({pct(acc['baseline_transformer'])}) has no pretraining and only
  about 38k sentences, so it underperforms the recurrent models.
"""
    if pretrained_done:
        md += f"""- **Fine-tuned mBERT ({pct(acc['pretrained_mbert'])}) and XLM-R ({pct(acc['pretrained_xlmr'])})** use
  sub-word tokenizers, so unseen words are not mapped to a single unknown token. They are also pretrained on Hindi,
  Bengali, Spanish and English text. They take far longer to train
  ({R['pretrained_mbert']['train_time_s']:.0f} s and {R['pretrained_xlmr']['train_time_s']:.0f} s vs
  {main['train_time_s']:.0f} s).
"""
    else:
        import re
        logtxt = (ROOT / "results" / "log.txt").read_text(encoding="utf-8")
        partial = re.findall(r"\[pretrained_mbert\] epoch (\d+) .*val_acc ([0-9.]+)", logtxt)
        md += """- **Pretrained mBERT / XLM-R (paper Fig. 11): not run.** Fine-tuning them on the laptop GPU needed
  4–6 hours because of thermal throttling, which didn't fit the submission deadline. So this report gives
  **no test-set numbers for them**, and the paper's claim that the proposed model beats BERT, mBERT and XLM-R is
  **not verified here**.
"""
        if partial:
            ep, va = partial[-1]
            md += f"""- *Preliminary observation (not a final result):* an interrupted mBERT run reached
  **{100 * float(va):.1f}% validation accuracy after {ep} epoch(s)**, which is above the proposed model. If this holds on
  the test set, fine-tuned multilingual transformers would **outperform** the CNN-BiLSTM. Before the paper claims
  the opposite, these runs should be completed (`python src/run_experiments.py pretrained`).
"""

    md += f"""
## 6. Optimizer comparison (paper Table 9)

{T['optimizers']}

![Optimizers](figures/fig_optimizers.png)

**Explanation.** At the same learning rate of 0.001:
- **SGD** converges far more slowly. After 50 epochs it is still only at {pct(acc['optimizer_sgd'])}.
- **Adaptive methods (Adam, AdamW, RMSprop)** scale each parameter's step size. This matters for the embedding
  matrix, where most rows (rare words) receive a gradient only occasionally. These are the "sparse gradients" the
  paper mentions.
- The three adaptive methods are within {100 * (max(adaptive) - min(adaptive)):.1f} points of each other. That is
  inside the seed-to-seed variation (±{100 * std['accuracy']:.1f}), so **Adam is not clearly better than AdamW or
  RMSprop on this data.**

## 7. Ablation study (paper Table 11)

{T['ablation']}

![Ablation](figures/fig_ablation.png)

**Explanation.**
- **Removing the BiLSTM** changes accuracy by {100 * (acc['baseline_cnn'] - acc['proposed_seed42']):+.1f} points.
  Sentence context is the most important architectural ingredient.
- **Replacing Adam by SGD** changes it by {100 * (acc['optimizer_sgd'] - acc['proposed_seed42']):+.1f} points.
- **Removing the CNN** changes it by {100 * (acc['baseline_bilstm'] - acc['proposed_seed42']):+.1f} points.

## 8. Learning rate × batch size (paper Fig. 9/10)

{T['grid']}

![Learning rate and batch size](figures/fig_lr_batch.png)

Runs are capped at 20 epochs with early stopping.
- With lr 1e-5 the model is still far from converged after 20 epochs.
- Larger learning rates converge much faster.
"""
    if grid_best:
        md += f"- Best setting in the grid: `{grid_best}` ({pct(acc[grid_best])}).\n"
    md += f"""
## 9. Error analysis

Per language (proposed model, seed 42):

{T['per_label']}

Per language pair:

{T['per_pair']}

![Per pair](figures/fig_per_pair.png)

Confusion matrix (tokens, test set):

{T['confusion']}

![Confusion matrix](figures/fig_confusion_matrix.png)

The most frequent error is **{LABELS[ei]} predicted as {LABELS[ej]}** ({cm[ei, ej]:,} tokens). The main causes:
1. **Words shared between languages:** *a, no, me, come, ta, na, ki* exist in more than one language, and in
   synthetic sentences the context is not always grammatical enough to disambiguate them.
2. **Romanization:** romanized Hindi and Bengali have overlapping spellings (*ke, ki, ar, kare*), which explains most
   Bengali↔Hindi errors.
3. **Unknown words:** {pct(main['test_oov_rate'])} of test tokens never occur in training. The model sees them as a
   single unknown token and has to rely only on context. A character-level component would help, which is a
   natural improvement.
4. **Silver labels in the Kokborok data:** some errors there are labeling noise, not model errors.

## 10. Our results vs. the paper's claims

| Claim in the paper | Paper | This reproduction | Holds? |
| --- | --- | --- | --- |
| CNN-BiLSTM-Adam accuracy | 0.926 | {mean['accuracy']:.3f} ± {std['accuracy']:.3f} | {"Yes (within seed noise)" if abs(mean['accuracy'] - 0.926) <= 2 * std['accuracy'] else "No"} |
| Proposed model beats all baselines | yes | best is `{best_model}` | {"Yes" if best_model == "proposed_seed42" else "No"} |
| CNN component helps (ablation) | +4.6 pts | {100 * (acc['proposed_seed42'] - acc['baseline_bilstm']):+.1f} pts | {"Yes" if not cnn_hurts else "No"} |
| BiLSTM component helps | +5.6 pts | {100 * (acc['proposed_seed42'] - acc['baseline_cnn']):+.1f} pts | {"Yes" if acc['proposed_seed42'] > acc['baseline_cnn'] else "No"} |
| Adam beats SGD | +4.4 pts | {100 * (acc['proposed_seed42'] - acc['optimizer_sgd']):+.1f} pts | {"Yes" if acc['proposed_seed42'] > acc['optimizer_sgd'] else "No"} |
| Adam beats AdamW / RMSprop | +1.1 / +2.5 pts | {100 * (acc['proposed_seed42'] - acc['optimizer_adamw']):+.1f} / {100 * (acc['proposed_seed42'] - acc['optimizer_rmsprop']):+.1f} pts | {"Yes" if acc['proposed_seed42'] > max(acc['optimizer_adamw'], acc['optimizer_rmsprop']) else "No (within seed noise)"} |
| Seed standard deviation | 0.002 | {std['accuracy']:.3f} | {"Yes" if std['accuracy'] <= 0.004 else "No (larger spread)"} |
| Proposed beats BERT / mBERT / XLM-R | 0.926 vs 0.90–0.91 | {"see section 5" if pretrained_done else "not run"} | {"see section 5" if pretrained_done else "Not verified"} |

## 11. Limitations

1. {100 * sum(d['synthetic'] for d in stats['pairs'].values()) / stats['total_samples']:.0f}% of the dataset is
   synthetic. Synthetic mixing does not follow real code-switching grammar.
2. English-Kokborok labels are silver (automatic) and need native-speaker checking.
3. No real Kokborok-Bengali code-mixed text was available.
4. Max-pooling with stride 1 is our interpretation of the paper's architecture.
5. Training times depend on hardware. One laptop GPU was used, with cooling pauses that are excluded from the
   reported times.
6. Pretrained transformer baselines (mBERT, XLM-R) were not completed because of time and hardware limits.

## 12. How to reproduce

```
python src/prepare_data.py      # builds data/processed/
python src/run_experiments.py   # all runs -> results/runs/*.json (resumable)
python src/make_figures.py      # results/figures/*.png
python src/make_report.py       # this report
```
"""
    OUT.write_text(md, encoding="utf-8")
    print("wrote", OUT)


if __name__ == "__main__":
    write()
