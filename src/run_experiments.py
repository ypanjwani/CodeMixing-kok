"""Run every experiment from the paper; each result goes to results/runs/<name>.json (resumable)."""
import json
import sys
from pathlib import Path

from lid import ROOT, thermal_guard, train_and_eval

RUNS = ROOT / "results" / "runs"
LOG = ROOT / "results" / "log.txt"

# name -> kwargs for train_and_eval
EXPERIMENTS = {}
# 1. proposed model, five seeds (Table 10)
for s in (42, 123, 256, 512, 1024):
    EXPERIMENTS[f"proposed_seed{s}"] = dict(arch="cnn_bilstm", optimizer="adam", seed=s)
# 2. baselines trained from scratch (Table 6); CNN-only / BiLSTM-only double as ablations (Table 11)
EXPERIMENTS["baseline_cnn"] = dict(arch="cnn")
EXPERIMENTS["baseline_bilstm"] = dict(arch="bilstm")
EXPERIMENTS["baseline_transformer"] = dict(arch="transformer", lr=1e-4)
# 3. optimizer comparison (Table 9), same lr and everything else
for o in ("sgd", "rmsprop", "adamw"):
    EXPERIMENTS[f"optimizer_{o}"] = dict(arch="cnn_bilstm", optimizer=o)
# 4. learning-rate x batch-size grid (Fig. 9/10), capped at 20 epochs
for lr in (1e-5, 1e-4, 1e-3):
    for bs in (32, 64, 128):
        EXPERIMENTS[f"grid_lr{lr:g}_bs{bs}"] = dict(arch="cnn_bilstm", lr=lr, batch_size=bs, epochs=20)
# 5. pretrained transformers (Fig. 11) - handled by pretrained.py
PRETRAINED = {"pretrained_mbert": "bert-base-multilingual-cased",
              "pretrained_xlmr": "xlm-roberta-base"}


def log(msg):
    print(msg, flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(msg + "\n")


def main(only=None):
    RUNS.mkdir(parents=True, exist_ok=True)
    todo = list(EXPERIMENTS) + list(PRETRAINED)
    if only:
        todo = [n for n in todo if any(n.startswith(o) for o in only)]
    for name in todo:
        out = RUNS / f"{name}.json"
        if out.exists():
            log(f"skip {name} (done)")
            continue
        thermal_guard(log, high=70, resume=65)
        log(f"=== start {name}")
        if name in PRETRAINED:
            from pretrained import finetune
            m = finetune(name, PRETRAINED[name], log=log)
        else:
            m = train_and_eval(name, log=log, **EXPERIMENTS[name])
        out.write_text(json.dumps(m, indent=2), encoding="utf-8")
        log(f"=== done {name}: acc {m['accuracy']:.4f} f1 {m['f1']:.4f} auc {m['auc_roc']:.4f} "
            f"time {m['train_time_s']:.0f}s")


if __name__ == "__main__":
    main(sys.argv[1:] or None)
