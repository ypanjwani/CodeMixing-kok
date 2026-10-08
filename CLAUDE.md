# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A reproduction of a paper's experiments (`Scientific_Report (1).pdf`) on token-level code-mixed language
identification across 5 labels (English, Hindi, Bengali, Kokborok, Spanish) and 5 language pairs. The
proposed model is a CNN-BiLSTM tagger; it's compared against baselines, optimizer/LR/batch-size sweeps,
and fine-tuned mBERT / XLM-R. The goal is honest reproduction: reported numbers must come from real runs,
never tuned toward the paper's claims (the paper's figures live only in `PAPER` in `src/make_report.py`,
for side-by-side comparison).

## Commands

Python 3.8, no test suite or linter. All scripts are run **from `src/`** (they import each other as
top-level modules; paths resolve via `ROOT = Path(__file__).parents[1]`).

```
cd src
python prepare_data.py      # data/raw -> data/processed/{train,val,test}.jsonl + stats.json
python run_experiments.py   # trains everything -> results/runs/<name>.json (resumable)
python make_figures.py      # -> results/figures/*.png
python make_report.py       # -> results/REPORT.md
python make_artifact.py     # -> results/report.html
```

Run a subset by name prefix: `python run_experiments.py proposed` (also `baseline`, `optimizer`, `grid`,
`pretrained`, or a full name like `proposed_seed42`). Experiments with an existing JSON in `results/runs/`
are skipped — delete the JSON to re-run. Progress is appended to `results/log.txt`.

## Architecture

Pipeline: `prepare_data.py` → `run_experiments.py` (uses `lid.py` / `pretrained.py`) → reporting scripts
that read only `results/runs/*.json` and `data/processed/stats.json`.

- **`prepare_data.py`** builds the dataset to match the paper's Table 2 targets (`TARGETS`: sample count,
  avg words, mixing ratio per pair). Only part of it is real: Hindi-English uses LinCE (`data/raw/hineng`)
  topped up with synthetic data; Spanish-English and Bengali-English are fully synthetic (monolingual
  FLORES-200/MCS-350 sentences spliced together by `mix_sentence`); English-Kokborok is real sentences from
  the `.docx` in the repo root with silver labels from the dictionary + English heuristics
  (`label_kokborok`); Kokborok-Bengali is synthetic from those Kokborok sentences. Every record carries
  `source: "real"|"synthetic"` and splits are made per source pool to avoid train/test leakage (it prints
  the real-sentence overlap). Hindi/Bengali are romanized from native script via ITRANS.
- **`lid.py`** holds everything for from-scratch models: vocab/encoding, `TokenTagger` (CNN and/or BiLSTM
  toggled by `arch` — `cnn`/`bilstm` baselines double as ablations), `TransformerTagger`, optimizer factory,
  `train_and_eval` (early stopping, `DEFAULTS` = paper's Table 5 hyperparameters, overridable via kwargs),
  and `compute_metrics`, which defines the JSON schema every downstream script reads (accuracy, weighted
  and macro P/R/F1, MCC, macro OvR AUC, confusion matrix, per-label and per-pair metrics, train time).
- **`pretrained.py`** fine-tunes HF token-classification models, labelling only the first sub-word of each
  word, and returns the same metrics dict.
- **`run_experiments.py`** is the single registry of experiments (`EXPERIMENTS` dict + `PRETRAINED`); add
  new runs there.
- **Thermal guard**: `lid.thermal_guard` polls `nvidia-smi` and pauses while the GPU is hot (laptop GPU);
  paused time is excluded from `train_time_s`. `run_experiments.py` calls it between runs at 70/65 °C.
- **Reporting**: `make_report.build()` computes the aggregates (e.g. mean over the 5 proposed seeds) and is
  reused by `make_artifact.py`, which renders a self-contained HTML report with inline SVG charts.
  Reporting scripts must tolerate missing runs (e.g. the `pretrained_*` runs may not exist yet).

## Data

`data/raw/` holds `dictionary/`, `flores200/`, `hineng/` (LinCE parquet), `mcs350/` (and optionally
`spaeng/`). The root-level `.docx`, `.doc`, and `.xlsx` files are the original Kokborok source material.
On OOM or slow training, the preferred fix is seeded random subsampling of the dataset rather than
shrinking the model.
