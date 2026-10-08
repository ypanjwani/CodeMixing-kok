# Code-mixed Language Identification — reproduction

Token-level LID (English, Hindi, Bengali, Kokborok, Spanish) with a CNN-BiLSTM, baselines,
optimizer/LR/batch-size sweeps and pretrained mBERT / XLM-R, reproducing the paper's experiments.

## Setup

Python 3.8. A CUDA GPU is used if available (CPU works, just slower).

```
pip install torch transformers scikit-learn numpy pandas matplotlib python-docx indic-transliteration pyarrow
```

Raw data goes in `data/raw/` (`dictionary/`, `flores200/`, `hineng/`, `mcs350/`, `spaeng/`).

## Run

All commands run from `src/`:

```
cd src
python prepare_data.py      # data/raw -> data/processed/{train,val,test}.jsonl + stats.json
python run_experiments.py   # trains everything -> results/runs/<name>.json
python make_figures.py      # -> results/figures/*.png
python make_report.py       # -> results/REPORT.md
python make_artifact.py     # -> results/report.html
```

`run_experiments.py` is resumable: experiments that already have a JSON in `results/runs/` are
skipped, so delete a JSON to re-run it. Progress is written to `results/log.txt`.

To run only some experiments, pass name prefixes:

```
python run_experiments.py proposed            # 5 seeds of the proposed model
python run_experiments.py baseline optimizer  # baselines + optimizer comparison
python run_experiments.py grid                # LR x batch-size grid
python run_experiments.py pretrained          # mBERT + XLM-R fine-tuning
```

Training pauses automatically if the GPU gets hotter than 70 °C and resumes at 65 °C.
