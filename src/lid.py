"""Models, data loading, training and token-level evaluation for code-mixed LID."""
import json
import random
import subprocess
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (accuracy_score, confusion_matrix, matthews_corrcoef,
                             precision_recall_fscore_support, roc_auc_score)

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed"
LABELS = ["English", "Hindi", "Bengali", "Kokborok", "Spanish"]
L2I = {l: i for i, l in enumerate(LABELS)}
PAD, UNK = 0, 1
IGNORE = -100
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Table 5 of the paper
DEFAULTS = dict(emb_dim=300, filters=128, kernel=3, hidden=128, max_len=128, dropout=0.5,
                batch_size=64, epochs=50, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, patience=5)


def gpu_temp():
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=temperature.gpu", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=10).stdout
        return int(out.strip().splitlines()[0])
    except Exception:
        return None


def thermal_guard(log=print, high=78, resume=65):
    """Pause (laptop cooling) while the GPU is hotter than `high` until it drops to `resume`.

    Returns the seconds spent paused so it can be excluded from training time.
    """
    t = gpu_temp()
    if t is None or t < high:
        return 0.0
    start = time.time()
    log(f"  GPU at {t}C - pausing to cool down")
    while t is not None and t > resume:
        time.sleep(30)
        t = gpu_temp()
    log(f"  GPU back to {t}C - continuing")
    return time.time() - start


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_split(name):
    with open(DATA / f"{name}.jsonl", encoding="utf-8") as f:
        return [json.loads(l) for l in f]


def build_vocab(records):
    counts = Counter(t for r in records for t in r["tokens"])
    itos = ["<pad>", "<unk>"] + sorted(counts)
    return {w: i for i, w in enumerate(itos)}


def encode(records, vocab, max_len):
    out = []
    for r in records:
        ids = [vocab.get(t, UNK) for t in r["tokens"][:max_len]]
        labs = [L2I[l] for l in r["labels"][:max_len]]
        out.append((ids, labs, r["pair"]))
    return out


def batches(encoded, batch_size, shuffle, rng=None):
    idx = list(range(len(encoded)))
    if shuffle:
        rng.shuffle(idx)
    for s in range(0, len(idx), batch_size):
        chunk = [encoded[i] for i in idx[s:s + batch_size]]
        n = max(len(c[0]) for c in chunk)
        x = torch.full((len(chunk), n), PAD, dtype=torch.long)
        y = torch.full((len(chunk), n), IGNORE, dtype=torch.long)
        for j, (ids, labs, _) in enumerate(chunk):
            x[j, :len(ids)] = torch.tensor(ids)
            y[j, :len(labs)] = torch.tensor(labs)
        yield x.to(DEVICE), y.to(DEVICE), [c[2] for c in chunk]


# ---------------------------------------------------------------- models
class TokenTagger(nn.Module):
    """Embedding -> [CNN + max-pool] -> [BiLSTM] -> per-token softmax.

    use_cnn/use_lstm switch the components off for the baselines and ablations.
    Max-pooling uses kernel 3, stride 1, so sequence length (one label per token) is preserved.
    """

    def __init__(self, vocab_size, use_cnn=True, use_lstm=True, emb_dim=300, filters=128,
                 kernel=3, hidden=128, dropout=0.5, n_labels=len(LABELS), **_):
        super().__init__()
        self.use_cnn, self.use_lstm = use_cnn, use_lstm
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=PAD)
        dim = emb_dim
        if use_cnn:
            self.conv = nn.Conv1d(emb_dim, filters, kernel, padding=kernel // 2)
            self.pool = nn.MaxPool1d(kernel, stride=1, padding=kernel // 2)
            dim = filters
        self.drop = nn.Dropout(dropout)
        if use_lstm:
            self.lstm = nn.LSTM(dim, hidden, batch_first=True, bidirectional=True)
            dim = 2 * hidden
        self.fc = nn.Linear(dim, n_labels)

    def forward(self, x):
        h = self.emb(x)
        if self.use_cnn:
            h = self.pool(torch.relu(self.conv(h.transpose(1, 2)))).transpose(1, 2)
        h = self.drop(h)
        if self.use_lstm:
            h, _ = self.lstm(h)
            h = self.drop(h)
        return self.fc(h)


class TransformerTagger(nn.Module):
    """Small Transformer encoder trained from scratch (the 'Transformer-Based' baseline)."""

    def __init__(self, vocab_size, emb_dim=300, dropout=0.1, max_len=128, n_labels=len(LABELS),
                 d_model=256, heads=4, layers=2, **_):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=PAD)
        self.proj = nn.Linear(emb_dim, d_model)
        self.pos = nn.Embedding(max_len, d_model)
        layer = nn.TransformerEncoderLayer(d_model, heads, 4 * d_model, dropout, batch_first=True)
        self.enc = nn.TransformerEncoder(layer, layers)
        self.fc = nn.Linear(d_model, n_labels)

    def forward(self, x):
        pos = torch.arange(x.size(1), device=x.device).unsqueeze(0)
        h = self.proj(self.emb(x)) + self.pos(pos)
        h = self.enc(h, src_key_padding_mask=x.eq(PAD))
        return self.fc(h)


def make_optimizer(name, params, cfg):
    if name == "adam":
        return torch.optim.Adam(params, lr=cfg["lr"], betas=cfg["betas"], eps=cfg["eps"])
    if name == "adamw":
        return torch.optim.AdamW(params, lr=cfg["lr"], betas=cfg["betas"], eps=cfg["eps"], weight_decay=0.01)
    if name == "rmsprop":
        return torch.optim.RMSprop(params, lr=cfg["lr"])
    if name == "sgd":
        return torch.optim.SGD(params, lr=cfg["lr"], momentum=0.9)
    raise ValueError(name)


# ---------------------------------------------------------------- evaluation
def compute_metrics(y_true, y_pred, probs, pairs=None):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    p_w, r_w, f_w, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted", zero_division=0)
    p_m, r_m, f_m, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    per = precision_recall_fscore_support(y_true, y_pred, labels=range(len(LABELS)), zero_division=0)
    m = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": p_w, "recall": r_w, "f1": f_w,
        "precision_macro": p_m, "recall_macro": r_m, "f1_macro": f_m,
        "mcc": matthews_corrcoef(y_true, y_pred),
        "auc_roc": roc_auc_score(y_true, probs, multi_class="ovr", average="macro", labels=range(len(LABELS))),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=range(len(LABELS))).tolist(),
        "per_label": {LABELS[i]: {"precision": per[0][i], "recall": per[1][i], "f1": per[2][i],
                                  "support": int(per[3][i])} for i in range(len(LABELS))},
        "n_tokens": int(len(y_true)),
    }
    if pairs is not None:
        pairs = np.asarray(pairs)
        m["per_pair_accuracy"] = {p: accuracy_score(y_true[pairs == p], y_pred[pairs == p])
                                  for p in sorted(set(pairs))}
    return {k: (float(v) if isinstance(v, (np.floating, float)) else v) for k, v in m.items()}


@torch.no_grad()
def predict(model, encoded, batch_size=256):
    model.eval()
    y_true, y_pred, probs, pairs, loss_sum, n = [], [], [], [], 0.0, 0
    lossf = nn.CrossEntropyLoss(ignore_index=IGNORE, reduction="sum")
    for x, y, pr in batches(encoded, batch_size, shuffle=False):
        logits = model(x)
        loss_sum += lossf(logits.reshape(-1, logits.size(-1)), y.reshape(-1)).item()
        mask = y.ne(IGNORE)
        n += mask.sum().item()
        p = torch.softmax(logits, -1)[mask]
        y_true += y[mask].tolist()
        y_pred += p.argmax(-1).tolist()
        probs.append(p.cpu().numpy())
        pairs += [pr[i] for i in range(len(pr)) for _ in range(int(mask[i].sum()))]
    return y_true, y_pred, np.concatenate(probs), pairs, loss_sum / n


# ---------------------------------------------------------------- training
def train_and_eval(name, arch="cnn_bilstm", optimizer="adam", seed=42, log=print, **overrides):
    cfg = {**DEFAULTS, **overrides}
    set_seed(seed)
    train, val, test = load_split("train"), load_split("val"), load_split("test")
    vocab = build_vocab(train)
    enc = {k: encode(v, vocab, cfg["max_len"]) for k, v in (("train", train), ("val", val), ("test", test))}

    if arch == "transformer":
        model = TransformerTagger(len(vocab), **{**cfg, "dropout": 0.1})
    else:
        model = TokenTagger(len(vocab), use_cnn=arch in ("cnn_bilstm", "cnn"),
                            use_lstm=arch in ("cnn_bilstm", "bilstm"), **cfg)
    model.to(DEVICE)
    opt = make_optimizer(optimizer, model.parameters(), cfg)
    lossf = nn.CrossEntropyLoss(ignore_index=IGNORE)
    rng = random.Random(seed)

    best, best_state, bad, history = float("inf"), None, 0, []
    t0, paused = time.time(), 0.0
    for epoch in range(1, cfg["epochs"] + 1):
        model.train()
        tot, nb = 0.0, 0
        for x, y, _ in batches(enc["train"], cfg["batch_size"], shuffle=True, rng=rng):
            opt.zero_grad()
            logits = model(x)
            loss = lossf(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
            loss.backward()
            opt.step()
            tot, nb = tot + loss.item(), nb + 1
            if nb % 100 == 0:
                paused += thermal_guard(log)
        yt, yp, _, _, vloss = predict(model, enc["val"])
        vacc = float(np.mean(np.asarray(yt) == np.asarray(yp)))
        history.append({"epoch": epoch, "train_loss": tot / nb, "val_loss": vloss, "val_acc": vacc})
        log(f"[{name}] epoch {epoch:2d} train_loss {tot / nb:.4f} val_loss {vloss:.4f} val_acc {vacc:.4f}")
        if vloss < best - 1e-4:
            best, bad = vloss, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= cfg["patience"]:
                log(f"[{name}] early stop at epoch {epoch}")
                break
    train_time = time.time() - t0 - paused
    model.load_state_dict(best_state)

    yt, yp, probs, pairs, _ = predict(model, enc["test"])
    m = compute_metrics(yt, yp, probs, pairs)
    oov = sum(t == UNK for ids, _, _ in enc["test"] for t in ids) / sum(len(ids) for ids, _, _ in enc["test"])
    m.update(name=name, arch=arch, optimizer=optimizer, seed=seed, train_time_s=train_time,
             epochs_run=len(history), best_val_loss=best, history=history, test_oov_rate=oov,
             config={k: v for k, v in cfg.items()},
             params=sum(p.numel() for p in model.parameters()))
    return m
