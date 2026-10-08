"""Fine-tune pretrained multilingual transformers (mBERT, XLM-R) for token-level LID."""
import random
import time

import numpy as np
import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer, get_linear_schedule_with_warmup

from lid import DEVICE, IGNORE, L2I, LABELS, compute_metrics, load_split, set_seed, thermal_guard


def encode(records, tok, max_len=128):
    out = []
    for r in records:
        enc = tok(r["tokens"], is_split_into_words=True, truncation=True, max_length=max_len)
        labels, prev = [], None
        for wid in enc.word_ids():
            # label only the first sub-word of each word
            labels.append(IGNORE if wid is None or wid == prev else L2I[r["labels"][wid]])
            prev = wid
        out.append((enc["input_ids"], labels, r["pair"]))
    return out


def batches(encoded, bs, pad_id, shuffle, rng=None):
    idx = list(range(len(encoded)))
    if shuffle:
        rng.shuffle(idx)
    for s in range(0, len(idx), bs):
        chunk = [encoded[i] for i in idx[s:s + bs]]
        n = max(len(c[0]) for c in chunk)
        x = torch.full((len(chunk), n), pad_id, dtype=torch.long)
        y = torch.full((len(chunk), n), IGNORE, dtype=torch.long)
        for j, (ids, labs, _) in enumerate(chunk):
            x[j, :len(ids)] = torch.tensor(ids)
            y[j, :len(labs)] = torch.tensor(labs)
        yield x.to(DEVICE), x.ne(pad_id).to(DEVICE), y.to(DEVICE), [c[2] for c in chunk]


@torch.no_grad()
def predict(model, encoded, pad_id, bs=64):
    model.eval()
    yt, yp, probs, pairs, loss_sum, n = [], [], [], [], 0.0, 0
    for x, att, y, pr in batches(encoded, bs, pad_id, shuffle=False):
        with torch.autocast("cuda", dtype=torch.float16):
            logits = model(input_ids=x, attention_mask=att).logits.float()
        mask = y.ne(IGNORE)
        loss_sum += torch.nn.functional.cross_entropy(logits[mask], y[mask], reduction="sum").item()
        n += mask.sum().item()
        p = torch.softmax(logits, -1)[mask]
        yt += y[mask].tolist()
        yp += p.argmax(-1).tolist()
        probs.append(p.cpu().numpy())
        pairs += [pr[i] for i in range(len(pr)) for _ in range(int(mask[i].sum()))]
    return yt, yp, np.concatenate(probs), pairs, loss_sum / n


def finetune(name, checkpoint, seed=42, epochs=3, lr=2e-5, bs=16, log=print):
    set_seed(seed)
    tok = AutoTokenizer.from_pretrained(checkpoint)
    model = AutoModelForTokenClassification.from_pretrained(checkpoint, num_labels=len(LABELS)).to(DEVICE)
    enc = {k: encode(load_split(k), tok) for k in ("train", "val", "test")}
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    steps = epochs * ((len(enc["train"]) + bs - 1) // bs)
    sched = get_linear_schedule_with_warmup(opt, int(0.1 * steps), steps)
    scaler = torch.cuda.amp.GradScaler()
    rng = random.Random(seed)

    best, best_state, history = float("inf"), None, []
    t0, paused = time.time(), 0.0
    for epoch in range(1, epochs + 1):
        model.train()
        tot, nb = 0.0, 0
        for x, att, y, _ in batches(enc["train"], bs, tok.pad_token_id, shuffle=True, rng=rng):
            opt.zero_grad()
            with torch.autocast("cuda", dtype=torch.float16):
                loss = model(input_ids=x, attention_mask=att, labels=y).loss
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            tot, nb = tot + loss.item(), nb + 1
            if nb % 100 == 0:
                paused += thermal_guard(log)
        yt, yp, _, _, vloss = predict(model, enc["val"], tok.pad_token_id)
        vacc = float(np.mean(np.asarray(yt) == np.asarray(yp)))
        history.append({"epoch": epoch, "train_loss": tot / nb, "val_loss": vloss, "val_acc": vacc})
        log(f"[{name}] epoch {epoch} train_loss {tot / nb:.4f} val_loss {vloss:.4f} val_acc {vacc:.4f}")
        if vloss < best:
            best = vloss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    train_time = time.time() - t0 - paused
    model.load_state_dict(best_state)

    yt, yp, probs, pairs, _ = predict(model, enc["test"], tok.pad_token_id)
    m = compute_metrics(yt, yp, probs, pairs)
    m.update(name=name, arch=checkpoint, optimizer="adamw", seed=seed, train_time_s=train_time,
             epochs_run=len(history), best_val_loss=best, history=history,
             config={"lr": lr, "batch_size": bs, "epochs": epochs, "max_len": 128},
             params=sum(p.numel() for p in model.parameters()))
    del model
    torch.cuda.empty_cache()
    return m
