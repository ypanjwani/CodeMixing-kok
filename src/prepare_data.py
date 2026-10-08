"""Build the token-level code-mixed LID dataset (real + disclosed synthetic data).

Outputs data/processed/{train,val,test}.jsonl and data/processed/stats.json.
Each record: {"tokens": [...], "labels": [...], "pair": str, "source": "real"|"synthetic"}
"""
import json
import random
import re
import unicodedata
from collections import Counter
from pathlib import Path

import docx
import pandas as pd
from indic_transliteration import sanscript

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "processed"
SEED = 42
SPLITS = ("train", "val", "test")
SPLIT_FRAC = (0.8, 0.1, 0.1)

# Table 2 of the paper: (pair, target samples, avg words/sample, code-mixing ratio)
TARGETS = {
    "Hindi-English": (17340, 12.5, 0.45),
    "Spanish-English": (10557, 11.8, 0.38),
    "Bengali-English": (8670, 10.2, 0.42),
    "English-Kokborok": (7330, 9.8, 0.40),
    "Kokborok-Bengali": (4500, 9.5, 0.39),
}
LABELS = ["English", "Hindi", "Bengali", "Kokborok", "Spanish"]

URL_RE = re.compile(r"(https?://\S+|www\.\S+|[@#]\S+)")
# High-frequency English function words that also exist as Kokborok forms
EN_FUNCTION = {"i", "a", "to", "the", "is", "you", "me", "my", "we", "he", "she", "it", "in", "on", "at",
               "of", "for", "and", "so", "do", "no", "be", "go", "are", "am", "an", "or", "not", "all"}
EN_MORPH = ("ing", "ed", "es", "s", "er", "ly", "tion", "ment")


# ---------------------------------------------------------------- normalization
def norm_token(tok):
    """Lowercase and keep letters only (accents kept, digits/punctuation dropped)."""
    tok = unicodedata.normalize("NFC", tok.lower())
    return "".join(ch for ch in tok if ch.isalpha())


def tokenize(text):
    text = URL_RE.sub(" ", text.replace("�", "'").replace("’", "'"))
    text = text.replace("'", "")  # don't -> dont
    toks = [norm_token(t) for t in re.split(r"[\s\-/]+", text)]
    return [t for t in toks if t]


def romanize(text, script):
    """Native Devanagari/Bengali -> simple lowercase Latin (social-media style)."""
    itrans = sanscript.transliterate(text, script, sanscript.ITRANS)
    out = []
    for w in tokenize(itrans):
        w = re.sub(r"[^a-z]", "", w.lower())
        # schwa deletion heuristic: bharata -> bharat
        if len(w) > 3 and w.endswith("a") and w[-2] not in "aeiou":
            w = w[:-1]
        if script == sanscript.BENGALI:
            w = w.replace("v", "b")  # Bengali has no /v/: bandhu, not vandhu
        if w:
            out.append(w)
    return out


# ---------------------------------------------------------------- source loading
def read_lines(path):
    return [l.strip() for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def load_monolingual(lang):
    """FLORES-200 (dev+devtest) + MCS-350 sentences for one language, normalized."""
    fl = RAW / "flores200" / "flores200_dataset"
    code = {"eng": "eng_Latn", "hin": "hin_Deva", "ben": "ben_Beng", "spa": "spa_Latn"}[lang]
    lines = read_lines(fl / "dev" / f"{code}.dev") + read_lines(fl / "devtest" / f"{code}.devtest")
    lines += read_lines(RAW / "mcs350" / f"{lang}.txt")
    script = {"hin": sanscript.DEVANAGARI, "ben": sanscript.BENGALI}.get(lang)
    sents = [romanize(l, script) if script else tokenize(l) for l in lines]
    return [s for s in sents if len(s) >= 3]


def load_lince():
    keep = {"lang1": "English", "lang2": "Hindi"}
    out = []
    for split in ("train", "dev"):
        df = pd.read_parquet(RAW / "hineng" / f"{split}.parquet")
        for toks, labs in zip(df.sentence, df.labels):
            t2, l2 = [], []
            for t, l in zip(toks, labs):
                if l in keep and not URL_RE.match(t):
                    nt = norm_token(t)
                    if nt:
                        t2.append(nt)
                        l2.append(keep[l])
            if len(t2) >= 3:
                out.append((t2, l2))
    return out


def load_dictionary():
    """English headwords + Kokborok gloss words from the English-Kokborok-Bengali dictionary."""
    text = (RAW / "dictionary" / "dict.txt").read_text(encoding="utf-8-sig")
    entry = re.compile(r"^([A-Za-z][A-Za-z \-\.']*?)\s*\[[^\]]*\],\s*[a-z\.\s]+?\s*-\s*(.*)$")
    eng, kok = set(), set()
    for line in text.splitlines():
        m = entry.match(line.strip())
        if not m:
            continue
        eng.update(tokenize(m.group(1)))
        latin_part = re.split(r"[-]", m.group(2))[0]  # Bengali (legacy font) follows
        latin_part = re.sub(r"\([^)]*\)", " ", latin_part)  # drop English notes in brackets
        kok.update(tokenize(latin_part))
    return eng, kok


def load_kokborok_docx():
    path = ROOT / "Code mixing and Code switching NEW DATASET 2023 (4).docx"
    d = docx.Document(str(path))
    paras = [p.text for p in d.paragraphs if p.text.strip()][1:]  # skip title line
    seen, out = set(), []
    for p in paras:
        toks = tokenize(p)
        key = " ".join(toks)
        if len(toks) >= 2 and key not in seen:
            seen.add(key)
            out.append(toks)
    return out


# ---------------------------------------------------------------- silver labels for Kokborok data
def english_stem(t, eng_vocab):
    """coding -> code, changes -> change: inflected English word whose stem is known."""
    for suf in EN_MORPH:
        if t.endswith(suf) and len(t) > len(suf) + 2:
            stem = t[: -len(suf)]
            if stem in eng_vocab or stem + "e" in eng_vocab:
                return True
    return False


def label_kokborok(sent, eng_vocab, kok_vocab):
    """Rule-based silver labels: dictionary lookup + context for words in both lists."""
    tags = []
    for t in sent:
        in_e, in_k = t in eng_vocab, t in kok_vocab
        if in_e and not in_k:
            tags.append("English")
        elif in_k and not in_e:
            tags.append("Kokborok")
        elif in_e and in_k:
            tags.append(None)  # resolve from context
        elif english_stem(t, eng_vocab):
            tags.append("English")
        else:
            # unknown words, names, and English stems with Kokborok suffixes (marketo) -> Kokborok
            tags.append("Kokborok")
    for i, t in enumerate(tags):
        if t is None:
            left = next((tags[j] for j in range(i - 1, -1, -1) if tags[j]), None)
            right = next((tags[j] for j in range(i + 1, len(tags)) if tags[j]), None)
            if sent[i] in EN_FUNCTION:
                tags[i] = "English" if "English" in (left, right) else "Kokborok"
            else:
                tags[i] = "English" if (left == "English" and right in ("English", None)) or (
                    right == "English" and left in ("English", None)) else "Kokborok"
    return tags


# ---------------------------------------------------------------- synthetic mixing
def split_pool(items, rng):
    items = list(items)
    rng.shuffle(items)
    n = len(items)
    a, b = int(n * SPLIT_FRAC[0]), int(n * (SPLIT_FRAC[0] + SPLIT_FRAC[1]))
    return {"train": items[:a], "val": items[a:b], "test": items[b:]}


def mix_sentence(rng, mat_sent, mat_lab, emb_pool, emb_lab, avg_len, ratio):
    """Insert 1-3 word spans of the embedded language into a matrix-language window."""
    length = max(4, min(30, round(rng.gauss(avg_len, 3))))
    n_emb = max(1, min(length - 1, round(length * rng.uniform(ratio - 0.1, ratio + 0.1))))
    n_mat = length - n_emb
    start = rng.randint(0, max(0, len(mat_sent) - n_mat))
    tokens = mat_sent[start:start + n_mat]
    labels = [mat_lab] * len(tokens)
    while sum(l == emb_lab for l in labels) < n_emb:
        src = rng.choice(emb_pool)
        span = rng.randint(1, min(3, n_emb - sum(l == emb_lab for l in labels)))
        s = rng.randint(0, max(0, len(src) - span))
        piece = src[s:s + span]
        pos = rng.randint(0, len(tokens))
        tokens[pos:pos] = piece
        labels[pos:pos] = [emb_lab] * len(piece)
    return tokens, labels


def synth_pair(rng, pool_a, lab_a, pool_b, lab_b, n, avg_len, ratio):
    out = []
    for _ in range(n):
        if rng.random() < 0.5:  # either language can be the matrix language
            t, l = mix_sentence(rng, rng.choice(pool_a), lab_a, pool_b, lab_b, avg_len, ratio)
        else:
            t, l = mix_sentence(rng, rng.choice(pool_b), lab_b, pool_a, lab_a, avg_len, ratio)
        out.append((t, l))
    return out


def kok_bengali(rng, kok_items, ben_pool, n):
    """Replace English spans in real Kokborok sentences with romanized Bengali spans."""
    out = []
    for _ in range(n):
        toks, labs = rng.choice(kok_items)
        new_t, new_l, i = [], [], 0
        while i < len(toks):
            if labs[i] == "English":
                j = i
                while j < len(toks) and labs[j] == "English":
                    j += 1
                src = rng.choice(ben_pool)
                span = min(j - i, len(src))
                s = rng.randint(0, len(src) - span)
                new_t += src[s:s + span]
                new_l += ["Bengali"] * span
                i = j
            else:
                new_t.append(toks[i])
                new_l.append(labs[i])
                i += 1
        if "Bengali" not in new_l:  # purely Kokborok sentence: insert a Bengali span
            src = rng.choice(ben_pool)
            span = rng.randint(1, min(3, len(src)))
            s = rng.randint(0, len(src) - span)
            pos = rng.randint(0, len(new_t))
            new_t[pos:pos] = src[s:s + span]
            new_l[pos:pos] = ["Bengali"] * span
        out.append((new_t, new_l))
    return out


def split_counts(total):
    a = round(total * SPLIT_FRAC[0])
    b = round(total * SPLIT_FRAC[1])
    return {"train": a, "val": b, "test": total - a - b}


# ---------------------------------------------------------------- main
def main():
    rng = random.Random(SEED)
    OUT.mkdir(parents=True, exist_ok=True)

    print("loading sources ...")
    mono = {l: split_pool(load_monolingual(l), rng) for l in ("eng", "hin", "ben", "spa")}
    lince = split_pool(load_lince(), rng)
    dict_eng, dict_kok = load_dictionary()
    eng_counts = Counter(t for split in mono["eng"].values() for s in split for t in s)
    eng_counts.update(t for split in lince.values() for s, l in split for t, lab in zip(s, l) if lab == "English")
    eng_vocab = dict_eng | {t for t, c in eng_counts.items() if c >= 2}
    kok_sents = load_kokborok_docx()
    kok_items = [(s, label_kokborok(s, eng_vocab, dict_kok)) for s in kok_sents]
    kok_pool = split_pool(kok_items, rng)
    print(f"  dictionary: {len(dict_eng)} English headwords, {len(dict_kok)} Kokborok words")
    print(f"  docx: {len(kok_sents)} unique English-Kokborok sentences")

    data = {s: [] for s in SPLITS}
    for split in SPLITS:
        def add(pair, items, source):
            data[split].extend({"tokens": t, "labels": l, "pair": pair, "source": source} for t, l in items)

        # Hindi-English: real LinCE + synthetic top-up
        need = split_counts(TARGETS["Hindi-English"][0])[split]
        real = lince[split][:need]
        add("Hindi-English", real, "real")
        _, avg, ratio = TARGETS["Hindi-English"]
        add("Hindi-English", synth_pair(rng, mono["eng"][split], "English", mono["hin"][split], "Hindi",
                                        need - len(real), avg, ratio), "synthetic")
        # Spanish-English / Bengali-English: synthetic
        for pair, lang, lab in (("Spanish-English", "spa", "Spanish"), ("Bengali-English", "ben", "Bengali")):
            n, avg, ratio = TARGETS[pair]
            add(pair, synth_pair(rng, mono["eng"][split], "English", mono[lang][split], lab,
                                 split_counts(n)[split], avg, ratio), "synthetic")
        # English-Kokborok: all real docx sentences (silver labels)
        add("English-Kokborok", kok_pool[split], "real")
        # Kokborok-Bengali: synthetic from real Kokborok sentences of the same split
        add("Kokborok-Bengali", kok_bengali(rng, kok_pool[split], mono["ben"][split],
                                            split_counts(TARGETS["Kokborok-Bengali"][0])[split]), "synthetic")
        rng.shuffle(data[split])

    # ------------------------------------------------------------ checks + stats
    for split in SPLITS:
        for r in data[split]:
            assert len(r["tokens"]) == len(r["labels"]) and r["tokens"], r
            assert all(l in LABELS for l in r["labels"]), r
    tr = {" ".join(r["tokens"]) for r in data["train"] if r["source"] == "real"}
    te = {" ".join(r["tokens"]) for r in data["test"] if r["source"] == "real"}
    print(f"  real-sentence overlap train/test: {len(tr & te)}")

    stats = {"splits": {s: len(data[s]) for s in SPLITS}, "pairs": {}}
    allrec = [r for s in SPLITS for r in data[s]]
    for pair in TARGETS:
        recs = [r for r in allrec if r["pair"] == pair]
        lab = Counter(l for r in recs for l in r["labels"])
        n_tok = sum(lab.values())
        stats["pairs"][pair] = {
            "samples": len(recs),
            "real": sum(r["source"] == "real" for r in recs),
            "synthetic": sum(r["source"] == "synthetic" for r in recs),
            "avg_words": round(n_tok / len(recs), 2),
            # mean per-sentence share of the minority language
            "mixing_ratio": round(sum(1 - max(Counter(r["labels"]).values()) / len(r["labels"])
                                      for r in recs) / len(recs), 3),
            "label_counts": dict(lab),
            "paper_target_samples": TARGETS[pair][0],
        }
    stats["total_samples"] = len(allrec)
    stats["total_tokens"] = sum(len(r["tokens"]) for r in allrec)
    stats["test_tokens"] = sum(len(r["tokens"]) for r in data["test"])

    for split in SPLITS:
        with open(OUT / f"{split}.jsonl", "w", encoding="utf-8") as f:
            for r in data[split]:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    (OUT / "stats.json").write_text(json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(stats, indent=2, ensure_ascii=False))

    for pair in TARGETS:
        print(f"\n== {pair}")
        for r in [r for r in data["train"] if r["pair"] == pair][:4]:
            print("  " + " ".join(f"{t}/{l[:2]}" for t, l in zip(r["tokens"], r["labels"])))


if __name__ == "__main__":
    main()
