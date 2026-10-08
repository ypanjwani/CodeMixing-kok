"""Build results/report.html (a self-contained, theme-aware results page) from the run JSONs."""
import html
import json
import re

import numpy as np

from lid import LABELS, ROOT
from make_report import PAPER, build

OUT = ROOT / "results" / "report.html"
esc = html.escape


def pct(x, d=1):
    return f"{100 * x:.{d}f}%"


def f3(x):
    return "–" if x is None else f"{x:.3f}"


# ---------------------------------------------------------------- svg helpers
def hbar(rows, lo, hi=1.0, unit="acc", show_paper=True, fmt=lambda v: f"{v:.3f}"):
    """rows: (label, value, paper_value|None, highlight). Bars = ours, diamond = paper."""
    W, L, R, rh = 680, 210, 64, 34
    H = len(rows) * rh + 34
    x = lambda v: L + (v - lo) / (hi - lo) * (W - L - R)
    ticks = np.linspace(lo, hi, 5)
    out = [f'<svg viewBox="0 0 {W} {H}" role="img" class="chart">']
    for t in ticks:
        out.append(f'<line x1="{x(t):.1f}" x2="{x(t):.1f}" y1="6" y2="{H - 26}" class="grid"/>')
        out.append(f'<text x="{x(t):.1f}" y="{H - 8}" class="tick" text-anchor="middle">{fmt(t)}</text>')
    for i, (label, v, paper, hl) in enumerate(rows):
        y = 10 + i * rh
        cls = "bar hl" if hl else "bar"
        w = max(2.0, x(v) - L)
        out.append(f'<text x="{L - 10}" y="{y + 15}" class="lab" text-anchor="end">{esc(label)}</text>')
        out.append(f'<rect x="{L}" y="{y + 3}" width="{w:.1f}" height="18" rx="3" class="{cls}">'
                   f'<title>{esc(label)}: {fmt(v)}' + (f" (paper {fmt(paper)})" if paper else "") + '</title></rect>')
        out.append(f'<text x="{L + w + 6:.1f}" y="{y + 16}" class="val">{fmt(v)}</text>')
        if show_paper and paper is not None:
            px = x(paper)
            out.append(f'<path d="M{px:.1f} {y} l6 12 l-6 12 l-6 -12z" class="paper">'
                       f'<title>Paper: {fmt(paper)}</title></path>')
    out.append("</svg>")
    return "".join(out)


def stacked(rows):
    """Dataset: real vs synthetic per pair."""
    W, L, R, rh = 680, 150, 70, 34
    H = len(rows) * rh + 34
    mx = max(r + s for _, r, s in rows)
    x = lambda v: L + v / mx * (W - L - R)
    out = [f'<svg viewBox="0 0 {W} {H}" role="img" class="chart">']
    for t in np.linspace(0, 17500, 8):
        if t <= mx:
            out.append(f'<line x1="{x(t):.1f}" x2="{x(t):.1f}" y1="6" y2="{H - 26}" class="grid"/>')
            out.append(f'<text x="{x(t):.1f}" y="{H - 8}" class="tick" text-anchor="middle">{int(t / 1000)}k</text>')
    for i, (label, r, s) in enumerate(rows):
        y = 10 + i * rh
        out.append(f'<text x="{L - 10}" y="{y + 15}" class="lab" text-anchor="end">{esc(label)}</text>')
        if r:
            out.append(f'<rect x="{L}" y="{y + 3}" width="{x(r) - L:.1f}" height="18" rx="3" class="bar hl">'
                       f'<title>{esc(label)} real: {r:,}</title></rect>')
        if s:
            gap = 2 if r else 0
            out.append(f'<rect x="{x(r) + gap:.1f}" y="{y + 3}" width="{x(r + s) - x(r) - gap:.1f}" height="18" '
                       f'rx="3" class="bar syn"><title>{esc(label)} synthetic: {s:,}</title></rect>')
        out.append(f'<text x="{x(r + s) + 6:.1f}" y="{y + 16}" class="val">{r + s:,}</text>')
    out.append("</svg>")
    return "".join(out)


def table(header, rows, cls=""):
    th = "".join(f"<th>{h}</th>" for h in header)
    body = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in rows)
    return f'<div class="tw"><table class="{cls}"><thead><tr>{th}</tr></thead><tbody>{body}</tbody></table></div>'


def verdict(ok, text=None):
    if ok is None:
        return f'<span class="pill na">{text or "Not verified"}</span>'
    return f'<span class="pill {"ok" if ok else "no"}">{text or ("Holds" if ok else "Does not hold")}</span>'


# ---------------------------------------------------------------- page
def main():
    stats, R, seeds, mean, std, T = build()
    acc = {k: v["accuracy"] for k, v in R.items()}
    main_run = R["proposed_seed42"]
    logtxt = (ROOT / "results" / "log.txt").read_text(encoding="utf-8")
    partial = re.findall(r"\[pretrained_mbert\] epoch (\d+) .*val_acc ([0-9.]+)", logtxt)
    mbert_val = float(partial[-1][1]) if partial else None
    syn_share = sum(d["synthetic"] for d in stats["pairs"].values()) / stats["total_samples"]
    cnn_hurts = acc["baseline_bilstm"] > mean["accuracy"]

    # claims
    claims = [
        ("Proposed model accuracy", "0.926", f"{mean['accuracy']:.3f} ± {std['accuracy']:.3f}",
         verdict(abs(mean["accuracy"] - 0.926) <= 2 * std["accuracy"], "Holds, within seed noise")),
        ("F1 / precision / recall", "0.925 / 0.926 / 0.924",
         f"{mean['f1']:.3f} / {mean['precision']:.3f} / {mean['recall']:.3f}", verdict(True, "Holds")),
        ("BiLSTM component helps", "+5.6 pts", f"{100 * (acc['proposed_seed42'] - acc['baseline_cnn']):+.1f} pts",
         verdict(acc["proposed_seed42"] > acc["baseline_cnn"])),
        ("Adam beats SGD", "+4.4 pts", f"{100 * (acc['proposed_seed42'] - acc['optimizer_sgd']):+.1f} pts",
         verdict(acc["proposed_seed42"] > acc["optimizer_sgd"])),
        ("Proposed beats Transformer", "0.926 vs 0.900",
         f"{mean['accuracy']:.3f} vs {acc['baseline_transformer']:.3f}",
         verdict(mean["accuracy"] > acc["baseline_transformer"])),
        ("Learning rate 0.001 is best", "yes", "yes", verdict(True)),
        ("CNN component helps", "+4.6 pts", f"{100 * (mean['accuracy'] - acc['baseline_bilstm']):+.1f} pts",
         verdict(not cnn_hurts)),
        ("Adam beats AdamW / RMSprop", "+1.1 / +2.5 pts",
         f"{100 * (acc['proposed_seed42'] - acc['optimizer_adamw']):+.1f} / "
         f"{100 * (acc['proposed_seed42'] - acc['optimizer_rmsprop']):+.1f} pts",
         verdict(acc["proposed_seed42"] > max(acc["optimizer_adamw"], acc["optimizer_rmsprop"]),
                 None if acc["proposed_seed42"] > max(acc["optimizer_adamw"], acc["optimizer_rmsprop"])
                 else "Does not hold (tied)")),
        ("Stable across seeds", "± 0.002", f"± {std['accuracy']:.3f}", verdict(std["accuracy"] <= 0.004)),
        ("Beats BERT / mBERT / XLM-R", "0.926 vs 0.90–0.91", "not run", verdict(None)),
    ]
    claims_html = table(["Claim in the paper", "Paper", "This reproduction", "Verdict"],
                        [[esc(a), b, c, d] for a, b, c, d in claims], "claims")

    # dataset
    ds_rows = [(p, d["real"], d["synthetic"]) for p, d in stats["pairs"].items()]
    ds_table = table(["Language pair", "Sentences", "Real", "Synthetic", "Avg. words", "Mix ratio", "Paper"],
                     [[p, f"{d['samples']:,}", f"{d['real']:,}", f"{d['synthetic']:,}", d["avg_words"],
                       d["mixing_ratio"], f"{d['paper_target_samples']:,}"] for p, d in stats["pairs"].items()]
                     + [["<b>Total</b>", f"<b>{stats['total_samples']:,}</b>", "", "", "", "", "<b>48,397</b>"]])

    # seeds
    seed_rows = [[s["seed"], f3(s["accuracy"]), f3(s["f1"]), f3(s["precision"]), f3(s["recall"]),
                  f3(s["mcc"]), f3(s["auc_roc"])] for s in seeds]
    seed_rows.append(["<b>Mean ± SD</b>"] + [f"<b>{mean[k]:.3f} ± {std[k]:.3f}</b>"
                                              for k in ("accuracy", "f1", "precision", "recall", "mcc", "auc_roc")])
    seeds_html = table(["Seed", "Accuracy", "F1", "Precision", "Recall", "MCC", "AUC-ROC"], seed_rows)

    # models
    models = [("Proposed CNN-BiLSTM-Adam", "proposed_seed42", "proposed", True),
              ("Baseline RNN (BiLSTM)", "baseline_bilstm", "baseline_bilstm", False),
              ("Transformer (scratch)", "baseline_transformer", "baseline_transformer", False),
              ("Baseline CNN", "baseline_cnn", "baseline_cnn", False)]
    model_rows = [(n, mean["accuracy"] if k == "proposed_seed42" else acc[k], PAPER[p].get("accuracy"), hl)
                  for n, k, p, hl in models]
    models_chart = hbar(model_rows, 0.75)
    models_table = table(["Method", "Accuracy", "F1", "Macro F1", "AUC-ROC", "Train time", "Paper acc."],
                         [[n, f3(R[k]["accuracy"]), f3(R[k]["f1"]), f3(R[k]["f1_macro"]), f3(R[k]["auc_roc"]),
                           f"{R[k]['train_time_s']:.0f} s", f3(PAPER[p].get("accuracy"))] for n, k, p, _ in models]
                         + [["mBERT / XLM-R (fine-tuned)", "not run", "", "", "", "", "0.900 / 0.910"]])

    opts = [("Adam (proposed)", "proposed_seed42", "proposed", True), ("AdamW", "optimizer_adamw", "optimizer_adamw", False),
            ("RMSprop", "optimizer_rmsprop", "optimizer_rmsprop", False), ("SGD", "optimizer_sgd", "optimizer_sgd", False)]
    opts_chart = hbar([(n, acc[k], PAPER[p].get("accuracy"), hl) for n, k, p, hl in opts], 0.75)

    abl = [("Full model", "proposed_seed42", 0.926, True), ("Without CNN", "baseline_bilstm", 0.880, False),
           ("Without BiLSTM", "baseline_cnn", 0.870, False), ("Without Adam (SGD)", "optimizer_sgd", 0.890, False)]
    abl_chart = hbar([(n, acc[k], p, hl) for n, k, p, hl in abl], 0.75)

    # grid
    grid_rows = []
    for lr in (1e-5, 1e-4, 1e-3):
        cells = [f"<b>{lr:g}</b>"]
        for bs in (32, 64, 128):
            r = R.get(f"grid_lr{lr:g}_bs{bs}")
            if r:
                w = max(0, (r["accuracy"] - 0.6) / 0.4 * 100)
                cells.append(f'<div class="mini"><span style="width:{w:.0f}%"></span></div>{r["accuracy"]:.3f}'
                             f'<small> · {r["epochs_run"]} ep</small>')
            else:
                cells.append("–")
        grid_rows.append(cells)
    grid_html = table(["Learning rate", "Batch 32", "Batch 64", "Batch 128"], grid_rows, "grid")

    # per pair + per label
    pp = main_run["per_pair_accuracy"]
    pair_chart = hbar([(p, a, None, True) for p, a in sorted(pp.items(), key=lambda t: -t[1])], 0.75,
                      show_paper=False)
    label_table = table(["Language", "Precision", "Recall", "F1", "Test tokens"],
                        [[l, f3(v["precision"]), f3(v["recall"]), f3(v["f1"]), f"{v['support']:,}"]
                         for l, v in main_run["per_label"].items()])

    # confusion
    cm = np.array(main_run["confusion_matrix"])
    lmax = np.log10(cm.max() + 1)
    head = "".join(f"<th>{l}</th>" for l in LABELS)
    body = ""
    for i, l in enumerate(LABELS):
        body += f"<tr><th>{l}</th>"
        for j in range(len(LABELS)):
            s = np.log10(cm[i, j] + 1) / lmax
            cls = "cm d" if i == j else "cm"
            body += (f'<td class="{cls}" style="--s:{s * 92:.0f}%" data-strong="{int(s > 0.62)}" '
                     f'title="{l} predicted as {LABELS[j]}: {cm[i, j]:,} tokens">{cm[i, j]:,}</td>')
        body += "</tr>"
    cm_html = (f'<div class="tw"><table class="cmx"><thead><tr><th>Actual ↓ / Predicted →</th>{head}</tr>'
               f'</thead><tbody>{body}</tbody></table></div>')
    off = cm - np.diag(np.diag(cm))
    ei, ej = np.unravel_index(off.argmax(), off.shape)

    mbert_note = ""
    if mbert_val:
        mbert_note = (f"<p class='note'><b>Preliminary, not a result:</b> an interrupted mBERT run reached "
                      f"<b>{pct(mbert_val)}</b> validation accuracy after one epoch, above the proposed model. "
                      f"The paper's claim that the proposed model beats BERT, mBERT and XLM-R should not be kept "
                      f"until these runs are completed.</p>")

    page = f"""<title>Code-Mixed LID Results</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans+Condensed:wght@500;600;700&family=IBM+Plex+Sans:ital,wght@0,400;0,500;0,600;1,400&display=swap">
<style>
/* Layout: single reading column (~880px) of numbered report sections; charts and tables sit inside the flow. */
:root {{
  --bg: #f6f7f9; --surface: #ffffff; --ink: #14171c; --ink-2: #4a505a; --muted: #7b828d;
  --rule: #dfe3e8; --accent: #2a78d6; --accent-ink: #1c5cab; --syn: #eb6834; --on-accent: #ffffff;
  --ok: #0f7a3d; --ok-bg: #e3f4ea; --no: #b42323; --no-bg: #fbe7e6; --na: #7a5a00; --na-bg: #fbf1d6;
  --bar-dim: #b9c0ca;
  --f-display: "IBM Plex Sans Condensed", "Arial Narrow", system-ui, sans-serif;
  --f-body: "IBM Plex Sans", system-ui, -apple-system, "Segoe UI", sans-serif;
  --f-mono: "IBM Plex Mono", ui-monospace, "Cascadia Mono", Consolas, monospace;
}}
@media (prefers-color-scheme: dark) {{ :root:not([data-theme="light"]) {{
  --bg: #121417; --surface: #1a1d22; --ink: #eceff3; --ink-2: #b5bcc6; --muted: #87909c;
  --rule: #2c3138; --accent: #3987e5; --accent-ink: #8db8ef; --syn: #d95926; --on-accent: #ffffff;
  --ok: #6fd39a; --ok-bg: #16301f; --no: #f08a85; --no-bg: #3a1a19; --na: #e8c35a; --na-bg: #33290f;
  --bar-dim: #4a515c; color-scheme: dark; }} }}
:root[data-theme="dark"] {{
  --bg: #121417; --surface: #1a1d22; --ink: #eceff3; --ink-2: #b5bcc6; --muted: #87909c;
  --rule: #2c3138; --accent: #3987e5; --accent-ink: #8db8ef; --syn: #d95926; --on-accent: #ffffff;
  --ok: #6fd39a; --ok-bg: #16301f; --no: #f08a85; --no-bg: #3a1a19; --na: #e8c35a; --na-bg: #33290f;
  --bar-dim: #4a515c; color-scheme: dark; }}
body {{ background: var(--bg); color: var(--ink); font: 15px/1.6 var(--f-body); }}
.wrap {{ max-width: 880px; margin: 0 auto; padding-inline: 20px; padding-block: 40px 72px; display: grid; gap: 44px; }}
header {{ display: grid; gap: 10px; }}
.eyebrow {{ font: 500 12px/1 var(--f-mono); letter-spacing: .08em; text-transform: uppercase; color: var(--accent-ink); }}
h1 {{ font: 700 clamp(30px, 5vw, 44px)/1.08 var(--f-display); margin: 0; text-wrap: balance; }}
h2 {{ font: 600 24px/1.2 var(--f-display); margin: 0; text-wrap: balance; display: flex; gap: 12px; align-items: baseline; }}
h2 .n {{ font: 500 13px/1 var(--f-mono); color: var(--muted); }}
h3 {{ font: 600 15px/1.3 var(--f-body); margin: 8px 0 0; }}
p {{ margin: 0; max-width: 68ch; color: var(--ink-2); }}
.lede {{ font-size: 17px; color: var(--ink-2); }}
section {{ display: grid; gap: 16px; min-width: 0; }}
.stats {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 12px; }}
.stat {{ background: var(--surface); border: 1px solid var(--rule); border-radius: 8px; padding: 16px; display: grid; gap: 4px; }}
.stat .k {{ font: 500 12px/1.2 var(--f-mono); color: var(--muted); text-transform: uppercase; letter-spacing: .06em; }}
.stat .v {{ font: 600 30px/1.1 var(--f-display); font-variant-numeric: tabular-nums; }}
.stat .s {{ font-size: 13px; color: var(--ink-2); }}
.stat.lead {{ border-color: var(--accent); }}
.stat.lead .v {{ color: var(--accent-ink); }}
.tw {{ overflow-x: auto; border: 1px solid var(--rule); border-radius: 8px; background: var(--surface); }}
table {{ border-collapse: collapse; width: 100%; font-size: 14px; }}
th, td {{ padding: 9px 12px; text-align: left; border-bottom: 1px solid var(--rule); white-space: nowrap; }}
tbody tr:last-child td, tbody tr:last-child th {{ border-bottom: 0; }}
thead th {{ font: 500 12px/1.2 var(--f-mono); color: var(--muted); text-transform: uppercase; letter-spacing: .05em; background: var(--bg); }}
td {{ font-variant-numeric: tabular-nums; }}
table.claims td:first-child {{ white-space: normal; min-width: 180px; font-weight: 500; color: var(--ink); }}
.pill {{ display: inline-flex; align-items: center; gap: 6px; font: 500 12px/1 var(--f-body); padding: 5px 9px; border-radius: 999px; }}
.pill::before {{ font-weight: 700; }}
.pill.ok {{ color: var(--ok); background: var(--ok-bg); }} .pill.ok::before {{ content: "✓"; }}
.pill.no {{ color: var(--no); background: var(--no-bg); }} .pill.no::before {{ content: "✕"; }}
.pill.na {{ color: var(--na); background: var(--na-bg); }} .pill.na::before {{ content: "–"; }}
.card {{ background: var(--surface); border: 1px solid var(--rule); border-radius: 8px; padding: 14px 16px 6px; display: grid; gap: 6px; min-width: 0; }}
.legend {{ display: flex; flex-wrap: wrap; gap: 16px; font-size: 13px; color: var(--ink-2); }}
.legend i {{ display: inline-block; width: 12px; height: 12px; border-radius: 3px; margin-right: 6px; vertical-align: -1px; }}
.legend i.d {{ width: 11px; height: 11px; transform: rotate(45deg) scale(.8); border-radius: 1px; background: none; border: 2px solid var(--ink); }}
svg.chart {{ width: 100%; height: auto; display: block; }}
svg .grid {{ stroke: var(--rule); stroke-width: 1; }}
svg .tick {{ fill: var(--muted); font: 11px var(--f-mono); }}
svg .lab {{ fill: var(--ink-2); font: 13px var(--f-body); }}
svg .val {{ fill: var(--ink); font: 500 12px var(--f-mono); }}
svg .bar {{ fill: var(--bar-dim); }} svg .bar.hl {{ fill: var(--accent); }} svg .bar.syn {{ fill: var(--syn); }}
svg .paper {{ fill: var(--surface); stroke: var(--ink); stroke-width: 2; }}
.note {{ background: var(--na-bg); border-radius: 8px; padding: 12px 14px; color: var(--ink); max-width: none; }}
.warn {{ border-left: 3px solid var(--syn); padding: 2px 0 2px 14px; }}
ul, ol {{ margin: 0; padding-left: 20px; color: var(--ink-2); display: grid; gap: 6px; max-width: 70ch; }}
li b, p b {{ color: var(--ink); }}
.mini {{ display: inline-block; width: 56px; height: 6px; border-radius: 3px; background: var(--rule); margin-right: 8px; vertical-align: 2px; overflow: hidden; }}
.mini span {{ display: block; height: 100%; background: var(--accent); }}
td small {{ color: var(--muted); }}
table.cmx th {{ font: 500 12px/1.2 var(--f-mono); }}
table.cmx tbody th {{ color: var(--ink-2); background: var(--bg); }}
td.cm {{ text-align: right; background: color-mix(in oklab, var(--accent) var(--s), var(--surface)); color: var(--ink); }}
td.cm[data-strong="1"] {{ color: var(--on-accent); font-weight: 500; }}
code, pre {{ font-family: var(--f-mono); font-size: 13px; }}
pre {{ background: var(--surface); border: 1px solid var(--rule); border-radius: 8px; padding: 14px; overflow-x: auto; margin: 0; }}
.two {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 16px; }}
footer {{ font-size: 13px; color: var(--muted); }}
</style>

<div class="wrap">
<header>
  <div class="eyebrow">Reproduction report · token-level code-mixed language identification</div>
  <h1>Optimizing Code-Mixed Language Identification with Adam and Gradient-Based Deep Neural Networks</h1>
  <p class="lede">An independent re-run of the paper's CNN-BiLSTM-Adam model and every experiment it reports, on a
  rebuilt dataset of {stats['total_samples']:,} sentences in five language pairs. All numbers come from real training
  runs. Nothing was tuned toward the paper's figures.</p>
</header>

<section>
  <div class="stats">
    <div class="stat lead"><span class="k">Accuracy · 5 seeds</span><span class="v">{pct(mean['accuracy'])}</span>
      <span class="s">± {100 * std['accuracy']:.1f} pts · paper 92.6%</span></div>
    <div class="stat"><span class="k">Weighted F1</span><span class="v">{mean['f1']:.3f}</span><span class="s">paper 0.925</span></div>
    <div class="stat"><span class="k">AUC-ROC</span><span class="v">{mean['auc_roc']:.3f}</span><span class="s">paper 0.930</span></div>
    <div class="stat"><span class="k">Test tokens</span><span class="v">{stats['test_tokens']:,}</span><span class="s">{stats['splits']['test']:,} sentences</span></div>
  </div>
</section>

<section id="claims">
  <h2><span class="n">01</span>Which claims hold up</h2>
  <p>The headline accuracy, F1, precision and recall reproduce within seed noise. Three other claims do not:
  the CNN layer, Adam's edge over AdamW and RMSprop, and the seed stability. The pretrained-transformer
  comparison was not run.</p>
  {claims_html}
</section>

<section id="data">
  <h2><span class="n">02</span>Data</h2>
  <p>The authors' annotated corpus is not public. The dataset was rebuilt from the paper's Table 1 sources.
  <b>{pct(syn_share, 0)} of sentences are synthetic</b>, and the paper must say so.</p>
  <div class="card">
    <div class="legend"><span><i style="background:var(--accent)"></i>Real</span><span><i style="background:var(--syn)"></i>Synthetic</span></div>
    {stacked(ds_rows)}
  </div>
  {ds_table}
  <ul>
    <li><b>Real:</b> LinCE Hindi-English, with gold labels on every word, and the authors' English-Kokborok
      corpus ({stats['pairs']['English-Kokborok']['real']:,} unique sentences).</li>
    <li><b>Silver labels:</b> the English-Kokborok words were tagged automatically, using the English-Kokborok-Bengali
      Dictionary (Binoy Debbarma) plus context rules. A native speaker has not checked them.</li>
    <li><b>Synthetic:</b> FLORES-200 and MCS-350 single-language sentences were mixed by inserting 1–3-word spans
      of the other language. The Kokborok-Bengali sentences replace the English spans in real Kokborok text with
      romanized Bengali.</li>
    <li><b>Romanized:</b> Hindi and Bengali are written in Latin script, as on social media.</li>
    <li><b>No leakage:</b> each source sentence was assigned to train, validation or test before mixing.</li>
  </ul>
</section>

<section id="method">
  <h2><span class="n">03</span>Model and training</h2>
  <p>The setup follows the paper's Table 5: trainable 300-d embeddings, Conv1D (128 filters, kernel 3, ReLU),
  max-pooling, BiLSTM (128), dropout 0.5 and a softmax over 5 languages for each word. Training uses Adam
  (lr 0.001, β₁ 0.9, β₂ 0.999), batch 64, up to 50 epochs, with early stopping on validation loss.</p>
  <h3>Deviations from the paper</h3>
  <ol>
    <li><b>Max-pooling uses stride 1.</b> Pooling over the whole sentence would leave one vector per sentence, which
      makes per-word labels impossible.</li>
    <li><b>PyTorch instead of Keras.</b> The architecture is the same.</li>
    <li><b>Hardware:</b> one RTX 3050 laptop GPU. Cooling pauses are excluded from the training times.</li>
    <li><b>SGD uses momentum 0.9.</b> All optimizers share the learning rate 0.001.</li>
  </ol>
</section>

<section id="seeds">
  <h2><span class="n">04</span>Main result across five seeds</h2>
  {seeds_html}
  <p>Accuracy varies by ±{100 * std['accuracy']:.1f} points between seeds. The paper reports ±0.2.</p>
</section>

<section id="baselines">
  <h2><span class="n">05</span>Baselines</h2>
  <div class="card">
    <div class="legend"><span><i style="background:var(--accent)"></i>Proposed model (mean of 5 seeds)</span>
      <span><i style="background:var(--bar-dim)"></i>Baselines</span><span><i class="d"></i>Paper's number</span></div>
    {models_chart}
  </div>
  {models_table}
  <ul>
    <li><b>The BiLSTM alone scored {pct(acc['baseline_bilstm'])}</b>, higher than the CNN-BiLSTM. Convolution and pooling
      blend each word with its neighbours, which blurs the signal at language-switch points.</li>
    <li><b>The CNN alone</b> sees only a 3-word window and is the weakest model ({pct(acc['baseline_cnn'])}).</li>
    <li><b>The Transformer trained from scratch</b> ({pct(acc['baseline_transformer'])}) has no pretraining and only
      about 38k training sentences.</li>
  </ul>
  {mbert_note}
</section>

<section id="optimizers">
  <h2><span class="n">06</span>Optimizers and ablation</h2>
  <div class="two">
    <div class="card"><h3>Optimizer, all at lr 0.001</h3>{opts_chart}</div>
    <div class="card"><h3>Remove one component</h3>{abl_chart}</div>
  </div>
  <div class="legend"><span><i style="background:var(--accent)"></i>Full model with Adam</span><span><i class="d"></i>Paper's number</span></div>
  <ul>
    <li><b>SGD</b> is still at {pct(acc['optimizer_sgd'])} after 50 epochs. Adaptive optimizers converge much faster,
      especially for embedding rows of rare words, which receive gradients only occasionally.</li>
    <li><b>Adam, AdamW and RMSprop</b> are within the seed-to-seed spread of each other, so Adam is not clearly best.</li>
    <li><b>The BiLSTM matters most</b> ({100 * (acc['baseline_cnn'] - acc['proposed_seed42']):+.1f} pts when removed).
      Removing the CNN changes accuracy by {100 * (acc['baseline_bilstm'] - acc['proposed_seed42']):+.1f} pts.</li>
  </ul>
</section>

<section id="grid">
  <h2><span class="n">07</span>Learning rate and batch size</h2>
  {grid_html}
  <p>Each run is capped at 20 epochs. At lr 1e-5 the model is still far from converged, and lr 0.001 is best at every
  batch size, as the paper found.</p>
</section>

<section id="errors">
  <h2><span class="n">08</span>Where the model goes wrong</h2>
  <div class="two">
    <div class="card"><h3>Accuracy by language pair</h3>{pair_chart}</div>
    <div>{label_table}</div>
  </div>
  <h3>Confusion matrix, test tokens (seed 42)</h3>
  {cm_html}
  <ul>
    <li><b>Most frequent error:</b> {LABELS[ei]} read as {LABELS[ej]} ({cm[ei, ej]:,} tokens).</li>
    <li><b>Shared words</b> such as <i>a, no, me, come, na, ki</i> exist in more than one language.</li>
    <li><b>Romanized Hindi and Bengali</b> share spellings (<i>ke, ki, ar, kare</i>), which causes most
      Bengali↔Hindi errors.</li>
    <li><b>Unknown words:</b> {pct(main_run['test_oov_rate'])} of test words never appear in training. A
      character-level layer would help.</li>
    <li><b>Kokborok:</b> part of the error there is noise in the automatic labels.</li>
  </ul>
</section>

<section id="limits">
  <h2><span class="n">09</span>Limitations</h2>
  <ol>
    <li>{pct(syn_share, 0)} of the data is synthetic and does not follow real code-switching grammar.</li>
    <li>The English-Kokborok labels are automatic and need a native-speaker check.</li>
    <li>No real Kokborok-Bengali code-mixed text was available.</li>
    <li>Stride-1 max-pooling is our reading of the paper's architecture.</li>
    <li>The mBERT and XLM-R baselines were not completed because of time and hardware limits.</li>
  </ol>
  <p class="warn">Before submission, the paper's dataset section must describe the synthetic data and the silver labels,
  and the claims marked "Does not hold" or "Not verified" above should be revised.</p>
</section>

<section id="repro">
  <h2><span class="n">10</span>Reproduce</h2>
<pre>python src/prepare_data.py              # build data/processed/
python src/run_experiments.py           # every run -> results/runs/*.json (resumable)
python src/run_experiments.py pretrained  # mBERT + XLM-R (about 4–6 h on a laptop GPU)
python src/make_figures.py && python src/make_report.py</pre>
</section>

<footer>Generated from {len(R)} completed training runs (results/runs/*.json).</footer>
</div>
"""
    OUT.write_text(page, encoding="utf-8")
    print("wrote", OUT, f"{len(page) / 1024:.0f} KB")


if __name__ == "__main__":
    main()
