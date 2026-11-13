"""Chart for docs: error rates per language and model (reads the docs/results_2a_*.csv files)."""
import csv, os
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
DOCS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "docs")
def load(name, mode):
    rows = [r for r in csv.DictReader(open(os.path.join(DOCS, name), encoding="utf-8")) if r["mode"] == mode]
    out = {}
    for lang in ("en", "hi", "kn", "ml"):
        g = [r for r in rows if r["lang"] == lang]
        if g: out[lang] = (np.mean([float(r["wer"]) for r in g]), np.mean([float(r["cer"]) for r in g]), len(g))
    return out
small = load("results_2a_whisper_small.csv", "auto"); large = load("results_2a_whisper_large_v3.csv", "auto")
spec = load("results_2a_malayalam_specialist.csv", "forced")
series = [("Whisper small (local fallback)", small, "#777777"), ("Whisper large-v3 (= cloud model)", large, "#2a6fbb"),
          ("Malayalam fine-tuned large-v3", spec, "#2e8b57")]
langs = ["en", "hi", "kn", "ml"]; names = {"en": "English", "hi": "Hindi", "kn": "Kannada", "ml": "Malayalam"}
plt.rcParams.update({"figure.dpi": 130, "font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
fig, ax = plt.subplots(1, 2, figsize=(10, 3.6), sharey=False)
for j, (idx, title) in enumerate([(0, "Word error rate (lower is better)"), (1, "Character error rate (lower is better)")]):
    w = 0.26
    for i, (lab, d, col) in enumerate(series):
        xs, ys = [], []
        for k, l in enumerate(langs):
            present = [sd for _, sd, _ in series if l in sd]            # centre the bars that exist
            if l in d:
                xs.append(k + (present.index(d) - (len(present) - 1) / 2) * w); ys.append(d[l][idx])
        ax[j].bar(xs, [min(y, 2.0) for y in ys], w, color=col, label=lab)
        for x, y in zip(xs, ys): ax[j].text(x, min(y, 2.0) + .03, f"{y:.2f}", ha="center", fontsize=7)
    ax[j].set_xticks(range(4)); ax[j].set_xticklabels([f"{names[l]}\nn={small[l][2]} / {large[l][2]}" + (" / 8" if l == "ml" else "") for l in langs])
    ax[j].set_title(title); ax[j].axhline(1.0, color="#c0392b", lw=.7, ls="--"); ax[j].set_ylim(0, 2.2)
ax[0].legend(fontsize=7, loc="upper left"); ax[0].text(-0.45, 1.04, "error rate 1.0 = unusable", color="#c0392b", fontsize=7, ha="left")
fig.suptitle("Real speech (FLEURS read sentences). n = clips per model: small / large-v3 / Malayalam specialist", fontsize=9, y=1.0)
fig.tight_layout(); fig.savefig(os.path.join(DOCS, "figures", "fig9_stt_results.png"), bbox_inches="tight")
print({k: {l: tuple(round(x, 2) for x in v[:2]) for l, v in d.items()} for k, d, _ in series})
