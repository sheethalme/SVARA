"""Test BOTH speech-to-text engines on real recordings in Malayalam, Kannada, Hindi and English.

    # 1) get real clips (once; see scripts/fetch_fleurs_samples.py) - or use your own recordings
    python scripts/fetch_fleurs_samples.py --out stt_testdata
    # 2) run
    python scripts/test_stt_languages.py --data stt_testdata                       # local engine only
    set GROQ_API_KEY=...   (Windows)   /   export GROQ_API_KEY=...  (Linux/Mac)
    python scripts/test_stt_languages.py --data stt_testdata --engines local cloud # both engines

Per clip it reports: detected language (correct?), language confidence, WER and CER against the
reference transcript, and latency.  Two modes are run for each engine:
    auto   : the engine must identify the language itself      (tests language ID + recognition)
    forced : the correct language is given as a hint           (tests recognition alone)
Manifest format (CSV): lang,file,gender,seconds,reference    (file is relative to --data)

WER/CER here are computed on text lower-cased with punctuation removed (like the FLEURS
references).  Numbers written as digits vs words and spelling variants of Indic words count as
errors, so Indic WER is pessimistic; compare engines with each other rather than to English.
"""
import argparse, csv, os, sys, time, unicodedata
from collections import defaultdict
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)
import stt


def norm(t):
    t = unicodedata.normalize("NFC", t.lower())
    t = "".join(" " if unicodedata.category(c)[0] in "PS" else c for c in t)
    return " ".join(t.split())


def edit_distance(a, b):
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y)))
        prev = cur
    return prev[-1]


def wer(ref, hyp):
    r, h = norm(ref).split(), norm(hyp).split()
    return edit_distance(r, h) / max(1, len(r))


def cer(ref, hyp):
    r, h = norm(ref), norm(hyp)
    return edit_distance(r, h) / max(1, len(r))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="stt_testdata")
    ap.add_argument("--engines", nargs="+", default=["local"], choices=["local", "cloud"])
    ap.add_argument("--n", type=int, default=0, help="max clips per language (0 = all)")
    ap.add_argument("--langs", nargs="+", default=None, choices=["ml", "kn", "hi", "en"], help="only these languages")
    ap.add_argument("--modes", nargs="+", default=["auto", "forced"], choices=["auto", "forced"])
    ap.add_argument("--local-model", default=None,
                    help="e.g. small (default), medium, large-v3, or a Hugging Face CTranslate2 repo / folder")
    ap.add_argument("--preprocess", action="store_true", help="run the 1B cleaning chain first")
    ap.add_argument("--out", default="stt_results.csv")
    a = ap.parse_args()

    rows = list(csv.DictReader(open(os.path.join(a.data, "manifest.csv"), encoding="utf-8")))
    if a.langs:
        rows = [r for r in rows if r["lang"] in a.langs]
    if a.n:
        seen = defaultdict(int); keep = []
        for r in rows:
            if seen[r["lang"]] < a.n: keep.append(r); seen[r["lang"]] += 1
        rows = keep
    cfg = stt.STTConfig.from_env(**({"local_model": a.local_model} if a.local_model else {}))
    s = stt.SpeechToText(cfg)
    if "cloud" in a.engines and not s.cloud.available:
        print("!! cloud engine NOT configured (set GROQ_API_KEY or OPENAI_API_KEY): cloud will be skipped")
        a.engines = [e for e in a.engines if e != "cloud"]
    if "local" in a.engines:
        print(f"loading local model '{cfg.local_model}' ..."); s.warm_up()

    out, t_all = [], time.time()
    for eng in a.engines:
        for mode in a.modes:
            for k, r in enumerate(rows, 1):
                hint = r["lang"] if mode == "forced" else None
                try:
                    res = s.transcribe_full(os.path.join(a.data, r["file"]), language=hint, engine=eng,
                                            preprocess=a.preprocess)
                    text, lang, conf, lat = res.text, res.language, res.language_confidence, res.latency_s
                    err = ""
                except Exception as e:                       # keep going; record the failure
                    text, lang, conf, lat, err = "", "error", None, 0.0, f"{type(e).__name__}: {e}"[:120]
                out.append(dict(engine=eng, mode=mode, lang=r["lang"], file=r["file"], gender=r["gender"],
                                detected=lang, lang_ok=int(lang == r["lang"]), confidence=conf,
                                wer=round(wer(r["reference"], text), 3), cer=round(cer(r["reference"], text), 3),
                                latency_s=round(lat, 2), seconds=r["seconds"], hypothesis=text,
                                reference=r["reference"], error=err))
                print(f"[{eng}/{mode}] {k}/{len(rows)} {r['lang']} -> {lang} WER {out[-1]['wer']:.2f}", flush=True)
    with open(a.out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(out[0])); w.writeheader(); w.writerows(out)

    print("\n" + "=" * 86)
    print(f"{'engine':7} {'mode':7} {'lang':5} {'clips':>5} {'lang-ID acc':>12} {'mean WER':>9} {'mean CER':>9} {'latency(s)':>11}")
    for eng in a.engines:
        for mode in a.modes:
            for lang in ("ml", "kn", "hi", "en"):
                g = [o for o in out if o["engine"] == eng and o["mode"] == mode and o["lang"] == lang]
                if not g: continue
                n = len(g)
                print(f"{eng:7} {mode:7} {lang:5} {n:5d} {sum(o['lang_ok'] for o in g) / n:12.0%} "
                      f"{sum(o['wer'] for o in g) / n:9.2f} {sum(o['cer'] for o in g) / n:9.2f} "
                      f"{sum(o['latency_s'] for o in g) / n:11.1f}")
    print(f"\nresults saved to {a.out}   (total time {time.time() - t_all:.0f} s)")


if __name__ == "__main__":
    main()
