"""Accuracy report for 2B on the labelled cases (tests/data/extractor_cases.csv) - rules only, no network.

    python scripts/eval_extractor.py                       # table per field and per language
    python scripts/eval_extractor.py --cases my_cases.csv  # your own cases: field,language,text,expected
    python scripts/eval_extractor.py --show-errors

IMPORTANT: the default cases were written by hand by the author of the module (they are NOT recordings), so
this measures "does the rule engine handle the forms of speech its author thought of", not real-world accuracy.
Add rows with real recogniser output (what stt.transcribe returned for your own recordings) to measure that.
"""
import argparse, collections, csv, datetime as dt, os, sys
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)
import extractor as ex

NOISE = {"clean": lambda t: t, "leading filler + full stop": lambda t: "Um, " + t + ".",
         "capital + full stop": lambda t: t[:1].upper() + t[1:] + ".", "spaces + ellipsis": lambda t: "  " + t.replace(" ", "  ") + " ...",
         "trailing thank you (Latin)": lambda t: t + (" thank you" if t.isascii() else ""), "quotes": lambda t: '"' + t + '"'}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cases", default=os.path.join(ROOT, "tests", "data", "extractor_cases.csv"))
    ap.add_argument("--today", default="2026-11-13", help="'today' used for date-of-birth checks (YYYY-MM-DD)")
    ap.add_argument("--show-errors", action="store_true")
    ap.add_argument("--out", default=None, help="write the per-field summary to this CSV")
    a = ap.parse_args()
    e = ex.Extractor(today=dt.date.fromisoformat(a.today))
    rows = list(csv.DictReader(open(a.cases, encoding="utf-8", newline="")))
    by_f, by_l, errors = collections.defaultdict(lambda: [0, 0]), collections.defaultdict(lambda: [0, 0]), []
    for r in rows:
        got = e.extract(r["text"], r["field"], language=r["language"]).value
        ok = (got or "") == r["expected"]
        for d, k in ((by_f, r["field"]), (by_l, r["language"])):
            d[k][0] += ok; d[k][1] += 1
        if not ok:
            errors.append((r["field"], r["language"], r["text"], r["expected"], got))
    tot = sum(v[1] for v in by_f.values()); good = sum(v[0] for v in by_f.values())
    print(f"{good}/{tot} correct ({good / tot:.1%})   [{len(rows)} cases, rules only]\n")
    print(f"{'field':15}{'cases':>6}{'correct':>9}{'accuracy':>10}")
    for f, (g, n) in sorted(by_f.items()):
        print(f"{f:15}{n:6d}{g:9d}{g / n:10.0%}")
    print(f"\n{'language':15}{'cases':>6}{'correct':>9}{'accuracy':>10}")
    for l, (g, n) in sorted(by_l.items()):
        print(f"{l:15}{n:6d}{g:9d}{g / n:10.0%}")
    pos = [r for r in rows if r["expected"]]
    print(f"\nrobustness to recogniser-style noise ({len(pos)} cases with an expected value):")
    for name, fn in NOISE.items():
        g = sum((e.extract(fn(r["text"]), r["field"]).value or "") == r["expected"] for r in pos)
        print(f"  {name:28}{g:4d}/{len(pos)}  {g / len(pos):.1%}")
    if a.show_errors or errors:
        print(f"\n{len(errors)} error(s):")
        for x in errors[:50]:
            print("  ", x)
    if a.out:
        with open(a.out, "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh); w.writerow(["field", "cases", "correct", "accuracy"])
            for f, (g, n) in sorted(by_f.items()):
                w.writerow([f, n, g, f"{g / n:.3f}"])


if __name__ == "__main__":
    main()
