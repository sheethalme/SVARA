"""
python -m formreader analyze form.pdf [--out folder]     read a blank form: list the fields, write template.json + preview PNGs
python -m formreader fill form.pdf answers.json out.pdf  fill it from {call_name: value} (uses a fresh analysis)
Nothing is sent anywhere: the whole reading is local.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

from formreader.analyze import analyze_pdf
from formreader.fill import fill_pdf
from formreader.render import render_overlay


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="formreader", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("analyze"); a.add_argument("pdf"); a.add_argument("--out", default="form_analysis")
    f = sub.add_parser("fill"); f.add_argument("pdf"); f.add_argument("answers"); f.add_argument("out")
    args = ap.parse_args(argv)
    raw = open(args.pdf, "rb").read()
    t = analyze_pdf(raw, filename=os.path.basename(args.pdf))
    if args.cmd == "analyze":
        os.makedirs(args.out, exist_ok=True)
        with open(os.path.join(args.out, "template.json"), "w", encoding="utf-8") as fh:
            fh.write(t.to_json())
        for p in range(len(t.pages)):
            with open(os.path.join(args.out, f"page{p + 1}_fields.png"), "wb") as fh:
                fh.write(render_overlay(raw, t, p))
        print(f"{t.title}  ({'fillable' if t.acroform else 'flat'} PDF, {len(t.pages)} page(s))")
        for i, fl in enumerate(t.fields, 1):
            opts = f" options={[o.label for o in fl.options]}" if fl.options else ""
            print(f"{i:>3}. p{fl.page + 1} {fl.shape:<9} {'IGNORED ' if fl.ignored else ''}{fl.label!r:<34} -> {fl.call_name or '-':<26} "
                  f"conf {fl.confidence:.2f}{opts}")
        for w in t.warnings:
            print("WARNING:", w)
        print(f"Wrote {args.out}/template.json and page previews.")
        return 0
    values = json.load(open(args.answers, encoding="utf-8"))
    pdf, rep = fill_pdf(t, raw, values)
    open(args.out, "wb").write(pdf)
    print(json.dumps(rep, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
