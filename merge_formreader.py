"""Adds what the call page needs to the SHARED files without overwriting them (safe to run twice).

Run from the project root (the folder that contains audio_io.py):    python merge_formreader.py

  requirements.txt : adds `gTTS` (online voice for the call; optional but recommended) if it is not listed
  .gitignore       : adds data/calls/ (personal data!), data/tts_cache/, formreader_server.log if not listed
Originals are saved as *.before_formreader.bak (delete them afterwards).
"""
import os, re, shutil, sys

REQ = ["pdfplumber>=0.11", "pypdf>=4.0", "reportlab>=4.0", "pypdfium2>=4.0", "Pillow>=10.0"]
IGN = ["data/templates/", "data/filled_forms/", "form_analysis/"]


def norm(n):
    return re.sub(r"[-_.]+", "-", n.strip().lower())


def backup(p):
    if not os.path.exists(p + ".before_formreader.bak"):
        shutil.copy2(p, p + ".before_formreader.bak")


def main():
    if not os.path.exists("audio_io.py"):
        sys.exit("Run this from the project root (the folder that contains audio_io.py).")
    if not os.path.isdir("formreader"):
        sys.exit("The formreader/ folder is missing: unzip the form-reader files into the project root first.")
    p = "requirements.txt"
    if os.path.exists(p):
        lines = open(p, encoding="utf-8").read().splitlines()
        have = {norm(m.group(1)) for l in lines if not l.lstrip().startswith(("#", "-")) for m in [re.match(r"\s*([A-Za-z0-9_.\-]+)", l)] if m}
        add = [r for r in REQ if norm(re.match(r"[A-Za-z0-9_.\-]+", r).group(0)) not in have]
        if add:
            backup(p)
            with open(p, "a", encoding="utf-8", newline="\n") as fh:
                fh.write(("\n" if lines and lines[-1].strip() else "") + "# --- PDF form reader ---\n" + "\n".join(add) + "\n")
        print(f"requirements.txt: {'added ' + ', '.join(add) if add else 'nothing to add'}")
    p = ".gitignore"
    lines = open(p, encoding="utf-8").read().splitlines() if os.path.exists(p) else []
    have = {l.strip() for l in lines}
    add = [g for g in IGN if g not in have and g.rstrip("/") not in have]
    if add:
        if os.path.exists(p):
            backup(p)
        with open(p, "a", encoding="utf-8", newline="\n") as fh:
            fh.write(("\n" if lines and lines[-1].strip() else "") + "# --- PDF form reader ---\n" + "\n".join(add) + "\n")
    print(f".gitignore: {'added ' + ', '.join(add) if add else 'nothing to add'}")
    print("\nNext:  pip install -r requirements.txt   then   python -m pytest -q   then   python -m formreader analyze your_form.pdf")


if __name__ == "__main__":
    main()
