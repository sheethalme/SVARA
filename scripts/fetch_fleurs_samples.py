"""Download a few REAL read-speech clips per language (human speakers) for testing 2A.

Source: Google FLEURS (CC-BY 4.0), read from Hugging Face's Parquet copy.
    python scripts/fetch_fleurs_samples.py                      # 8 clips x ml, kn, hi, en
    python scripts/fetch_fleurs_samples.py --langs hi_in --n 5  # only Hindi

Needs:  pip install pyarrow huggingface_hub
WARNING: Parquet stores each language's test split as ONE block, so the first run downloads
about 300-900 MB per language (Malayalam ~900 MB, Kannada ~730 MB, English ~410 MB,
Hindi ~310 MB).  Run it once, on a good connection; only the clips you ask for are kept.

Output:  <out>/<lang>/<id>.wav   (16 kHz mono, as published)   and   <out>/manifest.csv
         columns: lang, file, gender, seconds, reference (FLEURS normalised transcript)
"""
import argparse, csv, io, os, sys
import pyarrow.parquet as pq
from huggingface_hub import HfFileSystem

CODE = {"ml_in": "ml", "kn_in": "kn", "hi_in": "hi", "en_us": "en"}


def fetch(lang_id, n, out):
    fs = HfFileSystem()
    path = f"datasets/google/fleurs@refs%2Fconvert%2Fparquet/{lang_id}/test/0000.parquet"
    print(f"[{lang_id}] reading {path} (large download, please wait)...", flush=True)
    pf = pq.ParquetFile(fs.open(path, block_size=2 * 2**20), pre_buffer=False)
    batch = next(pf.iter_batches(batch_size=80, columns=["id", "num_samples", "audio", "transcription", "gender"])).to_pylist()
    ok = [r for r in batch if 64000 <= r["num_samples"] <= 240000]          # 4-15 s clips
    by_gender = {"MALE": [r for r in ok if r["gender"] == 0], "FEMALE": [r for r in ok if r["gender"] == 1]}
    picked, i = [], 0
    pools = [by_gender["MALE"], by_gender["FEMALE"]]
    while len(picked) < n and any(pools):                                     # alternate voices
        pool = pools[i % 2]
        if pool:
            picked.append(pool.pop(0))
        i += 1
    os.makedirs(os.path.join(out, CODE[lang_id]), exist_ok=True)
    rows = []
    for r in picked:
        name = f"{r['id']}.wav"
        with open(os.path.join(out, CODE[lang_id], name), "wb") as fh:
            fh.write(r["audio"]["bytes"])
        rows.append([CODE[lang_id], f"{CODE[lang_id]}/{name}", "male" if r["gender"] == 0 else "female",
                     round(r["num_samples"] / 16000, 1), r["transcription"]])
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="stt_testdata")
    ap.add_argument("--langs", nargs="+", default=list(CODE), choices=list(CODE))
    ap.add_argument("--n", type=int, default=8, help="clips per language")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    man = os.path.join(a.out, "manifest.csv")
    old = []
    if os.path.exists(man):
        with open(man, newline="", encoding="utf-8") as fh:
            old = [r for r in csv.reader(fh)][1:]
        old = [r for r in old if r[0] not in {CODE[l] for l in a.langs}]
    rows = list(old)
    for l in a.langs:
        rows += fetch(l, a.n, a.out)
        print(f"[{l}] done", flush=True)
    with open(man, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh); w.writerow(["lang", "file", "gender", "seconds", "reference"]); w.writerows(rows)
    print(f"manifest: {man} ({len(rows)} clips)")


if __name__ == "__main__":
    main()
