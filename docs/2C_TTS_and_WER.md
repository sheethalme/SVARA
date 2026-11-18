---
title: "SVARA — Sub-module 2C: Text-to-Speech and WER Metrics"
subtitle: "Member 2 (Speech Recognition and AI) · implementation, design and test document"
---

# 1. What 2C is and what it promises

2C gives SVARA its **voice** and its **yardstick**:

* **Text-to-speech (`tts.py`)** asks each question aloud and reads the extracted answer back ("I heard your date of birth as 15 മാർച്ച് 2000. Is that correct?") in the user's language.
* **Word error rate (`wer.py`)** is the project's single, official WER function. Member 1's noise experiment (1C, `evaluate.py`) uses it to score the recogniser.

> **`audio = speak(text, language)`** → MP3 bytes (play with `st.audio`, save, or decode with 1A).
> **`wer = compute_wer(reference, hypothesis)`** → a fraction, e.g. `0.25` = 25 %.

| Call | Returns |
|---|---|
| `speak(text, language="en", slow=False, mixed_script=True)` | MP3 bytes |
| `speak_full(...)` | `SpeechResult`: `audio`, `segments` (each piece's language, text, cache hit, file), `cached`, `latency_s`, `to_signal()` (→ 16 kHz float32 via 1A), `save(path)` |
| `prefetch(texts, language)` | makes the audio ahead of time (the admin page does this when a form is saved) |
| `split_by_script(text, language)` | the pieces and their voices, e.g. `[("en", "I heard ... as 15"), ("ml", "മാർച്ച് 2000."), ...]` |
| `cache_info()`, `clear_cache()` | cache folder, number of files, bytes |
| `compute_wer(reference, hypothesis, normalize=True, language=None)` | WER as a float |
| `compute_wer_full(...)` | `WERResult`: `wer`, `substitutions`, `deletions`, `insertions`, `hits`, `n_ref`, `n_hyp`, `alignment`, `pretty()` |
| `compute_cer(...)` | the same on characters (character error rate) |
| `corpus_wer(pairs or results)` | total errors / total reference words over a test set |
| `edit_distance_matrix(ref_words, hyp_words)` | the dynamic-programming table (for reports and the viva) |

| Item | Value |
|---|---|
| Files | `tts.py`, `wer.py`; tests `tests/test_tts.py`, `tests/test_wer.py` |
| Needs | `gTTS` (new, ~100 KB, pure Python) for speech; WER needs only the Python standard library |
| Works offline | WER: **yes**. TTS: sentences already in the cache **yes**; new sentences need internet (the app then falls back to the browser's voice) |
| Languages | English (Indian accent), Hindi, Malayalam, Kannada |
| Tests | 49 new tests for 2C (867 in the whole project), all passing without internet; 4 extra live gTTS tests with `SVARA_ONLINE_TESTS=1`, also passing |

# 2. Text-to-speech

## 2.1 Why gTTS

| Option | Malayalam / Kannada | Cost / key | Offline | Chosen |
|---|---|---|---|---|
| **gTTS** (Google Translate voice) | yes, natural | free, no key | no (cache helps) | **yes** |
| Browser `speechSynthesis` (was used by the voice page) | often **no voice installed** on Windows / Android | free | yes | kept as the fallback |
| pyttsx3 / Windows SAPI | no ml / kn voices by default | free | yes | no |
| Cloud TTS (Azure, Google Cloud) | yes | needs a paid key | no | no |

gTTS is the only free option that speaks all four project languages. Its weak point (needs internet) is handled by the cache and the browser fallback.

## 2.2 How it works

1. **Clean the text**: Unicode NFC, collapse spaces. Zero-width joiners are kept (Malayalam chillu letters need them).
2. **Split by script** (`split_by_script`). The questions and read-backs mix English with the user's language. Each character's *script* decides the voice: Latin → English, Malayalam block (U+0D00–0D7F) → `ml`, Kannada (U+0C80–0CFF) → `kn`, Devanagari (U+0900–097F) → `hi`. Digits, spaces and punctuation stay with the piece before them. Text with no letters at all ("9 8 4 7 2") is read in the requested language.
3. **For each piece**: look in the cache; if it is not there, call gTTS (`tld="co.in"`, so English has an Indian accent), then save it.
4. **Join the MP3s.** MP3 is a stream of independent frames, so the pieces can be joined byte by byte and play as one clip.

## 2.3 Audio caching

* **Key** = SHA-256 of `engine | language | accent | speed | text`. The same sentence in the same voice always maps to the same file (`tts_cache/ab/ab12….mp3`); a different language or the slow voice is a different file.
* **Per piece, not per sentence**: "Is that correct?" is cached once and reused after every read-back.
* **Safe writes**: written to a temporary `.part` file, then renamed, so a crash never leaves half a file. A cached file that is empty or not MP3 is ignored and made again.
* **Prefetch**: when the admin saves a form, all its questions are synthesised, so users hear them instantly and they still work if the internet drops later.
* Location: `tts_cache/` (in `.gitignore`), or set `SVARA_TTS_CACHE`.

**Measured on this machine (live gTTS, 16 Nov 2026):**

| Sentence | Language | First time (gTTS) | Again (cache) |
|---|---|---|---|
| "What is your age?" | en | 0.66 s | 0.01 s |
| "आपकी उम्र क्या है?" | hi | 0.29 s | 0.02 s |
| "നിങ്ങളുടെ പേര് പറയൂ" | ml | 0.32 s | 0.02 s |
| "ನಿಮ್ಮ ಹೆಸರು ಹೇಳಿ" | kn | 0.23 s | 0.01 s |
| "I heard your date of birth as 15 മാർച്ച് 2000. Is that correct?" (3 pieces: en + ml + en) | ml | 0.79 s | 0.07 s |

The joined English + Malayalam MP3 decodes through 1A (`to_signal()`) to a 5.9 s, 16 kHz signal.

## 2.4 Where it is used

* **Fill Form by Voice** page: every question and read-back is played with `st.audio(..., autoplay=True)`. If gTTS fails and the sentence is not cached, the page says so and uses the browser's voice instead.
* **Admin Form Setup** page: prefetches the questions on save.
* **Test page** (`scripts/try_it_app.py`): a 2C section speaks the 2B read-back (or any sentence) and shows which voice read which piece and whether the cache was used.

## 2.5 Limits

* gTTS is an unofficial client of Google Translate's voice. It can change or rate-limit without notice; the cache and the browser fallback keep the app usable.
* **Privacy**: the *text* (not the voice) is sent to Google, and read-backs can contain a phone or Aadhaar number. For strict deployments, read those fields with the browser voice (on the device) only.
* The question *wording* comes from the form (3A/admin) and is in English by default; 2C reads whatever text it is given in the correct voice, but does not translate.

# 3. Word error rate

## 3.1 Definition (viva)

The recogniser's output (**hypothesis**) is aligned with what was really said (**reference**) using the fewest edits. Each difference is one of:

* **Substitution (S)**: a word was recognised as a different word.
* **Deletion (D)**: a reference word is missing.
* **Insertion (I)**: an extra word appears that was never said.

$$\text{WER} = \frac{S + D + I}{N}, \qquad N = \text{number of words in the reference}$$

Word accuracy = 1 − WER. Because insertions are not bounded by N, **WER can be above 100 %** ("yes" → "yes yes yes" is 2 / 1 = 200 %). WER is **not symmetric**: swapping reference and hypothesis changes N.

## 3.2 How the minimum edits are found: Levenshtein distance by dynamic programming

`d[i][j]` = the fewest edits that turn the first `i` reference words into the first `j` hypothesis words.

```
d[i][0] = i                      (delete all i words)
d[0][j] = j                      (insert all j words)
d[i][j] = min( d[i-1][j-1] + (0 if ref[i] == hyp[j] else 1),   match / substitution  (diagonal)
               d[i-1][j]   + 1,                                deletion              (from above)
               d[i][j-1]   + 1 )                               insertion             (from the left)
```

`d[N][M]` is S + D + I. Walking back from `d[N][M]` to `d[0][0]` splits it: a diagonal step is a match or substitution, a step up is a deletion, a step left is an insertion. The time and memory are O(N × M).

## 3.3 Worked example

Reference "my name is athira mohan" (N = 5), hypothesis "my name athira mohanan kumar" (M = 5):

```
              #      my    name  athira mohanan   kumar
      #       0       1       2       3       4       5
     my       1       0       1       2       3       4
   name       2       1       0       1       2       3
     is       3       2       1       1       2       3
 athira       4       3       2       1       2       3
  mohan       5       4       3       2       2       3
```

```
REF: my name is athira mohan   *****
HYP: my name ** athira mohanan kumar
             D         S       I
S=1 D=1 I=1 N=5  WER = (1+1+1)/5 = 60.00%
```

(`python wer.py "my name is athira mohan" "my name athira mohanan kumar" --raw` prints exactly this.)

**Several alignments can have the same minimum.** Here "is athira mohan" could also be read as three substitutions. All minimal paths give the same total, so the WER is always the same; `wer.py` shows the one with the **most correct words** (a second table keeps, for each cell, the most matches among the minimal paths), which is the alignment a person would draw.

## 3.4 Normalisation before scoring

Formatting is not a recognition error, so both texts are normalised first (the rules 1C already used, now in `wer.normalize_text`): lower case, punctuation and symbols removed (Indic vowel signs and viramas kept), and for English Whisper's normaliser, so "twenty five" = "25" and "98472 16530" = "9847216530". `normalize=False` scores the raw strings.

## 3.5 CER and corpus WER

* **CER** (`compute_cer`): the same algorithm on characters (spaces not counted). Useful for Malayalam and Kannada, where one long word can hold a whole phrase and a one-letter slip would otherwise cost a full word: "athira" → "adhira" is WER 100 % but CER 1/6 = 16.7 %.
* **Corpus WER** (`corpus_wer`): total errors / total reference words over all clips, so long answers weigh more than short ones. This is how 1C reports each SNR condition.

## 3.6 One implementation for the project

`evaluate.py` (1C) now calls `wer.compute_wer_full` instead of its own copy, so the experiment, the test page and any report use the same numbers. Its existing tests (35) still pass unchanged.

# 4. Tests

| File | Tests | What they check |
|---|---|---|
| `tests/test_wer.py` | 22 | each error type, the formula and N, WER > 1, empty inputs, asymmetry, DP borders, alignment rebuilds both sentences, 300 random cases against an independent Levenshtein, 100 random cases against `jiwer`, Hindi/Malayalam/Kannada, CER, corpus WER, 1C uses this module |
| `tests/test_tts.py` | 27 (+4 live) | MP3 returned, second call from cache, cache works offline in a new process, key depends on language and speed, broken cache file replaced, cache off, clear, prefetch, language names, script splitting (mixed, single script, digits only, Malayalam chillu + ZWJ), one voice per piece and joined audio, empty text, gTTS errors become `TTSError`; live: one real request per language decoded by 1A |

Run: `python -m pytest -q tests/test_wer.py tests/test_tts.py` (offline), and `SVARA_ONLINE_TESTS=1 python -m pytest -q tests/test_tts.py -k real` (needs internet).

# 5. Command line

```
python tts.py "How old are you?" --lang en -o q.mp3
python tts.py "നിങ്ങളുടെ പേര് പറയൂ" --lang ml -o name_ml.mp3
python tts.py --cache-info
python wer.py "my name is athira" "my name athira" --cer
```
