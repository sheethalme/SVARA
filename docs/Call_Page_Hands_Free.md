---
title: "SVARA — Hands-free voice call page (phone-call style form filling)"
subtitle: "Design, how it works, how to run it, and what was and was not tested"
---

# 1. What this is, and why

Until now, filling a form by voice still needed hands: press **record**, press **stop**, press **play**, press **Yes** or **No**. For people who cannot click, tap or hold a mouse, that defeats the purpose of the project.

The **call page** works like a **phone call**. After one tap on **Start call**, everything is by voice:

> **SVARA:** "Hello. I will ask you a few questions to fill in the Clinic registration form. Your answers will be saved. Do you agree? Please say yes or no."
> **You:** "Yes."
> **SVARA:** "What is your full name?" **You:** "My name is Anil Kumar."
> **SVARA:** "I heard your name as Anil Kumar. Is that correct?" **You:** "Yes."
> **SVARA:** "How old are you?" **You:** "Twenty five." **SVARA:** "I heard your age as 25. Is that correct?" **You:** "No, it is twenty six."
> **SVARA:** "Okay. So your age is 26. Is that correct?" **You:** "Yes." …
> **SVARA:** "Here is the information I have. name: Anil Kumar. age: 26. … Is everything correct?" **You:** "Yes."
> **SVARA:** "Thank you. Your form has been saved. Goodbye."

* **No buttons during the call.** The page speaks, then listens by itself, notices when you stop talking, understands the answer, reads it back and asks "Is that correct?" **by voice**; you say yes, no, "no, it is 26" or "say again".
* **The questions come from the form the admin created** (the fields and their order, custom questions, required or optional).
* **Every detail is stored** as the call goes on (answers, what was heard, confirmations, timestamps), so a call that drops halfway still keeps its answers.

# 2. How one call works

```
 browser page (one tap to start)                            Python server (this project)
 ───────────────────────────────                            ────────────────────────────
 speak the question  ◄──────────── say: "What is your age?" ────────────  call controller
        │                                                                  (state machine)
 open the microphone, wait for speech                                          ▲
 detect speech start … detect the pause that ends it                           │ text
 cut the utterance out, make a 16 kHz WAV ──────── POST /turn ──►  1A load → (1B) → 2A speech-to-text
        │                                                          → 2B extract the value / understand yes-no
 speak the next sentence  ◄───────── next turn ────────────────────────────────┘
 … until the call is done; the record is saved after every turn
```

**States of a call:** consent → ask field 1 → confirm → ask field 2 → confirm … → review → done. Extra paths: "no" → ask again; "no, it is 26" → corrected value is read back again; "say again" → the value is read again; silence → "I could not hear you" and the question is repeated; three silences → the call ends politely and the answers so far are saved; an optional question that cannot be understood after three tries is skipped, a required one hands over to a person ("needs help"); at the review the user can say which field to change ("change my phone number", "the second one").

# 3. The parts

| File | Role |
|---|---|
| `callpage/static/index.html` | The page: voice out, microphone, **voice-activity detection**, WAV encoding, the call loop, big accessible interface |
| `callpage/session.py` | **The call controller**: the state machine above; no audio, no web (fully unit-tested) |
| `callpage/forms.py` | Loads the form JSON the admin created; field names / aliases are those of 2B; custom fields are registered with 2B |
| `callpage/prompts.py` | Everything SVARA says, in English, Hindi, Malayalam, Kannada (see the warning in Section 9) |
| `callpage/pipeline.py` | The real speech path: 1A (load) → optional 1B → 2A (speech-to-text) |
| `callpage/tts.py` | Online voice (gTTS) with a disk cache; falls back to the browser's voice |
| `callpage/store.py`, `export.py` | Saves each call as JSON after every turn; table / CSV export |
| `callpage/server.py`, `launcher.py` | The small web server (standard library only) and a helper that starts it from Streamlit |
| `data/forms/clinic_registration.json` | A sample form (the 5-field clinic registration) |

It **reuses** 1A, 1B, 2A and 2B: 2B's `extract_field_full` reads the value, `handle_confirmation` understands "yes / no / no, it is 26 / say again", `spoken_form` decides how to read a value aloud (digits one by one, month names in the caller's language).

# 4. The form (how the admin's work becomes questions)

The Admin "set up a form" page must save a form file, `data/forms/<id>.json`. The easiest way from Python:

```python
from callpage.forms import FormStore, build_form_dict
FormStore("data/forms").save(build_form_dict(
    "Clinic registration", ["name", "age", "gender", "phone", {"name": "blood_group", "required": False}],
    language="en", languages=["en", "hi", "ml", "kn"]))
```

```json
{ "id": "clinic_registration", "title": "Clinic registration", "language": "en",
  "languages": ["en", "hi", "ml", "kn"],
  "settings": {"require_consent": true, "store_audio": false, "max_attempts": 3, "redact_sensitive_log": true},
  "fields": [
    {"name": "name", "required": true},
    {"name": "age",  "question": {"en": "How old are you?", "hi": "आपकी उम्र कितनी है?"}},
    {"name": "blood_group", "required": false},
    {"name": "ration_card_no", "kind": "text", "label": {"en": "ration card number"},
     "question": "What is your ration card number?"} ] }
```

* `name` is any field or alias of 2B ("age", "DOB", "mobile", "blood type", …). A name that 2B does not know is a **custom field**: give it a `kind` (default `text`; the kinds are `name, age, gender, phone, date, date_of_birth, address, pincode, aadhaar, blood_group, email, text, number, yes_no`).
* `question` and `label` can be omitted (built-in fields have questions and labels in four languages), a text, or an object per language.
* `required: false` lets the caller say "skip" for that question.
* Settings: `require_consent` (ask permission first), `store_audio` (keep the recordings, **only if the caller agreed**), `max_attempts`, `redact_sensitive_log`.
* Invalid forms give readable errors ("field 'age' appears twice", "unknown kind 'foo'").

# 5. What is stored

One file per call, `data/calls/<call_id>.json`, rewritten (atomically) **after every turn**:

* who/what: `call_id`, `form_id`, `form_title`, `language`, `started_at`, `ended_at`, `outcome` (`completed`, `declined`, `timed_out`, `needs_help`, `hangup`, `error`, `in_progress`), `consent`;
* per field: `value`, `display`, `status` (`confirmed`, `skipped`, `needs_help`, `pending`), `confidence`, `method` (rules / LLM), `attempts`, `rejections`, the raw recognised answers, `confirmed_at`;
* `events`: the whole conversation (what SVARA said, what the user said) with timestamps.

`python -c "..."` or the helpers in `callpage/export.py` turn calls into a table / CSV for the Admin page.

**Privacy (this is personal health data):** the files are plain text. Keep `data/calls/` out of git (it is in `.gitignore`), limit who can read it, delete old calls (`CallStore.delete`, `purge_older_than(days)`). Phone, Aadhaar, PIN and e-mail are redacted in the *conversation log* (the form value itself is stored, because the form needs it). Audio is saved only when the form allows it **and** the caller said yes. The call starts with a spoken consent question. Speech is sent to the cloud recogniser (Groq) and the online voice (Google) unless you use the local engine and the browser voice. **Nothing is encrypted at rest.**

# 6. How the page listens (voice-activity detection)

A real phone call needs the page to decide, by itself, **when the person starts and stops speaking**. The page does this in the browser:

1. The microphone is opened once (with the browser's echo cancellation and noise suppression). Audio is cut into 20 ms frames and the loudness (RMS) of each frame is measured.
2. **Adaptive threshold:** the background noise level is the 20th percentile of the last 3 seconds of loudness; speech means "louder than 3.2 × noise" (never below 0.015). So a noisy room and a quiet room both work.
3. **Speech starts** after 120 ms of loud frames; the 0.5 s *before* that are kept (so the first word is not cut off).
4. **Speech ends** after a pause: 0.84 s for yes/no turns, 1.2 s for answers (people pause while giving an address); adjustable in **Settings**. Maximum 25 s.
5. Sounds shorter than 0.2 s (a cough, a click) are ignored.
6. If nobody speaks for 7–9 s the server is told "no speech" and repeats the question.
7. The utterance is converted to **16 kHz mono WAV** and sent to the server.

The page **does not listen while it is speaking** (so it cannot hear itself), and waits 0.35 s after speaking for the room echo to die. That means a person cannot interrupt SVARA mid-sentence (no "barge-in"); see Section 9.

# 7. Run it

```powershell
pip install -r requirements.txt                    # (after python merge_callpage.py, which adds gTTS)
python -m callpage --open                          # opens http://127.0.0.1:8765
```
Press **Start call**, allow the microphone the first time, and speak. Use a **headset** for the best result; with laptop speakers keep the volume moderate. Options: `--port`, `--forms`, `--calls`, `--engine auto|cloud|local`, `--preprocess off|light|full`, `--no-tts` (use only the browser voice). Settings are read from `.env` (`GROQ_API_KEY`, …) like the rest of the project. The page opens directly on a form with `http://127.0.0.1:8765/?form=clinic_registration&lang=hi`.

**From the Streamlit app:** `from callpage.launcher import ensure_server, call_url` → `ensure_server()` starts the server in the background; `st.link_button("Start voice call", call_url(form_id, lang))` opens the call page in a new tab. (A separate page is used because a call needs the microphone and speaker to run continuously, which Streamlit's re-run model cannot do.)

**Speech-recognition settings:** the cloud engine is used first (about 0.4 s per answer in your test), the local model is the fallback. `--preprocess off` is the default because in my tests the 1B band-pass made Whisper worse on clean speech; try `light` on your own recordings.

# 8. Testing

| What | Result |
|---|---|
| Call controller: every path (happy call, corrections, say again, three failures, optional skip, review and change, silence, hang-up, consent declined, technical errors, Hindi call, redaction, saving after every turn) | 79 tests |
| Phrases in four languages (all keys present, placeholders match, **no stray characters from another script**), forms, call store (atomic saves, path-trick ids rejected) | 24 tests |
| HTTP server: a whole call over HTTP, token and Host-header checks, size limits, bad requests, leak-free errors, parallel calls, idle clean-up; launcher; export | 29 tests |
| **Real browser (Chromium with a simulated microphone), 3 complete calls:** happy path (12 spoken turns, one click), correction path ("no, say again", "no, it is twenty six", optional skip by voice), and a silent room (3 timeouts) | all pass |
| Whole project | 844 tests pass |

**Bugs the tests found (all fixed):** the "is this just noise?" filter cut Hindi / Malayalam / Kannada words apart and would have discarded every Hindi "हाँ" and a one-digit answer like "5"; the page sent an empty request instead of the recording (found only by the real-browser test); a test accidentally depended on a developer's `.env`.

# 9. What was NOT verified, and limits (stated plainly)

* **Real human speech through a real microphone was not tested.** The browser test uses a simulated microphone (looping synthetic speech-like bursts) and a **scripted** recogniser (I cannot synthesise spoken answers here). The voice-detection thresholds (3.2 × noise, 0.84 / 1.2 s pauses) are reasonable defaults, **not tuned on real voices and rooms**: try it, and use the pause setting if it cuts people off or waits too long.
* **The cloud recogniser and the online voice (gTTS) were not run inside the call loop** (my environment cannot reach them). Each was tested separately (your own cloud test; gTTS only as "falls back cleanly when unavailable").
* **Only Chromium was tested** (Chrome and Edge use it). Firefox should work; Safari was not considered. The page needs `http://localhost` (a secure context) to use the microphone.
* **The Hindi, Malayalam and Kannada sentences are drafts written without a native-speaker review**; they are spoken to patients, so have them reviewed (`callpage/prompts.py`, or override per form).
* **Malayalam recognition is still poor** with stock Whisper (cloud or local); the call can ask and read back, but understanding what the caller says depends on the recogniser. See the 2A document.
* **No barge-in:** the caller cannot interrupt SVARA mid-sentence (it would need echo-proof detection; planned for a later phase, headset-first).
* **One tap is still needed to start** (browser rule for sound and microphone), plus the one-time microphone permission. For kiosks, a browser started with an auto-play flag and a stored microphone permission can remove it.
* **Language is chosen before the call** (the form's default or the selector), not by voice, because automatic language detection is unreliable for Malayalam and Kannada.
* Long forms mean a long call. The simulated 5-field call (consent, five questions with read-backs, and the review) took about 36 seconds with instant scripted recognition; a real call also waits for the recogniser and the voice, and depends on how fast the caller speaks.
* **Not encrypted, no login, localhost only.** Anyone who can use the laptop can open the page and read the stored calls.

# 10. Security notes

The server listens only on `127.0.0.1`; the `Host` header must be local (blocks DNS-rebinding); every API call needs a random token that only the served page knows (other websites cannot drive it); uploads are size-limited; call ids are hex only (no path tricks); errors never show internals; sensitive values are redacted in logs.

# 11. Viva notes

* **How does it know when I stop talking?** Energy-based voice-activity detection with an adaptive noise floor, a pre-roll buffer, a minimum speech length and a pause timer (Section 6).
* **Why a separate page?** Streamlit re-runs the script on every event and its recorder needs a click; continuous listening needs a page that keeps the microphone open.
* **Why confirm by voice?** For users who cannot press buttons; it reuses 2B's yes / no / correction logic.
* **Why store after every turn?** A dropped call keeps its answers; the file is replaced atomically so it can never be half-written.
* **What happens when the recogniser is wrong?** Every value is read back; the user corrects it by voice; after repeated failures a required field hands over to a person, an optional one is skipped.
