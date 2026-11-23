# Prompt for the VS Code agent: integrate the hands-free voice-call page into the SVARA app

Copy everything below the line into your VS Code agent (Claude Code / Copilot agent mode) with the project open.

---

You are working in the SVARA repository (Streamlit app + Python modules). Read this whole message first, then explore the repo before changing anything.

## Context
SVARA lets people fill in forms (clinic registration, government schemes) by speaking Malayalam, Kannada, Hindi or English. Modules already in the repo (do NOT rewrite them): `audio_io.py` (1A), `preprocess.py` (1B), `stt.py` (2A speech-to-text, cloud Groq + local Whisper), `extractor.py` + `lexicon.py` + `normalizers.py` (2B field extraction and yes/no/"say again" understanding), and the Streamlit app with pages **"try it app"**, **"Admin Form Setup"** (admin picks fields in order, can add custom fields, writes the questions) and **"Fill Form by Voice"** (the current flow needs buttons: record, play, "Yes, it is correct" / "No, change it").

Many of our users cannot click, tap or hold buttons. We added a **hands-free "phone call" page**: after ONE tap on "Start call" everything is by voice (the system asks, listens by itself, reads the answer back, the user says yes / no / "no, it is 26" / "say again"; every detail is saved). It is a separate small web server (standard library only) because Streamlit cannot keep a microphone open continuously.

New files that were just added to the repo root (already unzipped; do not rewrite them, they have 130+ passing tests and a real-browser end-to-end test):
- `callpage/` : `session.py` (call controller / state machine), `forms.py` (`FormSpec`, `FormStore`, `build_form_dict`, `slugify`), `prompts.py`, `pipeline.py`, `tts.py`, `store.py` (`CallStore`), `export.py` (`records`, `to_rows`, `to_csv`), `server.py`, `launcher.py` (`ensure_server`, `call_url`, `stop_server`), `static/index.html`
- `data/forms/clinic_registration.json` (a sample form), `merge_callpage.py`, tests `tests/test_callpage_*.py`, `tests/e2e/`, docs `docs/Call_Page_Hands_Free.md`.
Read `docs/Call_Page_Hands_Free.md` sections 3, 4, 5 and 7 first.

## Goal
When an admin creates a form on "Admin Form Setup", the voice call must ask **exactly the fields, in the order, with the questions the admin defined**, and the call results must be visible to the admin. Add a button on "Fill Form by Voice" that starts the hands-free call for the selected form.

## Step 0: set up and explore (report findings before big edits)
1. Run `python merge_callpage.py`, then `pip install -r requirements.txt`, then `python -m pytest tests/test_callpage_session.py tests/test_callpage_core.py tests/test_callpage_server.py tests/test_callpage_extras.py -q`. They must pass.
2. Find the Streamlit pages ("Admin Form Setup", "Fill Form by Voice", "try it app"). Find out **how the admin's form is stored today** (session_state? a JSON/SQLite file? a module?), which field names/options exist (the "Pick the fields" list, "Add your own" custom fields, the "3. Questions" section, any language/required settings) and how "Fill Form by Voice" reads them. Summarise this in 10 lines.

## Tasks
1. **Form adapter.** Create `callpage/admin_adapter.py` with a pure function that converts the admin's form (however it is stored) into the call-page JSON using `callpage.forms.build_form_dict` and saves it with `FormStore("data/forms").save(...)`.
   - field order = the order the admin picked; built-in fields use their canonical names (the names/aliases in `extractor.list_fields()`; check `extractor.resolve_field`);
   - custom fields -> `{"name": <slug>, "kind": "text" (or number / date / yes_no / email if the admin chose a type), "label": {...}, "question": {...}}`;
   - the admin's custom questions -> `"question"` (text or per-language object); "required" flags -> `"required"`; form language(s) -> `language` / `languages`;
   - call it automatically when the admin **saves** a form, and also provide `sync_all_forms()` (used when "Fill Form by Voice" opens) so old forms get a JSON too;
   - on `FormError` show `st.error` with the message (it is readable). Keep the existing storage working (single source of truth: the existing storage; the JSON is an export).
2. **"Fill Form by Voice" page.** Add a prominent **"Start hands-free voice call"** button (`st.link_button`) using `from callpage.launcher import ensure_server, call_url`: call `ensure_server()` (start the server in the background; show `st.error` with the log path on failure) and link to `call_url(form_id, language)`. Keep the existing button-based flow below it, labelled **"Manual mode (buttons)"**. Add a short hint: use a headset, allow the microphone, Chrome or Edge, speak after the beep-free prompt.
3. **Results.** On the Admin side (or a new page "Calls") show the stored calls using `callpage.export` (`records`, `to_rows`, `to_csv`): a table per form, a CSV download button, sensitive values **masked by default** (`mask_sensitive=True`, with an explicit "show" toggle), the conversation of one selected call (`events`), and a delete button with confirmation (`CallStore.delete`). Never display the API key or `.env` contents.
4. **Tests.** Add `tests/test_admin_adapter.py` (tmp dirs, no network): ordering, built-in vs custom fields, custom questions, required flags, invalid input -> readable error, `sync_all_forms`. Run the WHOLE suite: `python -m pytest -q`. If Node + Playwright are installed also run `node tests/e2e/call_e2e.cjs happy`.
5. **Docs.** Add a short section to the README: how an admin creates a form and the user starts a call; the commands `python -m callpage --open` and `streamlit run ...`.

## Rules
- Python 3.11 compatible; **no new heavy dependencies** (only `gTTS`, already added by `merge_callpage.py`).
- Do not modify `callpage/*.py` core files, 1A/1B/2A/2B modules or existing tests unless a real bug forces it; if you must, add a test and say so.
- Never print, log, commit or paste API keys. `.env`, `data/calls/`, `data/tts_cache/` and `callpage_server.log` must stay out of git (check `git status` before finishing; do NOT commit or push, I will review).
- Keep changes small and in the existing code style; do not restructure the app.
- The page must work with **no key** (rules-only extraction, local Whisper fallback, browser voice) and with a `GROQ_API_KEY` in `.env` (cloud recognition).
- If something is ambiguous (for example how the admin page stores forms), stop and ask me one clear question instead of guessing.

## Acceptance checks (do them and report the result of each)
1. Admin creates "Test clinic" with fields name, age, phone, one custom field ("ration card number") and a custom question for age -> `data/forms/test_clinic.json` exists, valid, same order.
2. "Fill Form by Voice" -> select that form -> **Start hands-free voice call** opens the call page, and the call asks: consent, then the questions in that order (the custom age question and the custom field included).
3. Full test suite passes; `git status` shows no secrets, no `data/calls/`.
4. List every file you changed or created and anything you could not verify (e.g. a real microphone test needs me).

Finish with a summary: what you changed, how to run it, what is left.
