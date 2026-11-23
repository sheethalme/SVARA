"""
session.py -- the CALL CONTROLLER: a state machine that runs one voice call (no HTTP, no browser, fully testable)
==================================================================================================================

        consent ──yes──► ask(field 1) ──answer──► confirm ──"yes"──► ask(field 2) ... ──► review ──"yes"──► done
           │ no                ▲  │ not understood   │ "no" / "say again" / "no, it is 26"        │ "no"
           ▼                   │  └─(3 tries)─► skip (optional) / needs_help (required)           ▼
         ended                 └─────────────────────────────────────────────────────────────── change
                                                                                      ("which one?") ──► ask(that field)

The controller never touches the microphone or the speaker.  The page (browser) does that:

    turn = session.start()                       # {"say": "...", "listen": true, ...}
    ... the page speaks turn.say, listens, cuts the user's utterance out of the microphone signal ...
    turn = session.handle_audio(wav_bytes)       # or handle_no_speech() when the user stayed silent
    ... until turn.done

Everything the user said and every value is stored in ``session.record()`` (saved after every turn, so a call that
drops half way keeps its answers).

Reuses 2B for the understanding:  ``extract_field_full`` (the value), ``handle_confirmation`` (yes / no / "no, it is
26" / "say again") and ``spoken_form`` (how to read a value aloud).
"""

from __future__ import annotations

import datetime as _dt
import re
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import extractor
from callpage import prompts
from callpage.forms import FormField, FormSpec
from lexicon import fold
from normalizers import tokenize

__all__ = ["TranscribeResult", "Turn", "FieldState", "CallSession", "is_noise_text", "match_field"]

PHASES = ("consent", "ask", "confirm", "review", "change", "ended")
MAX_SILENCE = 3          # consecutive "I heard nothing" turns before the call is ended
MAX_ERRORS = 3           # consecutive technical errors before the call is ended
MAX_UNCLEAR = 3          # consecutive unclear yes/no replies before the question counts as failed


@dataclass
class TranscribeResult:
    text: str = ""
    reliable: bool = True
    language: str = ""
    engine: str = ""
    warnings: List[str] = field(default_factory=list)
    skipped: Optional[str] = None          # "silence" / "too short": nothing to transcribe
    error: Optional[str] = None            # a technical problem (network, ...)


Transcriber = Callable[[bytes, str, Optional[str]], TranscribeResult]


@dataclass
class Turn:
    """What the page must do next."""
    say: str
    lang: str
    phase: str
    listen: bool = True                    # open the microphone after speaking?
    listen_mode: str = "answer"            # "answer" | "yesno": tells the page how long a pause ends the utterance
    done: bool = False
    outcome: Optional[str] = None          # completed | declined | timed_out | needs_help | hangup | error
    field: Optional[str] = None
    progress: Tuple[int, int] = (0, 0)     # (fields finished, fields in the form)
    heard: Optional[str] = None            # what the user said in the last turn (for captions)
    value: Optional[str] = None            # the value under confirmation / just stored (for captions)

    def to_dict(self) -> Dict[str, Any]:
        return {"say": self.say, "lang": self.lang, "phase": self.phase, "listen": self.listen,
                "listen_mode": self.listen_mode, "done": self.done, "outcome": self.outcome, "field": self.field,
                "progress": list(self.progress), "heard": self.heard, "value": self.value}


@dataclass
class FieldState:
    spec: FormField
    status: str = "pending"                # pending | confirmed | skipped | needs_help
    value: Optional[str] = None
    display: Optional[str] = None
    confidence: float = 0.0
    method: str = ""
    attempts: int = 0                      # answers that could not be read
    rejections: int = 0                    # values the user said "no" to
    raw_answers: List[str] = field(default_factory=list)
    confirmed_at: Optional[str] = None


# ------------------------------------------------------------------------------------------ helpers
_NOISE = {"", "you", "thank you", "thanks", "thank you.", "thanks for watching", "thank you for watching", "bye", "bye bye",
          "subscribe", "like and subscribe", "music", "[music]", "(music)", "uh", "um", "hmm", "mm", "mhm", "ah", "oh"}


def is_noise_text(text: str) -> bool:
    """Whisper often invents "Thank you." / "Thanks for watching" for silence and background noise.
    (Words are split with 2B's tokenizer: a plain ``\\w`` regex would cut Hindi / Malayalam / Kannada words apart.)"""
    t = " ".join(fold(w) for w in tokenize(text or ""))
    return t in _NOISE or not any(c.isalnum() for c in t)


def _is_skip(text: str) -> bool:
    t = fold(text)
    return any(fold(p) in t for p in prompts.SKIP_PHRASES)


def match_field(text: str, fields: List[FieldState], lang: str) -> Optional[int]:
    """Which field does "change my phone number" / "the second one" / "फ़ोन नंबर" mean?  Index or None."""
    t = fold(text)
    best, best_len, tie = None, 0, False
    for i, fs in enumerate(fields):
        names = {fs.spec.name.replace("_", " "), *fs.spec.label.values(), *prompts.LABELS.get(fs.spec.name, {}).values()}
        names |= {a.replace("_", " ") for a, c in extractor._ALIASES.items() if c == fs.spec.name}
        for n in map(fold, names):
            if len(n) < 3:
                continue
            found = re.search(r"(?<!\w)" + re.escape(n) + r"(?!\w)", t) if n.isascii() else (n in t)
            if not found:
                continue
            if len(n) > best_len:
                best, best_len, tie = i, len(n), False
            elif len(n) == best_len and best != i:
                tie = True
    if best is not None and not tie:
        return best
    if best is None:
        num = extractor.parse_number(text, 1, len(fields))
        if num is not None:
            return num - 1
    return None


def _iso(ts: float) -> str:
    return _dt.datetime.fromtimestamp(ts, _dt.timezone.utc).isoformat(timespec="seconds")


# ------------------------------------------------------------------------------------------ the call
class CallSession:
    def __init__(self, form: FormSpec, language: Optional[str] = None, transcriber: Optional[Transcriber] = None,
                 store: Any = None, clock: Callable[[], float] = time.time, call_id: Optional[str] = None,
                 today: Optional[_dt.date] = None, ex: Optional[extractor.Extractor] = None):
        self.form = form
        self.lang = prompts.norm_lang(language or form.language)
        self.transcriber = transcriber
        self.store = store
        self.clock = clock
        self.call_id = call_id or uuid.uuid4().hex[:16]
        self.today = today
        self.ex = ex if ex is not None else extractor.get_extractor()
        self.settings = dict(form.settings)
        self.fields = [FieldState(f) for f in form.fields]
        self.phase = "consent" if self.settings.get("require_consent", True) else "ask"
        self.idx = 0
        self.consent: Optional[bool] = None
        self.outcome: Optional[str] = None
        self.pending: Optional[extractor.ExtractionResult] = None
        self.silence = self.errors = self.unclear = self.change_fail = self.consent_tries = 0
        self.review_return = False
        self.started_at = self.ended_at = None
        self.events: List[Dict[str, Any]] = []
        self.last_say = ""
        self.audio_turns = 0

    # ------------------------------------------------------------------ public API
    def start(self) -> Turn:
        self.started_at = self.clock()
        if self.phase == "consent":
            text = prompts.say("greeting_consent", self.lang, title=self.form.title)
            return self._turn(text, "yesno")
        text = prompts.say("greeting", self.lang, title=self.form.title)
        return self._ask(prefix=text)

    def handle_audio(self, wav: bytes) -> Turn:
        if self.phase == "ended":
            return self._ended_turn()
        if self.transcriber is None:
            raise RuntimeError("no transcriber configured")
        self.audio_turns += 1
        hint = prompts.CONFIRM_HINT[self.lang] if self._mode() == "yesno" else None
        try:
            r = self.transcriber(wav, self.lang, hint)
        except Exception as exc:                          # noqa: BLE001 - a technical failure must not end the call abruptly
            r = TranscribeResult(error=f"{type(exc).__name__}: {exc}"[:200])
        if r.error:
            return self._on_error(r.error)
        self.errors = 0
        if r.skipped or not (r.text or "").strip():
            return self.handle_no_speech()
        return self.handle_text(r.text, r.reliable)

    def handle_text(self, text: str, reliable: bool = True) -> Turn:
        if self.phase == "ended":
            return self._ended_turn()
        text = re.sub(r"\s+", " ", text or "").strip()
        if is_noise_text(text):
            return self.handle_no_speech()
        self.silence = self.errors = 0
        fs = self._cur() if self.phase in ("ask", "confirm") else None
        sensitive = bool(fs and fs.spec.sensitive)
        self._log("user", text, redact=sensitive)
        handler = {"consent": self._on_consent, "ask": self._on_answer, "confirm": self._on_confirm,
                   "review": self._on_review, "change": self._on_change}[self.phase]
        turn = handler(text)
        turn.heard = text if not sensitive else "…"
        return turn

    def handle_no_speech(self) -> Turn:
        if self.phase == "ended":
            return self._ended_turn()
        self.silence += 1
        if self.silence >= MAX_SILENCE:
            return self._end("timed_out", prompts.say("timeout", self.lang))
        return self._repeat(prompts.say("didnt_hear", self.lang))

    def hangup(self) -> Turn:
        if self.phase == "ended":
            return self._ended_turn()
        return self._end("hangup", prompts.say("hangup", self.lang))

    @property
    def finished(self) -> bool:
        return self.phase == "ended"

    # ------------------------------------------------------------------ phases
    def _on_consent(self, text: str) -> Turn:
        c = extractor.interpret_confirmation_full(text, self.lang)
        if c.label == "yes":
            self.consent = True
            self.phase = "ask"
            return self._ask(prefix=prompts.say("thanks", self.lang))
        if c.label == "no":
            self.consent = False
            return self._end("declined", prompts.say("consent_declined", self.lang))
        self.consent_tries += 1
        if self.consent_tries >= MAX_UNCLEAR:
            self.consent = False
            return self._end("declined", prompts.say("consent_declined", self.lang))
        return self._turn(prompts.say("ask_yes_no", self.lang) + " " + prompts.say("greeting_consent", self.lang,
                                                                                       title=self.form.title), "yesno")

    def _on_answer(self, text: str) -> Turn:
        fs = self._cur()
        fs.raw_answers.append("[redacted]" if fs.spec.sensitive and self.settings.get("redact_sensitive_log", True) else text)
        if _is_skip(text):
            if not fs.spec.required:
                fs.status = "skipped"
                return self._advance(prefix=prompts.say("skipped", self.lang))
            return self._ask(prefix=prompts.say("required", self.lang))
        r = self.ex.extract(text, fs.spec.name, self.lang, today=self.today)
        if r.value is not None and r.status in ("ok", "needs_check"):
            self.pending = r
            self.unclear = 0
            self.phase = "confirm"
            return self._confirm_turn(first=True)
        fs.attempts += 1
        if fs.attempts >= int(self.settings.get("max_attempts", 3)):
            return self._give_up(fs)
        issue = r.issues[0] if (r.issues and self.lang == "en") else ""
        return self._ask(prefix=(prompts.say("didnt_catch", self.lang) + (" " + issue if issue else "")))

    def _on_confirm(self, text: str) -> Turn:
        fs = self._cur()
        o = extractor.handle_confirmation(text, fs.spec.name, self.lang, extractor=self.ex, today=self.today)
        if o.action == "accept":
            r = self.pending
            fs.status, fs.value, fs.display = "confirmed", r.value, r.display
            fs.confidence, fs.method = r.confidence, r.method
            fs.confirmed_at = _iso(self.clock())
            self.pending = None
            return self._advance()
        if o.action == "update":
            self.pending = o.extraction
            fs.rejections += 1
            if fs.rejections >= int(self.settings.get("max_attempts", 3)) + 1:
                return self._give_up(fs)
            return self._confirm_turn(first=False)
        if o.action == "repeat_readback":
            self.unclear = 0
            return self._confirm_turn(first=True)
        if o.action == "reask_field":
            fs.rejections += 1
            self.pending = None
            if fs.rejections >= int(self.settings.get("max_attempts", 3)):
                return self._give_up(fs)
            self.phase = "ask"
            return self._turn(prompts.say("reask", self.lang, label=fs.spec.label_for(self.lang)), "answer")
        self.unclear += 1                                  # ask_again: not a yes / no
        if self.unclear >= MAX_UNCLEAR:
            self.unclear = 0
            fs.rejections += 1
            if fs.rejections >= int(self.settings.get("max_attempts", 3)):
                return self._give_up(fs)
            self.pending = None
            self.phase = "ask"
            return self._ask(prefix=prompts.say("didnt_catch", self.lang))
        return self._turn(prompts.say("ask_yes_no", self.lang) + " " + self._confirm_text(first=True), "yesno",
                          field=fs.spec.name, value=self.pending.display if self.pending else None)

    def _on_review(self, text: str) -> Turn:
        c = extractor.interpret_confirmation_full(text, self.lang)
        if c.label == "yes":
            return self._end("completed", prompts.say("done", self.lang))
        if c.label == "no":
            self.phase, self.change_fail = "change", 0
            return self._turn(prompts.say("which_change", self.lang), "answer")
        if c.reason == "repeat_readback":
            return self._review_turn()
        self.unclear += 1
        if self.unclear >= MAX_UNCLEAR:
            return self._end("needs_help", prompts.say("trouble", self.lang))
        return self._turn(prompts.say("ask_yes_no", self.lang) + " " + prompts.say("review_question", self.lang), "yesno")

    def _on_change(self, text: str) -> Turn:
        i = match_field(text, self.fields, self.lang)
        if i is None:
            c = extractor.interpret_confirmation_full(text, self.lang)
            if c.label == "yes":                                              # "yes, it is fine" / "nothing to change"
                return self._end("completed", prompts.say("done", self.lang))
            self.change_fail += 1
            if self.change_fail >= 2:
                self.phase = "review"
                return self._review_turn()
            example = self.fields[0].spec.label_for(self.lang)
            return self._turn(prompts.say("which_change_again", self.lang, example=example), "answer")
        fs = self.fields[i]
        fs.status, fs.attempts, fs.rejections, fs.value, fs.display = "pending", 0, 0, None, None
        self.idx, self.review_return, self.phase, self.pending = i, True, "ask", None
        return self._ask()

    # ------------------------------------------------------------------ building turns
    def _cur(self) -> FieldState:
        return self.fields[self.idx]

    def _mode(self) -> str:
        return "yesno" if self.phase in ("consent", "confirm", "review") else "answer"

    def _progress(self) -> Tuple[int, int]:
        return (sum(1 for f in self.fields if f.status in ("confirmed", "skipped")), len(self.fields))

    def _turn(self, text: str, mode: str = "answer", **kw: Any) -> Turn:
        self.last_say = text
        fs = self.fields[self.idx] if self.phase in ("ask", "confirm") and self.idx < len(self.fields) else None
        sens = bool(fs and fs.spec.sensitive and self.phase == "confirm")
        self._log("system", text, redact=sens)
        kw.setdefault("field", fs.spec.name if fs else None)
        t = Turn(say=text, lang=self.lang, phase=self.phase, listen=True, listen_mode=mode, progress=self._progress(), **kw)
        self._save()
        return t

    def _ask(self, prefix: str = "") -> Turn:
        return self._turn((prefix + " " + self._cur().spec.question_for(self.lang)).strip(), "answer")

    def _confirm_text(self, first: bool) -> str:
        fs = self._cur()
        v = prompts.spoken_value(fs.spec.kind, self.pending.value if self.pending else None, self.lang)
        return prompts.say("confirm" if first else "confirm_again", self.lang, label=fs.spec.label_for(self.lang), value=v)

    def _confirm_turn(self, first: bool) -> Turn:
        return self._turn(self._confirm_text(first), "yesno", value=self.pending.display if self.pending else None)

    def _review_turn(self) -> Turn:
        parts = [prompts.say("review_intro", self.lang)]
        for fs in self.fields:
            label = fs.spec.label_for(self.lang)
            if fs.status == "confirmed":
                parts.append(prompts.say("review_item", self.lang, label=label,
                                         value=prompts.spoken_value(fs.spec.kind, fs.value, self.lang)))
            else:
                parts.append(prompts.say("review_skipped", self.lang, label=label))
        parts.append(prompts.say("review_question", self.lang))
        self.phase = "review"
        text = " ".join(parts)
        self.last_say = text
        self._log("system", text, redact=any(f.spec.sensitive and f.status == "confirmed" for f in self.fields))
        t = Turn(say=text, lang=self.lang, phase="review", listen=True, listen_mode="yesno", progress=self._progress())
        self._save()
        return t

    def _repeat(self, prefix: str) -> Turn:
        """Say ``prefix`` and ask the current question again (after silence or an error)."""
        if self.phase == "consent":
            return self._turn(prefix + " " + prompts.say("greeting_consent", self.lang, title=self.form.title), "yesno")
        if self.phase == "ask":
            return self._ask(prefix=prefix)
        if self.phase == "confirm":
            return self._turn(prefix + " " + self._confirm_text(first=True), "yesno",
                              value=self.pending.display if self.pending else None)
        if self.phase == "review":
            return self._turn(prefix + " " + prompts.say("review_question", self.lang), "yesno")
        return self._turn(prefix + " " + prompts.say("which_change", self.lang), "answer")

    def _on_error(self, message: str) -> Turn:
        self.errors += 1
        self._log("system", f"[technical error] {message}")
        if self.errors >= MAX_ERRORS:
            return self._end("error", prompts.say("trouble", self.lang))
        return self._repeat(prompts.say("error", self.lang))

    def _advance(self, prefix: str = "") -> Turn:
        """Move to the next field that still needs an answer, or to the review."""
        self.unclear = 0
        if self.review_return:
            self.review_return = False
            return self._review_turn()
        for i in range(self.idx + 1, len(self.fields)):
            if self.fields[i].status == "pending":
                self.idx, self.phase = i, "ask"
                return self._ask(prefix=prefix)
        return self._review_turn()

    def _give_up(self, fs: FieldState) -> Turn:
        self.pending = None
        if not fs.spec.required:
            fs.status = "skipped"
            self.phase = "ask"
            return self._advance(prefix=prompts.say("skipped", self.lang))
        fs.status = "needs_help"
        return self._end("needs_help", prompts.say("trouble", self.lang))

    def _end(self, outcome: str, text: str) -> Turn:
        self.outcome, self.phase = outcome, "ended"
        self.ended_at = self.clock()
        self.last_say = text
        self._log("system", text)
        t = Turn(say=text, lang=self.lang, phase="ended", listen=False, done=True, outcome=outcome,
                 progress=self._progress())
        self._save()
        return t

    def _ended_turn(self) -> Turn:
        return Turn(say="", lang=self.lang, phase="ended", listen=False, done=True, outcome=self.outcome,
                    progress=self._progress())

    # ------------------------------------------------------------------ record
    def _log(self, role: str, text: str, redact: bool = False) -> None:
        if redact and self.settings.get("redact_sensitive_log", True):
            text = "[redacted]"
        self.events.append({"t": _iso(self.clock()), "role": role, "text": text, "phase": self.phase,
                            "field": self.fields[self.idx].spec.name if self.phase in ("ask", "confirm") and self.idx < len(self.fields) else None})

    def record(self) -> Dict[str, Any]:
        return {
            "call_id": self.call_id, "form_id": self.form.id, "form_title": self.form.title, "language": self.lang,
            "started_at": _iso(self.started_at) if self.started_at else None,
            "ended_at": _iso(self.ended_at) if self.ended_at else None,
            "outcome": self.outcome or "in_progress", "consent": self.consent, "phase": self.phase,
            "progress": list(self._progress()),
            "fields": [{"name": f.spec.name, "label": f.spec.label_for(self.lang), "kind": f.spec.kind,
                        "required": f.spec.required, "sensitive": f.spec.sensitive, "status": f.status,
                        "value": f.value, "display": f.display, "confidence": round(f.confidence, 3),
                        "method": f.method, "attempts": f.attempts, "rejections": f.rejections,
                        "raw_answers": f.raw_answers, "confirmed_at": f.confirmed_at} for f in self.fields],
            "events": self.events,
            "settings": {k: self.settings[k] for k in sorted(self.settings)},
        }

    def _save(self) -> None:
        if self.store is not None:
            try:
                self.store.save(self.record())
            except OSError:                                # the call must go on even if the disk is full
                pass
