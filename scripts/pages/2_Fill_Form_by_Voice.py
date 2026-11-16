"""User page: the system asks one question at a time (spoken and written); the user answers and corrects by voice."""
import io, json, os, sys
import streamlit as st
import streamlit.components.v1 as components

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "scripts"))
os.environ.setdefault("SVARA_WHISPER_MODEL", "large-v3-turbo")
import audio_io
import formflow as ff

st.set_page_config(page_title="SVARA: fill a form by voice", layout="centered")
st.title("Fill a form by voice")


def speak(text: str, lang: str, turn: int) -> None:
    """Read `text` aloud with the browser's own speech synthesis (no install, no cost)."""
    components.html(
        f"""<button id="b" style="font:14px sans-serif;padding:4px 10px">🔊 Replay question</button>
<script>
const t={json.dumps(text)}, l={json.dumps(ff.TTS_LANG.get(lang, 'en-IN'))};
function say(){{ speechSynthesis.cancel(); const u=new SpeechSynthesisUtterance(t); u.lang=l; speechSynthesis.speak(u); }}
document.getElementById('b').onclick=say; say();
</script><!-- turn {turn} -->""", height=44)


forms = ff.list_forms()
if not forms:
    st.info("No form has been set up yet. Ask the admin to create one on the 'Admin Form Setup' page.")
    st.stop()

flow = st.session_state.get("flow")
if flow is None:
    by_title = {f"{f['title']}  ({len(f['fields'])} fields)": f for f in forms}
    pick = st.selectbox("Which form do you want to fill?", list(by_title))
    lang = st.selectbox("Language you will speak", list(ff.LANGUAGES), format_func=ff.LANGUAGES.get)
    st.caption("Allow microphone access when your browser asks. Speak clearly after pressing the record button.")
    if st.button("Start", type="primary"):
        st.session_state["flow"] = ff.FormFlow(by_title[pick], lang)
        st.rerun()
    st.stop()

form, lang = flow.form, flow.language
st.subheader(form["title"])
st.progress(flow.progress, text=f"{min(flow.idx, len(form['fields']))} of {len(form['fields'])} fields done")

if not flow.done:
    st.markdown(f"### {flow.say}")
    speak(flow.say, lang, flow.turn)
    for n in flow.notes:
        st.warning(n)
    if st.session_state.get("err"):
        st.error(st.session_state["err"])
    if flow.heard:
        st.caption(f"I heard you say: “{flow.heard}”")
    if flow.phase == "confirm":
        y, n = st.columns(2)
        if y.button("✅ Yes, it is correct", type="primary", use_container_width=True, key=f"yes_{flow.turn}"):
            flow.handle("yes"); st.rerun()
        if n.button("❌ No, change it", use_container_width=True, key=f"no_{flow.turn}"):
            flow.handle("no"); st.rerun()
        st.caption("Or use the microphone: say “yes”, or say the correct value "
                   "(you can spell it out, for example “no, it is A D H I R A”).")
    audio = st.audio_input("Press to record your answer, press again to stop", key=f"mic_{flow.turn}")
    if audio is not None:
        with st.spinner("Listening..."):
            try:
                heard = ff.transcribe(audio, language=lang)
                st.session_state.pop("err", None)
            except Exception as e:                           # keep the message: st.rerun() would wipe it
                heard = ""; st.session_state["err"] = f"Could not process the recording: {type(e).__name__}: {e}"
        flow.handle(heard)
        st.rerun()
    c1, c2 = st.columns(2)
    if not flow.field.get("required", True) and c1.button("Skip this field"):
        flow.skip(); st.rerun()
    if flow.phase == "ask":                                  # no typing for yes/no: the buttons above do that
        with c2.expander("Microphone not working? Type your answer instead"):
            typed = st.text_input("Type your answer", key=f"typed_{flow.turn}")
            if typed:
                flow.handle(typed); st.rerun()
else:
    st.success("All done. Please check your answers below.")
    speak(flow.say, lang, flow.turn)

if flow.rows():
    st.markdown("#### Your answers")
    st.table([{"Field": r["label"].capitalize(), "Answer": r["display"],
               "Status": "confirmed" if r["status"] == "ok" else r["status"]} for r in flow.rows()])

if flow.done:
    rows = flow.rows()
    b1, b2, b3 = st.columns(3)
    if b1.button("Submit", type="primary"):
        st.success(f"Saved to {os.path.relpath(ff.save_submission(form, rows), ROOT)}")
    b2.download_button("Download CSV", "field,answer\n" + "\n".join(f'"{r["label"]}","{r["display"]}"' for r in rows),
                       file_name="answers.csv")
    if b3.button("Start over"):
        st.session_state.pop("flow"); st.rerun()
elif st.button("Cancel and start over"):
    st.session_state.pop("flow"); st.rerun()
