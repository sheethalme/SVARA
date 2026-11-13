"""Stand-in evaluation clips for 1C's evaluate.py, made with the Windows text-to-speech voices.

    python scripts/make_tts_clips.py                 # -> data/test_clips_tts/*.wav + references.csv

The real experiment uses the 20-30 clips that 1A records from team members (with their
reference transcripts).  Until those exist, this script synthesises 24 English form
answers (names, ages, dates, phone numbers, addresses) with the built-in Windows voices
(System.Speech: David, Zira, Hazel) so that the WER pipeline can be run and checked end to
end on *real words*.  TTS speech is far cleaner and more regular than a real speaker, so
its WER numbers are a sanity check, not a result.

Each clip is 16 kHz mono with 0.6 s of silence before and after the answer (as a recording
made with the app would have), and the CSV has the columns ``file,transcript,language``.
Windows only (needs PowerShell and System.Speech).
"""
import csv, os, subprocess, sys, tempfile
import numpy as np, soundfile as sf

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
OUT = os.path.join(ROOT, "data", "test_clips_tts")
SR = 16000
PAD_S = 0.6

ANSWERS = [
    "my name is athira mohan",
    "my name is ravi kumar",
    "the patient name is lakshmi narayanan",
    "i am twenty five years old",
    "my age is sixty two",
    "he is eight years old",
    "my date of birth is fifteenth march two thousand",
    "i was born on the second of october nineteen eighty",
    "the appointment is on monday the fourth of june",
    "my phone number is nine eight four seven two one six five three zero",
    "you can call me on seven seven zero one four four two nine eight six",
    "my mobile number is eight one two three four five six seven eight nine",
    "i live at twelve gandhi road kochi",
    "my address is forty five mg road bangalore",
    "house number seven church street mysore",
    "the pin code is five six zero zero two nine",
    "my blood group is b positive",
    "i am allergic to penicillin",
    "i have had a fever and a cough for three days",
    "my father's name is mohan das",
    "i am a farmer from the village",
    "my aadhaar number is four five six seven one two three four",
    "yes that is correct",
    "no please say it again",
]
VOICES = ["Microsoft David Desktop", "Microsoft Zira Desktop", "Microsoft Hazel Desktop"]

PS = r"""
Add-Type -AssemblyName System.Speech
$s = New-Object System.Speech.Synthesis.SpeechSynthesizer
$fmt = New-Object System.Speech.AudioFormat.SpeechAudioFormatInfo(16000, [System.Speech.AudioFormat.AudioBitsPerSample]::Sixteen, [System.Speech.AudioFormat.AudioChannel]::Mono)
$s.Rate = 0
foreach ($line in Get-Content -LiteralPath $args[0] -Encoding UTF8) {
    $voice, $path, $text = $line -split "`t", 3
    $s.SelectVoice($voice)
    $s.SetOutputToWaveFile($path, $fmt)
    $s.Speak($text)
}
$s.SetOutputToNull()
"""


def build(out=OUT):
    os.makedirs(out, exist_ok=True)
    jobs = [(VOICES[i % len(VOICES)], os.path.join(out, f"clip{i + 1:02d}.wav"), text)
            for i, text in enumerate(ANSWERS)]
    with tempfile.TemporaryDirectory(prefix="svara_tts_") as tmp:
        script, todo = os.path.join(tmp, "tts.ps1"), os.path.join(tmp, "jobs.txt")
        with open(script, "w", encoding="utf-8") as fh:
            fh.write(PS)
        with open(todo, "w", encoding="utf-8") as fh:
            fh.write("\n".join("\t".join((v, os.path.abspath(p), t)) for v, p, t in jobs))
        subprocess.run(["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File",
                        script, todo], check=True)
    pad = np.zeros(int(PAD_S * SR), np.float32)
    for _, path, _ in jobs:                       # add the leading / trailing pause
        x, sr = sf.read(path, dtype="float32")
        assert sr == SR, sr
        sf.write(path, np.concatenate([pad, x, pad]), SR, subtype="PCM_16")
    with open(os.path.join(out, "references.csv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["file", "transcript", "language"])
        w.writerows([os.path.basename(p), t, "en"] for _, p, t in jobs)
    return [os.path.basename(p) for _, p, _ in jobs]


if __name__ == "__main__":
    if sys.platform != "win32":
        sys.exit("make_tts_clips.py needs Windows (System.Speech). Record real clips instead.")
    print(build())
