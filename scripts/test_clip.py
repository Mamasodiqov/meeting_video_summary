"""Quick GPU sanity test: transcribe a 60s slice of the meeting (20:00-21:00)."""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))
from run_pipeline import AUDIO_PATH, PROJECT  # noqa: E402

from run_pipeline import read_wav, setup_cuda_dll_path  # noqa: E402
setup_cuda_dll_path()

from faster_whisper import WhisperModel  # noqa: E402

START, END = 20 * 60, 21 * 60

audio, sr = read_wav(AUDIO_PATH)
clip = audio[int(START * sr):int(END * sr)]

t0 = time.time()
model = WhisperModel("large-v3", device="cuda", compute_type="float16")
print(f"model load: {time.time() - t0:.0f}s", flush=True)

t0 = time.time()
segments, info = model.transcribe(
    clip, language=None, beam_size=5, condition_on_previous_text=False,
)
print(f"detected language: {info.language} (p={info.language_probability:.2f})", flush=True)
for s in segments:
    print(f"{s.start + START:8.2f} -> {s.end + START:8.2f}  {s.text.strip()}", flush=True)
print(f"transcribe: {time.time() - t0:.1f}s for 60s of audio", flush=True)
