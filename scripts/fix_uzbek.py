"""Re-transcribe chunks whose auto-detected language is not ru/en as Uzbek.

Whisper's language detector confuses Uzbek with Kazakh/Azerbaijani/Turkish
(both are Turkic). Forcing language='uz' on those chunks yields clean
Latin-script Uzbek. ru and en chunks are kept as-is.

Recomputes the identical VAD chunk layout (deterministic), replaces the
mislabeled chunks' segments, and rewrites output/raw_segments.json.
"""
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))
from run_pipeline import AUDIO_PATH, OUT_DIR, chunk_speech, read_wav, setup_cuda_dll_path  # noqa: E402

KEEP = {"ru", "en"}

setup_cuda_dll_path()
from faster_whisper import WhisperModel  # noqa: E402


def main():
    raw_path = os.path.join(OUT_DIR, "raw_segments.json")
    segs = json.load(open(raw_path, encoding="utf-8"))
    audio, sr = read_wav(AUDIO_PATH)
    chunks = chunk_speech(audio)
    print(f"chunks: {len(chunks)}", flush=True)

    chunk_lang = {}
    for i, (cs, ce) in enumerate(chunks):
        for s in segs:
            if cs <= s["start"] < ce:
                chunk_lang[i] = s["language"]
                break

    redo = {i for i, lang in chunk_lang.items() if lang not in KEEP}
    print(f"chunks to redo as Uzbek: {len(redo)} "
          f"({sum(1 for i in redo for s in segs if chunks[i][0] <= s['start'] < chunks[i][1])} segments)",
          flush=True)

    model = WhisperModel("large-v3", device="cuda", compute_type="float16")

    new_segs = []
    t0 = time.time()
    done = 0
    for i, (cs, ce) in enumerate(chunks):
        if i in redo:
            chunk_audio = audio[int(cs * sr):int(ce * sr)]
            if len(chunk_audio) < sr // 10:
                continue
            try:
                it_segs, info = model.transcribe(
                    chunk_audio,
                    language="uz",
                    beam_size=5,
                    temperature=0.0,
                    condition_on_previous_text=False,
                    vad_filter=False,
                    no_speech_threshold=0.55,
                    compression_ratio_threshold=2.4,
                )
                for s in it_segs:
                    t = s.text.strip()
                    if t:
                        new_segs.append({
                            "start": cs + s.start,
                            "end": cs + s.end,
                            "language": "uz",
                            "lang_prob": 1.0,
                            "text": t,
                        })
            except Exception as ex:
                print(f"chunk {i} failed ({ex}); keeping original", flush=True)
                new_segs.extend(s for s in segs if cs <= s["start"] < ce)
            done += 1
            if done % 10 == 0:
                print(f"  {done}/{len(redo)} re-transcribed, "
                      f"{time.time() - t0:.0f}s elapsed", flush=True)
        else:
            new_segs.extend(s for s in segs if cs <= s["start"] < ce)

    new_segs.sort(key=lambda s: s["start"])
    with open(raw_path, "w", encoding="utf-8") as f:
        json.dump(new_segs, f, ensure_ascii=False, indent=1)
    print(f"done: {len(new_segs)} segments total, "
          f"{time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
