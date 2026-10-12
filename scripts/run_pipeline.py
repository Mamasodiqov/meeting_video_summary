"""Meeting transcription pipeline: VAD chunking -> per-chunk Whisper (auto language)
-> pyannote diarization -> speaker-assigned, ordered transcript.

Outputs (in <project>/output/):
  diarization.json  - raw speaker timeline
  raw_segments.json - whisper segments with global timestamps
  segments.json     - final speaker-assigned, merged turns
  transcript.txt    - readable ordered transcript
  transcript.srt    - SRT subtitles with speaker labels

Usage:
  python scripts/run_pipeline.py [--redo-transcribe] [--redo-diarize]
"""
import argparse
import json
import os
import sys
import time
import wave
from collections import Counter

PROJECT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
AUDIO_PATH = os.path.join(PROJECT, "audio", "meeting_16k.wav")
OUT_DIR = os.path.join(PROJECT, "output")

LANG_NAMES = {"uz": "Uzbek", "en": "English", "ru": "Russian"}


def load_env():
    env = {}
    with open(os.path.join(PROJECT, ".env"), encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                env[k.strip()] = v.strip()
    return env


def read_wav(path):
    """Load a 16-bit mono WAV as float32 numpy (stdlib, avoids PyAV issues)."""
    import numpy as np

    with wave.open(path, "rb") as w:
        assert w.getsampwidth() == 2 and w.getnchannels() == 1, "expected 16-bit mono WAV"
        sr = w.getframerate()
        data = w.readframes(w.getnframes())
    audio = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
    return audio, sr


def setup_cuda_dll_path():
    """Add pip nvidia-* DLL dirs to PATH for ctranslate2 GPU support (Windows)."""
    import sysconfig

    sp = sysconfig.get_paths()["purelib"]
    for sub in ("nvidia/cublas/bin", "nvidia/cudnn/bin",
                "nvidia/cuda_runtime/bin", "nvidia/cuda_nvrtc/bin"):
        d = os.path.join(sp, *sub.split("/"))
        if os.path.isdir(d):
            os.environ["PATH"] = d + os.pathsep + os.environ.get("PATH", "")


def fmt_ts(sec: float) -> str:
    ms = int(round(sec * 1000))
    h, ms = divmod(ms, 3600000)
    m, ms = divmod(ms, 60000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def fmt_ts_dot(sec: float) -> str:
    ms = int(round(sec * 1000))
    h, ms = divmod(ms, 3600000)
    m, ms = divmod(ms, 60000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"


def chunk_speech(audio, sr=16000, min_silence_ms=500, pad_ms=300,
                 merge_gap_ms=250, max_chunk_ms=25000):
    """Run silero VAD, merge close chunks, cap chunk length."""
    from faster_whisper.vad import VadOptions, get_speech_timestamps

    options = VadOptions(min_silence_duration_ms=min_silence_ms, speech_pad_ms=pad_ms)
    stamps = get_speech_timestamps(audio, vad_options=options, sampling_rate=sr)
    chunks = [(float(s["start"]) / sr, float(s["end"]) / sr) for s in stamps]

    merged = []
    for s, e in chunks:
        if merged and s - merged[-1][1] < merge_gap_ms / 1000.0:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])

    capped = []
    for s, e in merged:
        while e - s > max_chunk_ms / 1000.0:
            capped.append((s, s + max_chunk_ms / 1000.0))
            s = s + max_chunk_ms / 1000.0
        capped.append((s, e))
    return capped


def run_diarization(token, audio, sr=16000):
    import numpy as np
    import torch
    from pyannote.audio import Pipeline

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[diarize] using device: {device}", flush=True)
    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1", token=token
    )
    pipeline.to(device)

    # Pass waveform directly (bypasses pyannote 4.x file loading via torchcodec)
    waveform = torch.from_numpy(np.ascontiguousarray(audio, dtype=np.float32)).unsqueeze(0)
    print("[diarize] running pyannote on 56 min of audio ...", flush=True)
    t0 = time.time()
    diag = pipeline({"waveform": waveform, "sample_rate": sr})

    # pyannote 4.x returns a DiarizeOutput; 3.x returned an iterable directly
    sd = getattr(diag, "speaker_diarization", diag)
    turns = []
    for item in sd:
        parts = list(item)
        seg = parts[0]
        turns.append({"start": float(seg.start), "end": float(seg.end),
                      "speaker": str(parts[-1])})
    speakers = sorted({t["speaker"] for t in turns})
    print(f"[diarize] done in {time.time() - t0:.0f}s, speakers: {speakers}", flush=True)
    return turns, speakers


def transcribe_chunks(audio, chunks):
    from faster_whisper import WhisperModel

    device = "cuda"
    compute = "float16"
    print(f"[whisper] loading large-v3 on {device}/{compute} ...", flush=True)
    t0 = time.time()
    model = WhisperModel("large-v3", device=device, compute_type=compute)
    print(f"[whisper] model loaded in {time.time() - t0:.0f}s", flush=True)

    raw = []
    n = len(chunks)
    t1 = time.time()
    for i, (cs, ce) in enumerate(chunks):
        chunk = audio[int(cs * 16000):int(ce * 16000)]
        if len(chunk) < 1600:  # <0.1s, skip
            continue
        try:
            segments, info = model.transcribe(
                chunk,
                language=None,
                beam_size=5,
                temperature=0.0,
                condition_on_previous_text=False,
                vad_filter=False,
                no_speech_threshold=0.55,
                compression_ratio_threshold=2.4,
            )
            lang = info.language
            for seg in segments:
                txt = seg.text.strip()
                if not txt:
                    continue
                raw.append({
                    "start": cs + seg.start,
                    "end": cs + seg.end,
                    "language": lang,
                    "lang_prob": round(info.language_probability, 3),
                    "text": txt,
                })
        except Exception as ex:
            print(f"[whisper] chunk {i} FAILED: {type(ex).__name__} {ex}", flush=True)
        if i % 25 == 0 or i == n - 1:
            el = time.time() - t1
            rate = (i + 1) / el if el > 0 else 0
            eta = (n - i - 1) / rate if rate > 0 else 0
            print(f"[whisper] {i + 1}/{n} chunks, {len(raw)} segments, "
                  f"{rate * 100:.0f}%/min, ETA {eta / 60:.1f} min", flush=True)
    return raw


def assign_speakers(raw, diar_turns):
    """Assign each segment to the diarization speaker with max overlap."""
    speakers = sorted({t["speaker"] for t in diar_turns})
    out = []
    for seg in raw:
        best, best_ov = None, 0.0
        for t in diar_turns:
            ov = min(seg["end"], t["end"]) - max(seg["start"], t["start"])
            if ov > best_ov:
                best, best_ov = t["speaker"], ov
        out.append({**seg, "speaker": best if best is not None else speakers[0]})
    return out


def merge_turns(segs, max_gap=1.2):
    """Merge consecutive same-speaker segments into speaking turns."""
    turns = []
    for seg in segs:
        if (turns and turns[-1]["speaker"] == seg["speaker"]
                and seg["start"] - turns[-1]["end"] <= max_gap):
            t = turns[-1]
            t["end"] = max(t["end"], seg["end"])
            t["text"] = (t["text"] + " " + seg["text"]).strip()
            t["n_raw"] += 1
        else:
            turns.append({
                "start": seg["start"],
                "end": seg["end"],
                "speaker": seg["speaker"],
                "language": seg["language"],
                "text": seg["text"],
                "n_raw": 1,
            })
    return turns


def write_outputs(turns, raw, diar_turns, speakers):
    os.makedirs(OUT_DIR, exist_ok=True)
    dur = raw[-1]["end"] if raw else 0.0
    lang_stat = Counter(s["language"] for s in raw)
    spk_stat = Counter(t["speaker"] for t in turns)
    spk_dur = {}
    for t in turns:
        spk_dur[t["speaker"]] = spk_dur.get(t["speaker"], 0.0) + (t["end"] - t["start"])

    segments = {
        "source_audio": os.path.relpath(AUDIO_PATH, PROJECT),
        "duration_sec": round(dur, 2),
        "speakers": speakers,
        "language_minutes": {
            k: round(sum(s["end"] - s["start"] for s in raw if s["language"] == k) / 60, 1)
            for k in lang_stat
        },
        "speaker_minutes": {k: round(v / 60, 1) for k, v in sorted(spk_dur.items())},
        "turn_count": len(turns),
        "turns": turns,
    }
    with open(os.path.join(OUT_DIR, "segments.json"), "w", encoding="utf-8") as f:
        json.dump(segments, f, ensure_ascii=False, indent=1)
    with open(os.path.join(OUT_DIR, "raw_segments.json"), "w", encoding="utf-8") as f:
        json.dump(raw, f, ensure_ascii=False, indent=1)
    with open(os.path.join(OUT_DIR, "diarization.json"), "w", encoding="utf-8") as f:
        json.dump(diar_turns, f, ensure_ascii=False, indent=1)

    with open(os.path.join(OUT_DIR, "transcript.txt"), "w", encoding="utf-8") as f:
        f.write(f"# Meeting transcript — {os.path.basename(AUDIO_PATH)}\n")
        f.write(f"# Duration: {fmt_ts_dot(dur)} | Speakers: {', '.join(speakers)}\n")
        f.write("# Languages (minutes): "
                + ", ".join(f"{LANG_NAMES.get(k, k)} {v}" for k, v in segments["language_minutes"].items())
                + "\n\n")
        for i, t in enumerate(turns, 1):
            lang = LANG_NAMES.get(t["language"], t["language"])
            f.write(f"[{fmt_ts_dot(t['start'])}] {t['speaker']} ({lang}):\n")
            f.write(f"  {t['text']}\n\n")

    with open(os.path.join(OUT_DIR, "transcript.srt"), "w", encoding="utf-8") as f:
        for i, t in enumerate(turns, 1):
            f.write(f"{i}\n")
            f.write(f"{fmt_ts(t['start'])} --> {fmt_ts(t['end'])}\n")
            f.write(f"{t['speaker']}\n{t['text']}\n\n")

    print(f"\n[out] wrote: segments.json ({len(turns)} turns), raw_segments.json "
          f"({len(raw)} segments), diarization.json, transcript.txt, transcript.srt", flush=True)
    print("[out] languages (min): "
          + ", ".join(f"{LANG_NAMES.get(k, k)}={v}" for k, v in segments["language_minutes"].items()),
          flush=True)
    print("[out] speaker minutes: "
          + ", ".join(f"{k}={v}" for k, v in segments["speaker_minutes"].items()), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--redo-transcribe", action="store_true")
    ap.add_argument("--redo-diarize", action="store_true")
    args = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    setup_cuda_dll_path()
    env = load_env()
    token = env.get("HF_TOKEN")

    print(f"[vad] loading audio {AUDIO_PATH}", flush=True)
    audio, sr = read_wav(AUDIO_PATH)
    dur = len(audio) / sr
    print(f"[vad] audio: {dur / 60:.1f} min @ {sr} Hz", flush=True)
    print("[vad] running silero VAD + chunking ...", flush=True)
    chunks = chunk_speech(audio)
    print(f"[vad] {len(chunks)} speech chunks", flush=True)

    raw_path = os.path.join(OUT_DIR, "raw_segments.json")
    if os.path.exists(raw_path) and not args.redo_transcribe:
        print("[whisper] using existing raw_segments.json", flush=True)
        raw = json.load(open(raw_path, encoding="utf-8"))
    else:
        raw = transcribe_chunks(audio, chunks)
        with open(raw_path, "w", encoding="utf-8") as f:
            json.dump(raw, f, ensure_ascii=False, indent=1)

    diag_path = os.path.join(OUT_DIR, "diarization.json")
    if os.path.exists(diag_path) and not args.redo_diarize:
        print("[diarize] using existing diarization.json", flush=True)
        diar_turns = json.load(open(diag_path, encoding="utf-8"))
    else:
        diar_turns, _ = run_diarization(token, audio, sr)
        with open(diag_path, "w", encoding="utf-8") as f:
            json.dump(diar_turns, f, ensure_ascii=False, indent=1)

    print("[merge] assigning speakers + merging turns ...", flush=True)
    assigned = assign_speakers(raw, diar_turns)
    turns = merge_turns(assigned)
    speakers = sorted({t["speaker"] for t in diar_turns})
    write_outputs(turns, raw, diar_turns, speakers)
    print("[merge] done.", flush=True)


if __name__ == "__main__":
    main()
