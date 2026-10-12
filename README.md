# Meeting Video Summary

Local GPU pipeline that turns a recorded meeting video into a **speaker-labeled, multilingual transcript** (Uzbek / Russian / English), plus a generator for formal Q&A reports (DOCX).

Built around a 56-minute vendor meeting recording (Cloudflare / DNS, 2026-07-24). Everything runs offline on a local CUDA GPU — the audio never leaves the machine.

## Pipeline

```
audio/meeting_16k.wav
  -> silero VAD chunking
  -> faster-whisper large-v3 transcription (per chunk, auto language detection)
  -> pyannote/speaker-diarization-3.1 (speaker timeline)
  -> speaker assignment (max overlap) + turn merging
  -> output/   (JSON + TXT + SRT)
```

## Outputs (`output/`)

| File | Contents |
|---|---|
| `segments.json` | Final speaker-assigned turns (start/end, speaker, language, text) plus per-language and per-speaker stats — the base for summaries and Q&A |
| `raw_segments.json` | Raw Whisper segments with global timestamps and detected language |
| `diarization.json` | Raw pyannote speaker timeline |
| `transcript.txt` | Readable ordered transcript with `[HH:MM:SS] speaker (language):` headers |
| `transcript.srt` | SRT subtitles with speaker labels |

## Prerequisites

- Python 3.14 (tested on Windows)
- CUDA GPU (developed/tested on RTX 4080 16 GB)
- A Hugging Face token with access to the gated model `pyannote/speaker-diarization-3.1`
- 16-bit mono 16 kHz WAV audio at `audio/meeting_16k.wav`, e.g. extracted from the video with:

  ```
  ffmpeg -i in.mp4 -ar 16000 -ac 1 -acodec pcm_s16le audio/meeting_16k.wav
  ```

## Setup

```
python -m venv .venv
.venv\Scripts\activate
pip install faster-whisper pyannote.audio torch torchaudio python-docx
```

Create `.env` in the project root:

```
HF_TOKEN=hf_xxx
```

Verify the environment:

```
python scripts/check_env.py      # faster-whisper, torch CUDA, HF model access
python scripts/test_clip.py      # transcribe a 60 s slice as a GPU smoke test
```

## Usage

```
python scripts/run_pipeline.py                 # full run (reuses cached raw_segments.json / diarization.json if present)
python scripts/run_pipeline.py --redo-transcribe
python scripts/run_pipeline.py --redo-diarize
```

Observed performance on an RTX 4080: large-v3 transcription at ~22x realtime, diarization ~35 s for 56 min of audio.

**Uzbek mislabeling fix.** Whisper's language detector confuses Uzbek with other Turkic languages (kk/az/tr). After (re)generating raw segments, run:

```
python scripts/fix_uzbek.py   # re-transcribes non-ru/en chunks with language="uz"
```

## Report generation

The formal Q&A report (`reports/meeting_qa_hisobot.docx`, Times New Roman 12, justified) is built from data in `scripts/report_content.py`:

```
python scripts/make_report.py
```

Edit the question/quote data in `report_content.py`; do not edit the DOCX by hand.

## Scripts

| Script | Purpose |
|---|---|
| `run_pipeline.py` | Main pipeline: VAD -> Whisper -> diarization -> merged turns + outputs |
| `fix_uzbek.py` | Re-transcribe Turkic-confused chunks as Uzbek |
| `make_report.py` | Build the Q&A hisobot DOCX |
| `report_content.py` | Report data: meta, participants, questions, verbatim quotes, methodology |
| `check_env.py` | Environment sanity check (faster-whisper, CUDA, HF access) |
| `test_clip.py` | Transcribe a 60 s slice as a GPU smoke test |
| `sample.py` | Print a few segments at fixed minute marks for a quick content scan |

`main.py` is the PyCharm sample stub — the real entry points live in `scripts/`.

## Notes

- The pipeline hardcodes the audio path and output dir (see `AUDIO_PATH` / `OUT_DIR` in `run_pipeline.py`).
- The WAV is loaded with the stdlib `wave` module, and the waveform tensor is passed to pyannote directly to bypass pyannote 4.x's torchcodec file loading.
- CUDA DLL paths from the pip `nvidia-*` packages are prepended to `PATH` at runtime for ctranslate2.
