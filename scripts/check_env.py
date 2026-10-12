"""Verify environment: faster-whisper, torch CUDA, and pyannote model access."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def load_env():
    env = {}
    with open(os.path.join(os.path.dirname(__file__), "..", ".env"), encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                env[k.strip()] = v.strip()
    return env


def main():
    import faster_whisper
    import torch

    print("faster_whisper:", getattr(faster_whisper, "__version__", "ok"))
    print("torch:", torch.__version__)
    print("cuda available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("gpu:", torch.cuda.get_device_name(0),
              f"{torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

    env = load_env()
    token = env.get("HF_TOKEN") or os.environ.get("HF_TOKEN")
    if not token:
        print("HF_TOKEN: MISSING")
        return

    from huggingface_hub import HfApi
    api = HfApi(token=token)
    try:
        info = api.whoami()
        print("huggingface user:", info.get("name"))
    except Exception as e:
        print("whoami FAILED:", type(e).__name__, e)
        return

    from huggingface_hub import snapshot_download
    from huggingface_hub.errors import GatedRepoError, EntryNotFoundError

    for repo in ["pyannote/segmentation-3.0", "pyannote/speaker-diarization-3.1"]:
        try:
            path = snapshot_download(repo_id=repo, token=token)
            print(f"OK {repo} -> {path}")
        except (GatedRepoError, EntryNotFoundError) as e:
            print(f"GATED {repo}: {str(e)[:300]}")
        except Exception as e:
            print(f"ERR {repo}: {type(e).__name__} {str(e)[:300]}")


if __name__ == "__main__":
    main()
