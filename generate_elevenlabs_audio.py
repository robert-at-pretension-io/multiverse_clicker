
#!/usr/bin/env python3
"""
generate_elevenlabs_audio.py
---------------------------------
Bulk audio generator for ElevenLabs Text‑to‑Speech.

Prerequisites
-------------
pip install requests tqdm

Environment
-----------
• ELEVENLABS_API_KEY – your ElevenLabs secret key
  (or pass --api-key CLI arg)

Usage
-----
  python generate_elevenlabs_audio.py prompts.json --output-dir ./audio

The JSON must be an array of objects:
  {
    "filename": "gardeners_intro.mp3",
    "voice_id": "21m00Tcm4TlvDq8ikWAM",   # or your custom voice ID
    "text": "Welcome to the Probabilistic Grove..."
  }

Optional additional keys:
  "model": "eleven_monolingual_v1",
  "stability": 0.3,
  "similarity_boost": 0.7
"""

import argparse, json, os, sys, pathlib, time, requests
from tqdm import tqdm
from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception

API_HOST = "https://api.elevenlabs.io"

# Custom retry condition for HTTP 429 errors
def is_http_429_error(retry_state):
    exception = retry_state.outcome.exception()
    return isinstance(exception, requests.exceptions.HTTPError) and exception.response.status_code == 429

def parse_args():
    p = argparse.ArgumentParser(description="Batch‑generate speech with ElevenLabs TTS.")
    p.add_argument("json_file", help="Path to JSON prompt list")
    p.add_argument("--output-dir", default="out_audio", help="Directory for MP3 files")
    p.add_argument("--api-key", default=os.getenv("ELEVENLABS_API_KEY"), help="ElevenLabs API key")
    p.add_argument("--model", default="eleven_multilingual_v2", help="Default model id")
    p.add_argument("--voice-stability", type=float, default=0.5, help="Default stability (0‑1)")
    p.add_argument("--voice-similarity", type=float, default=0.75, help="Default similarity boost (0‑1)")
    return p.parse_args()

@retry(
    wait=wait_exponential(multiplier=1, min=4, max=60),  # Exponential backoff: 1s, 2s, 4s, ... up to 60s
    stop=stop_after_attempt(5),  # Stop after 5 attempts
    retry=is_http_429_error,  # Retry only on HTTP 429 errors
    reraise=True,  # Reraise the last exception if all retries fail
)
def generate(tts_cfg, api_key):
    voice_id = tts_cfg["voice_id"]
    url = f"{API_HOST}/v1/text-to-speech/{voice_id}/stream"

    payload = {
        "text": tts_cfg["text"],
        "model_id": tts_cfg.get("model", "eleven_multilingual_v2"),
        "voice_settings": {
            "stability": tts_cfg.get("stability", 0.5),
            "similarity_boost": tts_cfg.get("similarity_boost", 0.75),
        },
    }
    headers = {
        "xi-api-key": api_key,
        "Content-Type": "application/json",
        "Accept": "audio/mpeg",
    }

    with requests.post(url, headers=headers, json=payload, stream=True, timeout=120) as r:
        r.raise_for_status()
        return r.iter_content(chunk_size=4096)

def main():
    args = parse_args()
    if not args.api_key:
        sys.exit("❌ Provide ELEVENLABS_API_KEY env var or --api-key")

    prompts = json.loads(pathlib.Path(args.json_file).read_text())
    outdir = pathlib.Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    for entry in tqdm(prompts, desc="Generating"):
        fname = entry["filename"]
        out_path = outdir / fname

        if out_path.exists():
            tqdm.write(f"Skipping {fname}, already exists.")
            continue

        try:
            stream = generate(entry, args.api_key)
            with open(out_path, "wb") as f:
                for chunk in stream:
                    f.write(chunk)
        except Exception as e:
            tqdm.write(f"Error for {fname}: {e}")

if __name__ == "__main__":
    main()
