#!/usr/bin/env python3

import argparse, json, os, sys, pathlib, time, requests
from tqdm import tqdm

API_HOST = "https://api.elevenlabs.io"
MAX_RETRIES = 3
RETRY_DELAY = 2  # seconds

def parse_args():
    p = argparse.ArgumentParser(description="Batch‑generate speech with ElevenLabs TTS.")
    p.add_argument("json_file", help="Path to JSON prompt list")
    p.add_argument("--output-dir", default="out_audio", help="Directory for MP3 files")
    p.add_argument("--api-key", default=os.getenv("ELEVENLABS_API_KEY"), help="ElevenLabs API key")
    p.add_argument("--model", default="eleven_multilingual_v2", help="Default model id")
    p.add_argument("--voice-stability", type=float, default=0.5, help="Default stability (0‑1)")
    p.add_argument("--voice-similarity", type=float, default=0.75, help="Default similarity boost (0‑1)")
    p.add_argument("--debug", action="store_true", help="Enable debug output")
    p.add_argument("--retry", action="store_true", help="Retry only zero-byte files")
    return p.parse_args()

def generate(tts_cfg, api_key, debug=False):
    voice_id = tts_cfg["voice_id"]
    url = f"{API_HOST}/v1/text-to-speech/{voice_id}/stream"
    
    if debug:
        print(f"Making request to {url}")
        print(f"Voice ID: {voice_id}")
        print(f"Text: \"{tts_cfg['text']}\"")

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

    for retry in range(MAX_RETRIES):
        try:
            with requests.post(url, headers=headers, json=payload, stream=True, timeout=120) as r:
                if debug:
                    print(f"Response status: {r.status_code}")
                    
                if r.status_code != 200:
                    error_msg = f"API error: {r.status_code} - {r.text}"
                    print(error_msg)
                    if retry < MAX_RETRIES - 1:
                        print(f"Retrying in {RETRY_DELAY} seconds... (attempt {retry+1}/{MAX_RETRIES})")
                        time.sleep(RETRY_DELAY)
                        continue
                    raise Exception(error_msg)
                    
                # Check for empty response
                first_chunk = next(r.iter_content(chunk_size=4096), None)
                if not first_chunk:
                    error_msg = "Empty response from API"
                    print(error_msg)
                    if retry < MAX_RETRIES - 1:
                        print(f"Retrying in {RETRY_DELAY} seconds... (attempt {retry+1}/{MAX_RETRIES})")
                        time.sleep(RETRY_DELAY)
                        continue
                    raise Exception(error_msg)
                    
                # Create an iterator that yields the first chunk followed by the rest
                def iter_content_with_first_chunk():
                    yield first_chunk
                    yield from r.iter_content(chunk_size=4096)
                    
                return iter_content_with_first_chunk()
        except requests.exceptions.RequestException as e:
            if retry < MAX_RETRIES - 1:
                print(f"Network error: {str(e)}")
                print(f"Retrying in {RETRY_DELAY} seconds... (attempt {retry+1}/{MAX_RETRIES})")
                time.sleep(RETRY_DELAY)
            else:
                raise
                
    # If we get here, all retries failed
    raise Exception(f"Failed after {MAX_RETRIES} attempts")

def main():
    args = parse_args()
    if not args.api_key:
        sys.exit("❌ Provide ELEVENLABS_API_KEY env var or --api-key")

    print(f"Using API key: {args.api_key[:5]}...{args.api_key[-4:]}")
    
    prompts = json.loads(pathlib.Path(args.json_file).read_text())
    print(f"Loaded {len(prompts)} prompts from {args.json_file}")
    
    outdir = pathlib.Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    # If retry mode, only process zero-byte files
    if args.retry:
        zero_files = []
        for entry in prompts:
            out_path = outdir / entry["filename"]
            if out_path.exists() and out_path.stat().st_size == 0:
                zero_files.append(entry)
        print(f"Found {len(zero_files)} zero-byte files to retry")
        prompts = zero_files

    for entry in tqdm(prompts, desc="Generating"):
        fname = entry["filename"]
        out_path = outdir / fname
        
        # Skip if not in retry mode and file exists with content
        if not args.retry and out_path.exists() and out_path.stat().st_size > 0:
            tqdm.write(f"Skipping existing file: {fname}")
            continue
            
        if args.debug:
            print(f"\nProcessing {fname} to {out_path}")
            
        try:
            # Get the audio stream
            stream = generate(entry, args.api_key, args.debug)
            
            # Write the stream to the file
            with open(out_path, "wb") as f:
                for chunk in stream:
                    f.write(chunk)
                    
            # Verify file size
            file_size = out_path.stat().st_size
            if file_size == 0:
                tqdm.write(f"Warning: {fname} has zero size after generation")
            else:
                tqdm.write(f"✓ {fname} generated ({file_size} bytes)")
                
        except Exception as e:
            tqdm.write(f"Error for {fname}: {e}")
            # We don't remove the file, so we can see which ones failed

if __name__ == "__main__":
    main()
