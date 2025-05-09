#!/usr/bin/env python3

import argparse, json, os, sys, pathlib, time, requests
from tqdm import tqdm
import logging

# Set up logging
logging.basicConfig(level=logging.DEBUG, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('elevenlabs_debug')

API_HOST = "https://api.elevenlabs.io"

def parse_args():
    p = argparse.ArgumentParser(description="Batch‑generate speech with ElevenLabs TTS.")
    p.add_argument("json_file", help="Path to JSON prompt list")
    p.add_argument("--output-dir", default="out_audio", help="Directory for MP3 files")
    p.add_argument("--api-key", default=os.getenv("ELEVENLABS_API_KEY"), help="ElevenLabs API key")
    p.add_argument("--model", default="eleven_multilingual_v2", help="Default model id")
    p.add_argument("--voice-stability", type=float, default=0.5, help="Default stability (0‑1)")
    p.add_argument("--voice-similarity", type=float, default=0.75, help="Default similarity boost (0‑1)")
    return p.parse_args()

def generate(tts_cfg, api_key):
    voice_id = tts_cfg["voice_id"]
    url = f"{API_HOST}/v1/text-to-speech/{voice_id}/stream"
    
    logger.debug(f"Making request to {url}")
    logger.debug(f"Using API key: {api_key[:5]}...{api_key[-4:]}")
    logger.debug(f"Voice ID: {voice_id}")

    payload = {
        "text": tts_cfg["text"],
        "model_id": tts_cfg.get("model", "eleven_multilingual_v2"),
        "voice_settings": {
            "stability": tts_cfg.get("stability", 0.5),
            "similarity_boost": tts_cfg.get("similarity_boost", 0.75),
        },
    }
    logger.debug(f"Payload: {json.dumps(payload)}")
    
    headers = {
        "xi-api-key": api_key,
        "Content-Type": "application/json",
        "Accept": "audio/mpeg",
    }
    logger.debug(f"Headers: {headers}")

    try:
        with requests.post(url, headers=headers, json=payload, stream=True, timeout=120) as r:
            logger.debug(f"Response status: {r.status_code}")
            logger.debug(f"Response headers: {r.headers}")
            
            # Get first portion of response content for debugging
            first_chunk = next(r.iter_content(chunk_size=4096), None)
            if first_chunk:
                logger.debug(f"Got first chunk, size: {len(first_chunk)} bytes")
                
                # Reset the response iterator to include the first chunk
                def iter_content_with_first_chunk():
                    yield first_chunk
                    yield from r.iter_content(chunk_size=4096)
                
                # Return the new iterator
                return iter_content_with_first_chunk()
            else:
                logger.error("No content received in response")
                logger.debug(f"Response text: {r.text}")
                raise Exception(f"Empty response from API: {r.text}")
    except Exception as e:
        logger.error(f"Error during API request: {e}")
        raise

def main():
    args = parse_args()
    if not args.api_key:
        sys.exit("❌ Provide ELEVENLABS_API_KEY env var or --api-key")

    logger.debug(f"API key found: {args.api_key[:5]}...{args.api_key[-4:]}")
    
    prompts = json.loads(pathlib.Path(args.json_file).read_text())
    logger.debug(f"Loaded {len(prompts)} prompts from {args.json_file}")
    
    outdir = pathlib.Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    logger.debug(f"Output directory: {outdir}")

    # Just try with the first prompt for debugging
    entry = prompts[0]
    fname = entry["filename"]
    out_path = outdir / fname
    logger.debug(f"Processing {fname} to {out_path}")
    
    try:
        stream = generate(entry, args.api_key)
        logger.debug("Got stream generator from generate()")
        
        with open(out_path, "wb") as f:
            for i, chunk in enumerate(stream):
                logger.debug(f"Writing chunk {i}, size: {len(chunk)} bytes")
                f.write(chunk)
                
        logger.debug(f"Finished writing {out_path}")
        file_size = out_path.stat().st_size
        logger.debug(f"Final file size: {file_size} bytes")
        
    except Exception as e:
        logger.error(f"Error for {fname}: {e}")

if __name__ == "__main__":
    main()
