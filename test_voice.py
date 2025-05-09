#!/usr/bin/env python3

import requests
import sys
import os

API_HOST = "https://api.elevenlabs.io"
API_KEY = os.getenv("ELEVENLABS_API_KEY")

if len(sys.argv) < 2:
    print("Usage: python test_voice.py <voice_id>")
    sys.exit(1)

voice_id = sys.argv[1]
url = f"{API_HOST}/v1/text-to-speech/{voice_id}/stream"

payload = {
    "text": "This is a test of the voice API.",
    "model_id": "eleven_multilingual_v2",
    "voice_settings": {
        "stability": 0.5,
        "similarity_boost": 0.75,
    },
}

headers = {
    "xi-api-key": API_KEY,
    "Content-Type": "application/json",
    "Accept": "audio/mpeg",
}

print(f"Testing voice ID: {voice_id}")
try:
    with requests.post(url, headers=headers, json=payload, stream=True, timeout=120) as r:
        print(f"Response status: {r.status_code}")
        if r.status_code != 200:
            print(f"Error response: {r.text}")
        else:
            # Check if we got any content
            content = next(r.iter_content(chunk_size=4096), None)
            if content:
                print(f"Success! Received {len(content)} bytes")
            else:
                print("Error: Empty response")
except Exception as e:
    print(f"Error: {str(e)}")
