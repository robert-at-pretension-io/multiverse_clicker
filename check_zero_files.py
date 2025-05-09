#!/usr/bin/env python3

import os
import pathlib
import json

# Load the input JSON
json_path = pathlib.Path("conflux_voice_prompts.json")
prompts = json.loads(json_path.read_text())

# Check all files in the output directory
output_dir = pathlib.Path("out_audio")
zero_files = []

for file_path in output_dir.glob("*.mp3"):
    if file_path.stat().st_size == 0:
        zero_files.append(file_path.name)

print(f"Found {len(zero_files)} zero-byte files:")
for filename in zero_files:
    # Find corresponding entry in JSON
    entry = next((p for p in prompts if p["filename"] == filename), None)
    if entry:
        print(f"{filename}: voice_id={entry['voice_id']}, text=\"{entry['text']}\"")
    else:
        print(f"{filename}: Not found in JSON file")

print("\nChecking completed.")
