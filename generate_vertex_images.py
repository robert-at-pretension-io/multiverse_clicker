#!/usr/bin/env python3
"""generate_vertex_images.py

Bulk image generator for Google Vertex AI Imagen.

Requirements:
  pip install google-cloud-aiplatform --upgrade

Before running:
  * Enable the Vertex AI API and Generative AI features in your Google Cloud project.
  * Authenticate with Application Default Credentials (`gcloud auth application-default login`)
  * Set the environment variable GOOGLE_APPLICATION_CREDENTIALS if needed.

Usage:
  python generate_vertex_images.py prompts.json --project YOUR_PROJECT_ID --location us-central1 --model imagen-3.0-generate-002 --output-dir ./out --n 1

The JSON must look like:
[
  { "prompt": "A cat riding a rocket", "filename": "rocket_cat.png" },
  ...
]
"""

import json
import pathlib
import argparse
import sys
import time

import vertexai
from vertexai.preview.vision_models import ImageGenerationModel


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate images from a JSON prompt list using Vertex AI Imagen."
    )
    parser.add_argument("json_file", help="Path to JSON file with prompts")
    parser.add_argument(
        "--project", required=True, help="Google Cloud project ID to bill"
    )
    parser.add_argument(
        "--location",
        default="us-central1",
        help="Vertex AI region (default: us-central1)",
    )
    parser.add_argument(
        "--model",
        default="imagen-3.0-generate-002",
        help="Model name (see docs for available versions)",
    )
    parser.add_argument(
        "--output-dir",
        default="out",
        help="Directory to write generated images (will be created if absent)",
    )
    parser.add_argument(
        "--n",
        type=int,
        default=1,
        help="Number of images per prompt (1-8 depending on model version)",
    )
    parser.add_argument(
        "--aspect_ratio",
        default="1:1",
        help="Aspect ratio, e.g. 1:1, 16:9, etc.",
    )
    parser.add_argument("--language", default="en", help="Prompt language code")
    parser.add_argument(
        "--safety_filter_level",
        default="block_medium_and_above",
        help="Safety filter level (see docs)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    prompt_entries = json.loads(pathlib.Path(args.json_file).read_text())
    out_dir = pathlib.Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    vertexai.init(project=args.project, location=args.location)
    model = ImageGenerationModel.from_pretrained(args.model)

    for entry in prompt_entries:
        prompt = entry.get("prompt")
        filename = entry.get("filename")
        if not prompt or not filename:
            sys.stderr.write(f"Skipping malformed entry: {entry}\n")
            continue

        try:
            sys.stderr.write(f"[+] Generating '{filename}'...\n")
            images = model.generate_images(
                prompt=prompt,
                number_of_images=args.n,
                aspect_ratio=args.aspect_ratio,
                language=args.language,
                safety_filter_level=args.safety_filter_level,
            )
            for i, img in enumerate(images):
                out_path = out_dir / filename
                if args.n > 1:
                    stem = out_path.stem
                    suffix = out_path.suffix or ".png"
                    out_path = out_dir / f"{stem}_{i}{suffix}"
                img.save(location=str(out_path), include_generation_parameters=False)
        except Exception as exc:
            sys.stderr.write(f"[!] Error generating '{filename}': {exc}\n")
            continue

        time.sleep(0.4)  # gentle pacing to avoid quota spikes


if __name__ == "__main__":
    main()
