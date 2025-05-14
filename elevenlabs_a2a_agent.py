#!/usr/bin/env python3

import os
import json
import pathlib
import logging
from typing import List, Optional, Iterator

import uvicorn
from pydantic import BaseModel, Field
from pydantic_ai import A2A, A2ATool, A2AInput, A2AOutput, FileArtifact # Removed Artifact as it's not directly used

# Attempt to import the generate function from the existing script
# This assumes generate_elevenlabs_audio.py is in the same directory or Python path
try:
    import generate_elevenlabs_audio
except ImportError:
    logging.error("Failed to import generate_elevenlabs_audio.py. Make sure it's in the PYTHONPATH.")
    # You might want to exit or raise a more critical error if this is essential at startup
    # For now, we'll let it proceed, but the tool will fail if generate_elevenlabs_audio is not found.
    pass


# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Configuration ---
DEFAULT_OUTPUT_DIR = os.getenv("A2A_OUTPUT_DIR", "out_audio_a2a")
DEFAULT_MODEL_ID = "eleven_multilingual_v2"
DEFAULT_STABILITY = 0.5
DEFAULT_SIMILARITY_BOOST = 0.75
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")

# --- Pydantic Models ---
class TTSConfig(BaseModel):
    text: str = Field(..., description="The text to be converted to speech.")
    voice_id: str = Field(..., description="Identifier of the voice to be used for generation.")
    filename: str = Field(..., description="Desired filename for the output MP3 file (e.g., 'greeting.mp3').")
    model: Optional[str] = Field(DEFAULT_MODEL_ID, description="Optional ElevenLabs model ID.")
    stability: Optional[float] = Field(DEFAULT_STABILITY, description="Optional voice stability setting (0-1).")
    similarity_boost: Optional[float] = Field(DEFAULT_SIMILARITY_BOOST, description="Optional voice similarity boost setting (0-1).")

class GenerateAudioToolInput(A2AInput):
    prompts: List[TTSConfig] = Field(..., description="List of TTS configurations to generate audio for.")
    output_dir: Optional[str] = Field(DEFAULT_OUTPUT_DIR, description="Directory to save generated MP3 files.")
    api_key: Optional[str] = Field(None, description="ElevenLabs API key. If not provided, uses ELEVENLABS_API_KEY env var.")
    debug: Optional[bool] = Field(False, description="Enable debug output for the generation process.")

class GenerateAudioToolOutput(A2AOutput):
    generated_files: List[FileArtifact] = Field(default_factory=list, description="List of generated audio file artifacts.")
    errors: List[str] = Field(default_factory=list, description="List of errors encountered during generation.")

# --- A2A Tool ---
class ElevenLabsTextToSpeechTool(A2ATool[GenerateAudioToolInput, GenerateAudioToolOutput]):
    name: str = "ElevenLabsTextToSpeechGenerator"
    description: str = "Generates speech from text using ElevenLabs API and saves it to MP3 files. Can process multiple prompts in a batch."
    input_model: type[GenerateAudioToolInput] = GenerateAudioToolInput
    output_model: type[GenerateAudioToolOutput] = GenerateAudioToolOutput

    async def execute(self, input_data: GenerateAudioToolInput) -> GenerateAudioToolOutput:
        if not hasattr(generate_elevenlabs_audio, 'generate'):
            logger.error("The 'generate' function from generate_elevenlabs_audio.py is not available.")
            return GenerateAudioToolOutput(errors=["Core generation function is missing."])

        api_key_to_use = input_data.api_key or ELEVENLABS_API_KEY
        if not api_key_to_use:
            logger.error("ElevenLabs API key not provided via input or ELEVENLABS_API_KEY environment variable.")
            return GenerateAudioToolOutput(errors=["ElevenLabs API key is required."])

        output_dir_path = pathlib.Path(input_data.output_dir or DEFAULT_OUTPUT_DIR)
        try:
            output_dir_path.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            logger.error(f"Failed to create output directory {output_dir_path}: {e}")
            return GenerateAudioToolOutput(errors=[f"Failed to create output directory: {e}"])

        generated_files_artifacts: List[FileArtifact] = []
        errors_list: List[str] = []

        logger.info(f"Processing {len(input_data.prompts)} prompts. Output directory: {output_dir_path}")

        for i, prompt_config in enumerate(input_data.prompts):
            tts_cfg = {
                "text": prompt_config.text,
                "voice_id": prompt_config.voice_id,
                "model": prompt_config.model or DEFAULT_MODEL_ID,
                "stability": prompt_config.stability if prompt_config.stability is not None else DEFAULT_STABILITY,
                "similarity_boost": prompt_config.similarity_boost if prompt_config.similarity_boost is not None else DEFAULT_SIMILARITY_BOOST,
            }
            
            # Sanitize filename and ensure .mp3 extension
            base_name_from_prompt = pathlib.Path(prompt_config.filename).name
            base_filename, ext = os.path.splitext(base_name_from_prompt)
            output_filename = base_filename + ".mp3"
            if ext.lower() != ".mp3" and ext != "": # Allow no extension, then we add .mp3
                 logger.warning(f"Provided filename '{prompt_config.filename}' (basename: '{base_name_from_prompt}') for prompt {i} had a non-MP3 extension '{ext}'. Using '{output_filename}'.")
            
            out_path = output_dir_path / output_filename

            if input_data.debug:
                logger.debug(f"Processing prompt {i}: '{output_filename}' to '{out_path}'")
                logger.debug(f"TTS Config for prompt {i}: {json.dumps(tts_cfg)}")

            try:
                # Create zero-byte file first (mimicking original script's behavior for process tracking)
                with open(out_path, "wb") as f:
                    pass 

                audio_stream_iterator: Iterator[bytes] = generate_elevenlabs_audio.generate(
                    tts_cfg=tts_cfg,
                    api_key=api_key_to_use,
                    debug=input_data.debug or False
                )

                bytes_written = 0
                with open(out_path, "wb") as f:
                    for chunk in audio_stream_iterator:
                        if chunk:
                            f.write(chunk)
                            bytes_written += len(chunk)
                
                file_size = out_path.stat().st_size
                if file_size == 0 and bytes_written == 0: # Check if anything was actually written
                    # This can happen if generate() raises an exception before yielding any data,
                    # or if the API returns an empty stream that generate() doesn't flag as an error.
                    error_msg = f"Warning: {output_filename} is zero size after generation attempt. The API might have returned an empty stream or an error occurred early."
                    logger.warning(error_msg)
                    errors_list.append(error_msg)
                    if out_path.exists(): # Clean up empty file
                        os.remove(out_path)
                        logger.info(f"Removed empty file: {out_path}")
                else:
                    logger.info(f"✓ '{output_filename}' generated ({file_size} bytes)")
                    artifact = FileArtifact.from_file(filepath=str(out_path.resolve()), mime_type="audio/mpeg")
                    generated_files_artifacts.append(artifact)

            except Exception as e:
                error_msg = f"Error generating '{output_filename}': {str(e)}"
                logger.error(error_msg, exc_info=input_data.debug) # Log stack trace if debug is on
                errors_list.append(error_msg)
                if out_path.exists() and out_path.stat().st_size == 0:
                    try:
                        os.remove(out_path)
                        logger.info(f"Removed empty file '{out_path}' due to error.")
                    except OSError as oe:
                        logger.error(f"Error removing empty file '{out_path}': {oe}")
        
        return GenerateAudioToolOutput(generated_files=generated_files_artifacts, errors=errors_list)

# --- A2A Agent Setup ---
elevenlabs_tts_tool = ElevenLabsTextToSpeechTool()

agent = A2A(
    label="ElevenLabs TTS Agent",
    description="An agent that uses ElevenLabs to generate speech from text and provides audio files.",
    tools=[elevenlabs_tts_tool],
    # No LLM is configured here, as this agent is purely tool-based via A2A.
)

# Expose the A2A server application for Uvicorn
app = agent.to_a2a()

# --- Main execution for Uvicorn ---
if __name__ == "__main__":
    if not ELEVENLABS_API_KEY:
        logger.warning("ELEVENLABS_API_KEY environment variable is not set. "
                       "The agent will rely on API key provided in requests, or fail if none is provided.")
    
    uvicorn_host = os.getenv("A2A_HOST", "127.0.0.1")
    uvicorn_port = int(os.getenv("A2A_PORT", "8000"))

    logger.info(f"Starting ElevenLabs A2A Agent on http://{uvicorn_host}:{uvicorn_port}")
    logger.info(f"Tool endpoint: /tools/{ElevenLabsTextToSpeechTool.name}/execute")
    
    uvicorn.run(app, host=uvicorn_host, port=uvicorn_port)
