#!/usr/bin/env python3
# this_file: src/e11ocutionist/elevenlabs_synthesizer.py
"""
ElevenLabs synthesizer for e11ocutionist.

This module provides functionality to synthesize text using
personal ElevenLabs voices and save the result as audio files.
"""

import os
import re
from typing import Any
import backoff
from tenacity import retry, stop_after_attempt, wait_exponential
from loguru import logger
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn

try:
    from elevenlabs import Voice, VoiceSettings, generate, set_api_key, voices
    from elevenlabs.api import Voices
except ImportError:
    logger.error(
        "ElevenLabs API not available. Please install it with: pip install elevenlabs"
    )

    # Define placeholders to prevent errors
    class Voice:
        pass

    class VoiceSettings:
        pass

    class Voices:
        pass

    def generate(*args, **kwargs):
        msg = "ElevenLabs API not available"
        raise ImportError(msg)

    def set_api_key(*args, **kwargs):
        pass

    def voices(*args, **kwargs):
        return []


def sanitize_filename(name: str) -> str:
    """
    Sanitize a string to be used as a filename.

    Args:
        name: Name to sanitize

    Returns:
        Sanitized filename
    """
    # Remove characters that are invalid in filenames
    s = re.sub(r'[\\/*?:"<>|]', "", name)
    # Replace spaces with underscores
    s = re.sub(r"\s+", "_", s)
    # Remove any other potentially problematic characters
    s = re.sub(r"[^\w\-.]", "", s)
    # Truncate if too long
    return s[:100]


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def get_personal_voices(api_key: str) -> list[Voice]:
    """
    Get all personal (cloned, generated, or professional) voices from ElevenLabs.

    Args:
        api_key: ElevenLabs API key

    Returns:
        List of Voice objects
    """
    set_api_key(api_key)

    # Get all voices
    all_voices = voices()

    # Filter for personal voices (cloned, generated, professional)
    personal_voices = [
        v
        for v in all_voices
        if any(
            category in v.category
            for category in ["cloned", "generated", "professional"]
        )
    ]

    logger.info(f"Found {len(personal_voices)} personal voices")
    return personal_voices


@backoff.on_exception(
    backoff.expo, (Exception), max_tries=3, jitter=backoff.full_jitter
)
def synthesize_with_voice(
    text: str,
    voice: Voice,
    output_dir: str,
    model_id: str = "eleven_multilingual_v2",
    output_format: str = "mp3_44100_128",
) -> str:
    """
    Synthesize text with a specific voice and save to a file.

    Args:
        text: Text to synthesize
        voice: Voice object to use
        output_dir: Directory to save the audio file
        model_id: ElevenLabs model ID to use
        output_format: Output format to use

    Returns:
        Path to the saved audio file
    """
    # Get API key from environment
    api_key = os.environ.get("ELEVENLABS_API_KEY")
    if not api_key:
        msg = "ElevenLabs API key not found in environment"
        raise ValueError(msg)

    # Set API key
    set_api_key(api_key)

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Sanitize the voice name for filename
    sanitized_name = sanitize_filename(voice.name)
    output_path = os.path.join(output_dir, f"{voice.voice_id}--{sanitized_name}.mp3")

    # Check if file already exists
    if os.path.exists(output_path):
        logger.info(f"File already exists, skipping: {output_path}")
        return output_path

    # Generate audio
    audio = generate(
        text=text, voice=voice, model=model_id, output_format=output_format
    )

    # Save to file
    with open(output_path, "wb") as f:
        f.write(audio)

    logger.info(f"Saved audio to: {output_path}")
    return output_path


def synthesize_with_all_voices(
    text: str,
    output_dir: str = "output_audio",
    api_key: str | None = None,
    model_id: str = "eleven_multilingual_v2",
    output_format: str = "mp3_44100_128",
) -> dict[str, Any]:
    """
    Synthesize text using all personal ElevenLabs voices.

    Args:
        text: Text to synthesize
        output_dir: Directory to save audio files
        api_key: ElevenLabs API key (falls back to environment variable)
        model_id: ElevenLabs model ID to use
        output_format: Output format to use

    Returns:
        Dictionary with processing statistics
    """
    # Get API key from environment if not provided
    if api_key is None:
        api_key = os.environ.get("ELEVENLABS_API_KEY")
        if not api_key:
            msg = "ElevenLabs API key not provided and not found in environment"
            raise ValueError(msg)

    set_api_key(api_key)

    # Get all personal voices
    personal_voices = get_personal_voices(api_key)

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Track results
    results = {
        "success": True,
        "voices_processed": 0,
        "voices_succeeded": 0,
        "voices_failed": 0,
        "output_files": [],
    }

    # Process each voice with a progress bar
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        TimeElapsedColumn(),
    ) as progress:
        task = progress.add_task(
            f"Synthesizing with {len(personal_voices)} voices...",
            total=len(personal_voices),
        )

        for voice in personal_voices:
            try:
                logger.info(f"Synthesizing with voice: {voice.name} ({voice.voice_id})")
                output_path = synthesize_with_voice(
                    text=text,
                    voice=voice,
                    output_dir=output_dir,
                    model_id=model_id,
                    output_format=output_format,
                )
                results["voices_succeeded"] += 1
                results["output_files"].append(output_path)
            except Exception as e:
                logger.error(f"Failed to synthesize with voice {voice.name}: {e}")
                results["voices_failed"] += 1

            results["voices_processed"] += 1
            progress.update(task, advance=1)

    logger.info(
        f"Synthesized with {results['voices_succeeded']} voices (failed: {results['voices_failed']})"
    )

    return results
