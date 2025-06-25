#!/usr/bin/env python3
# this_file: src/e11ocutionist/elevenlabs_synthesizer.py
"""
ElevenLabs synthesizer for e11ocutionist.

This module provides functionality to synthesize text using
personal ElevenLabs voices and save the result as audio files.

Used in:
- e11ocutionist/elevenlabs_synthesizer.py
"""

import os
import re
import backoff
from tenacity import retry, stop_after_attempt, wait_exponential
from loguru import logger
from pathlib import Path
from typing import Any

try:
    from elevenlabs import Voice as ElevenLabsVoice
    from elevenlabs import VoiceSettings, generate, set_api_key, voices
    from elevenlabs.api import Voices
except ImportError:
    logger.error("ElevenLabs API not available. Install with: pip install elevenlabs")

    # Define placeholders to prevent errors
    class ElevenLabsVoice:
        """Placeholder for ElevenLabs Voice class.

        Used in:
        - e11ocutionist/elevenlabs_synthesizer.py
        """

        def __init__(self) -> None:
            """Used in:
            - e11ocutionist/elevenlabs_synthesizer.py
            """
            self.name: str = ""
            self.voice_id: str = ""
            self.category: list[str] = []

    class VoiceSettings:
        """Placeholder for ElevenLabs VoiceSettings class.

        Used in:
        - e11ocutionist/elevenlabs_synthesizer.py
        """

    class Voices:
        """Placeholder for ElevenLabs Voices class.

        Used in:
        - e11ocutionist/elevenlabs_synthesizer.py
        """

    def generate(*args: Any, **kwargs: Any) -> bytes:
        """Placeholder for ElevenLabs generate function.

        Used in:
        - e11ocutionist/elevenlabs_synthesizer.py
        """
        msg = "ElevenLabs API not available"
        raise ImportError(msg)

    def set_api_key(*args: Any, **kwargs: Any) -> None:
        """Placeholder for ElevenLabs set_api_key function.

        Used in:
        - e11ocutionist/elevenlabs_synthesizer.py
        """

    def voices(*args: Any, **kwargs: Any) -> list[ElevenLabsVoice]:
        """Placeholder for ElevenLabs voices function.

        Used in:
        - e11ocutionist/elevenlabs_synthesizer.py
        """
        return []


# Type alias for Voice
Voice = ElevenLabsVoice


def sanitize_filename(name: str) -> str:
    """
    Sanitize a string to be used as a filename.

    Args:
        name: Name to sanitize

    Returns:
        Sanitized filename

    Used in:
    - e11ocutionist/elevenlabs_synthesizer.py
    """
    # Remove characters that are invalid in filenames
    s = re.sub(r'[\\/*?:"<>|]', "", name)
    # Replace spaces with underscores
    s = re.sub(r"\s+", "_", s)
    # Remove any other potentially problematic characters
    s = re.sub(r"[^\w\-.]", "", s)
    # Truncate if too long
    return s[:100]


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10),
)
def get_personal_voices(api_key: str | None) -> list[Voice]:
    """
    Get personal (cloned, generated, or professional) voices.

    Args:
        api_key: ElevenLabs API key

    Returns:
        List of Voice objects

    Raises:
        ValueError: If API key is not provided

    Used in:
    - e11ocutionist/elevenlabs_synthesizer.py
    """
    if not api_key:
        msg = "ElevenLabs API key not provided"
        raise ValueError(msg)

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

    Used in:
    - e11ocutionist/elevenlabs_synthesizer.py
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
    output_path = os.path.join(
        output_dir,
        f"{voice.voice_id}--{sanitized_name}.mp3",
    )

    # Check if file already exists
    if os.path.exists(output_path):
        logger.info(f"File already exists, skipping: {output_path}")
        return output_path

    # Generate audio
    audio = generate(
        text=text,
        voice=voice,
        model=model_id,
        output_format=output_format,
    )

    # Save to file
    with open(output_path, "wb") as f:
        f.write(audio)

    logger.info(f"Saved audio to: {output_path}")
    return output_path


def synthesize_with_all_voices(
    text: str,
    output_dir: str | Path,
    api_key: str | None = None,
    model_id: str = "eleven_multilingual_v2",
    output_format: str = "mp3_44100_128",
    verbose: bool = False,
) -> str:
    """Synthesize text with all available voices.

    Used in:
    - e11ocutionist/__init__.py
    - e11ocutionist/cli.py
    - e11ocutionist/elevenlabs_synthesizer.py
    """
    if not api_key and not os.environ.get("ELEVENLABS_API_KEY"):
        msg = "ElevenLabs API key not provided"
        raise ValueError(msg)

    set_api_key(api_key)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    voices = get_personal_voices(api_key)
    if not voices:
        msg = "No personal voices found"
        raise ValueError(msg)

    output_files = []
    for voice in voices:
        try:
            output_file = synthesize_with_voice(
                text=text,
                voice=voice,
                output_dir=str(output_dir),
                model_id=model_id,
                output_format=output_format,
            )
            output_files.append(output_file)
        except Exception as e:
            if verbose:
                logger.warning(f"Failed to synthesize with voice {voice.name}: {e}")

    return str(output_dir)
