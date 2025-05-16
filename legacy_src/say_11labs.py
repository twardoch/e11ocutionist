#!/usr/bin/env -S uv run -s
# /// script
# dependencies = ["elevenlabs", "fire", "python-dotenv", "loguru", "tenacity", "rich"]
# ///
# this_file: to11/say_11labs.py

import os
from pathlib import Path
from collections.abc import Iterator

import fire
from dotenv import load_dotenv
from loguru import logger
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)
from rich.progress import (
    Progress,
    TextColumn,
    BarColumn,
    TaskProgressColumn,
    TimeRemainingColumn,
)

from elevenlabs.client import ElevenLabs
from elevenlabs import save, Voice
from elevenlabs.core import ApiError

# Load environment variables from .env file
load_dotenv()


class VoiceSynthesizer:
    """
    A CLI tool to synthesize text using all personal ElevenLabs voices
    and save the output as MP3 files.
    """

    def __init__(self, api_key: str | None = None, verbose: bool = False):
        """
        Initializes the ElevenLabs client.

        Args:
            api_key: ElevenLabs API key. If None, it will try to load from the
                ELEVENLABS_API_KEY environment variable.
            verbose: Enable verbose logging
        """
        # Configure loguru
        if verbose:
            logger.add("say_11labs.log", rotation="10 MB", level="DEBUG")
        else:
            logger.add("say_11labs.log", rotation="10 MB", level="INFO")

        self.api_key = api_key or os.getenv("ELEVENLABS_API_KEY")
        if not self.api_key:
            logger.error(
                "ELEVENLABS_API_KEY not found. Please set it in your .env file "
                "or pass it as an argument."
            )
            msg = "API key is required."
            raise ValueError(msg)

        self.client = ElevenLabs(api_key=self.api_key)
        logger.info("ElevenLabs client initialized.")

    @retry(
        retry=retry_if_exception_type(ApiError),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        reraise=True,
    )
    def _get_my_voices(self) -> list[Voice]:
        """
        Retrieves all voices belonging to the user (cloned, generated, professional).
        Excludes premade and other categories that aren't typically "my" voices.

        Returns:
            List of user's voices
        """
        logger.debug("Fetching user's voices")
        all_voices_response = self.client.voices.get_all()
        my_voices = [
            voice
            for voice in all_voices_response.voices
            if voice.category
            in ["cloned", "generated", "professional"]  # Adjust as needed
        ]
        if not my_voices:
            logger.warning(
                "No personal/cloned/professional voices found for this API key."
            )
        logger.debug(f"Found {len(my_voices)} voices belonging to the user")
        return my_voices

    def _sanitize_filename(self, name: str) -> str:
        """
        Sanitizes a string to be used as part of a filename.
        Replaces spaces with underscores and removes problematic characters.

        Args:
            name: The string to sanitize

        Returns:
            Sanitized string suitable for filenames
        """
        name = name.replace(" ", "_")
        # Remove or replace characters not allowed in filenames on most OS
        name = "".join(c for c in name if c.isalnum() or c in ("_", "-"))
        return name if name else "unnamed_voice"

    @retry(
        retry=retry_if_exception_type(ApiError),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        reraise=True,
    )
    def _synthesize_text(
        self, voice_id: str, text: str, model_id: str, output_format: str
    ) -> Iterator[bytes]:
        """
        Synthesizes text to speech with the given voice and parameters.
        Includes retry logic for API errors.

        Args:
            voice_id: The voice ID to use
            text: The text to synthesize
            model_id: The model ID to use
            output_format: The output format for the audio

        Returns:
            Iterator of audio data bytes
        """
        logger.debug(f"Synthesizing text with voice ID {voice_id}")
        return self.client.text_to_speech.convert(
            voice_id=voice_id,
            text=text,
            model_id=model_id,
            output_format=output_format,  # type: ignore
        )

    def synthesize_all(
        self,
        text: str,
        output_folder: str = "output_audio",
        model_id: str = "eleven_multilingual_v2",
        output_format: str = "mp3_44100_128",
    ):
        """
        Synthesizes the given text using all personal voices and saves them to MP3s.

        Args:
            text: The text to synthesize.
            output_folder: The folder where MP3 files will be saved.
            model_id: The model ID to use for synthesis.
                Common options: "eleven_multilingual_v2", "eleven_turbo_v2_5".
            output_format: The output format for the audio.
        """
        if not text:
            logger.error("Text to synthesize cannot be empty.")
            return

        my_voices = self._get_my_voices()
        if not my_voices:
            logger.info("No voices to process.")
            return

        output_path = Path(output_folder)
        output_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"Output will be saved to: {output_path.resolve()}")

        successful_syntheses = 0
        failed_syntheses = 0

        # Create a progress bar for synthesis
        with Progress(
            TextColumn("[bold blue]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeRemainingColumn(),
        ) as progress:
            task = progress.add_task("Synthesizing voices", total=len(my_voices))

            for voice in my_voices:
                if not voice.voice_id or not voice.name:
                    logger.warning(f"Skipping voice with missing ID or name: {voice}")
                    failed_syntheses += 1
                    progress.update(task, advance=1)
                    continue

                sanitized_voice_name = self._sanitize_filename(voice.name)
                filename = f"{voice.voice_id}--{sanitized_voice_name}.mp3"
                filepath = output_path / filename

                progress.update(task, description=f"Synthesizing with {voice.name}")

                try:
                    audio_data = self._synthesize_text(
                        voice_id=voice.voice_id,
                        text=text,
                        model_id=model_id,
                        output_format=output_format,
                    )

                    # audio_data will be an iterator of bytes
                    full_audio_bytes = b"".join(audio_data)

                    save(full_audio_bytes, str(filepath))
                    logger.info(f"Successfully saved: {filepath}")
                    successful_syntheses += 1
                except ApiError as e:
                    logger.error(
                        f"API Error with voice {voice.name} (ID: {voice.voice_id}): "
                        f"{e.body or e}"
                    )
                    failed_syntheses += 1
                except Exception as e:
                    logger.error(
                        f"Error with voice {voice.name} (ID: {voice.voice_id}): {e}"
                    )
                    failed_syntheses += 1
                finally:
                    progress.update(task, advance=1)

        # Print summary
        logger.info("----- Synthesis Summary -----")
        logger.info(f"Total voices processed: {len(my_voices)}")
        logger.info(f"Successful syntheses: {successful_syntheses}")
        logger.info(f"Failed syntheses: {failed_syntheses}")
        logger.info(f"Output saved to: {output_path.resolve()}")


def say_all(text, output_folder="output_audio"):
    synthesizer = VoiceSynthesizer()
    synthesizer.synthesize_all(text, output_folder)


if __name__ == "__main__":
    fire.Fire(say_all)
