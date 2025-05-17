#!/usr/bin/env python3
# this_file: src/e11ocutionist/cli.py
"""CLI module for e11ocutionist."""

import os
from pathlib import Path
from typing import Optional

from loguru import logger

from e11ocutionist import (
    chunker,
    elevenlabs_converter as converter,
    elevenlabs_synthesizer as synthesizer,
    entitizer,
    neifix,
    orator,
    tonedown,
)

VALID_MODELS = ["gpt-4", "gpt-3.5-turbo"]


def _configure_logging(debug: bool = False, verbose: bool = False) -> None:
    """Configure logging based on debug and verbose flags."""
    logger.remove()  # Remove default handler
    level = "DEBUG" if debug else "INFO" if verbose else "WARNING"
    logger.add(sink=lambda msg: print(msg), level=level)


def _validate_temperature(temperature: float) -> None:
    """Validate temperature value."""
    if not 0 <= temperature <= 1:
        raise ValueError("Temperature must be between 0 and 1")


def _validate_model(model: str) -> None:
    """Validate model name."""
    if model not in VALID_MODELS:
        valid_models = ", ".join(VALID_MODELS)
        msg = f"Invalid model name. Must be one of: {valid_models}"
        raise ValueError(msg)


def _validate_min_distance(min_distance: int) -> None:
    """Validate minimum emphasis distance."""
    if min_distance <= 0:
        raise ValueError("Minimum emphasis distance must be positive")


def _validate_output_dir(output_dir: str) -> None:
    """Validate output directory."""
    path = Path(output_dir)
    if path.exists() and not path.is_dir():
        msg = f"{output_dir} exists and is not a directory"
        raise NotADirectoryError(msg)


def chunk(
    input_file: str,
    output_file: str,
    model: str = "gpt-4",
    temperature: float = 0.2,
    chunk_size: int = 12288,
    verbose: bool = False,
    debug: bool = False,
) -> str:
    """Chunk text into semantic units."""
    _configure_logging(debug, verbose)
    _validate_model(model)
    _validate_temperature(temperature)

    logger.info("Running chunking step:")
    logger.info(f"Input: {input_file}")
    logger.info(f"Output: {output_file}")

    try:
        chunker.process_document(
            input_file=input_file,
            output_file=output_file,
            model=model,
            temperature=temperature,
            chunk_size=chunk_size,
            verbose=verbose,
        )
        logger.info(f"Chunking completed: {output_file}")
        return output_file
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        raise


def entitize(
    input_file: str,
    output_file: str,
    model: str = "gpt-4",
    temperature: float = 0.2,
    verbose: bool = False,
    debug: bool = False,
) -> str:
    """Process named entities in text."""
    _configure_logging(debug, verbose)
    _validate_model(model)
    _validate_temperature(temperature)

    logger.info("Running entitizing step:")
    logger.info(f"Input: {input_file}")
    logger.info(f"Output: {output_file}")

    try:
        entitizer.process_document(
            input_file=input_file,
            output_file=output_file,
            model=model,
            temperature=temperature,
            verbose=verbose,
        )
        logger.info(f"Entitizing completed: {output_file}")
        return output_file
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        raise


def orate(
    input_file: str,
    output_file: str,
    model: str = "gpt-4",
    temperature: float = 0.7,
    all_steps: bool = False,
    sentences: bool = False,
    words: bool = False,
    punctuation: bool = False,
    emotions: bool = False,
    verbose: bool = False,
    debug: bool = False,
) -> str:
    """Process text for speech synthesis."""
    _configure_logging(debug, verbose)
    _validate_model(model)
    _validate_temperature(temperature)

    if not any([all_steps, sentences, words, punctuation, emotions]):
        raise ValueError("At least one processing step must be selected")

    logger.info("Running orating step:")
    logger.info(f"Input: {input_file}")
    logger.info(f"Output: {output_file}")

    steps = []
    if all_steps:
        steps.append("--all_steps")
    if sentences:
        steps.append("--sentences")
    if words:
        steps.append("--words")
    if punctuation:
        steps.append("--punctuation")
    if emotions:
        steps.append("--emotions")

    try:
        orator.process_document(
            input_file=input_file,
            output_file=output_file,
            model=model,
            temperature=temperature,
            steps=steps,
            verbose=verbose,
        )
        logger.info(f"Orating completed: {output_file}")
        return output_file
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        raise


def tone_down(
    input_file: str,
    output_file: str,
    model: str = "gpt-4",
    temperature: float = 0.1,
    min_em_distance: int = 5,
    verbose: bool = False,
    debug: bool = False,
) -> str:
    """Tone down emphasis in text."""
    _configure_logging(debug, verbose)
    _validate_model(model)
    _validate_temperature(temperature)
    _validate_min_distance(min_em_distance)

    logger.info("Running toning down step:")
    logger.info(f"Input: {input_file}")
    logger.info(f"Output: {output_file}")

    try:
        tonedown.process_document(
            input_file=input_file,
            output_file=output_file,
            model=model,
            temperature=temperature,
            min_emphasis_distance=min_em_distance,
            verbose=verbose,
        )
        logger.info(f"Toning down completed: {output_file}")
        return output_file
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        raise


def convert_11labs(
    input_file: str,
    output_file: str,
    dialog: bool = False,
    plaintext: bool = False,
    verbose: bool = False,
    debug: bool = False,
) -> str:
    """Convert text to ElevenLabs format."""
    _configure_logging(debug, verbose)

    if dialog and plaintext:
        raise ValueError("Cannot enable both dialog and plaintext modes")

    logger.info("Running ElevenLabs conversion step:")
    logger.info(f"Input: {input_file}")
    logger.info(f"Output: {output_file}")

    try:
        converter.process_document(
            input_file=input_file,
            output_file=output_file,
            dialog=dialog,
            plaintext=plaintext,
            verbose=verbose,
        )
        logger.info(f"ElevenLabs conversion completed: {output_file}")
        return output_file
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        raise


def say(
    text: Optional[str] = None,
    input_file: Optional[str] = None,
    output_dir: str = "output",
    api_key: Optional[str] = None,
    model_id: str = "eleven_multilingual_v2",
    output_format: str = "mp3_44100_128",
    verbose: bool = False,
    debug: bool = False,
) -> str:
    """Synthesize text using ElevenLabs voices."""
    _configure_logging(debug, verbose)
    _validate_output_dir(output_dir)

    if text is not None and input_file is not None:
        raise ValueError("Cannot provide both text and input file")

    if text is None and input_file is not None:
        with open(input_file, encoding="utf-8") as f:
            text = f.read()
    elif text is None:
        raise ValueError("Either text or input_file must be provided")

    if api_key is None:
        api_key = os.environ.get("ELEVENLABS_API_KEY")
        if not api_key:
            raise ValueError("ElevenLabs API key not provided")

    logger.info("Running speech synthesis:")
    logger.info(f"Output directory: {output_dir}")

    try:
        synthesizer.synthesize_with_all_voices(
            text=text,  # type: ignore
            output_dir=output_dir,
            api_key=api_key,
            model_id=model_id,
            output_format=output_format,
        )
        logger.info(f"Synthesis completed: {output_dir}")
        return output_dir
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        raise


def fix_nei(
    input_file: str,
    output_file: str,
    verbose: bool = False,
    debug: bool = False,
) -> str:
    """Fix named entity issues in text."""
    _configure_logging(debug, verbose)

    if input_file == output_file:
        raise ValueError("Input and output files must be different")

    logger.info("Running NEI fixing step:")
    logger.info(f"Input: {input_file}")
    logger.info(f"Output: {output_file}")

    try:
        neifix.transform_nei_content(
            input_file=input_file,
            output_file=output_file,
        )
        logger.info(f"NEI fixing completed: {output_file}")
        return output_file
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        raise
