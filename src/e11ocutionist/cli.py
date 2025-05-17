#!/usr/bin/env python3
# this_file: src/e11ocutionist/cli.py
"""CLI module for e11ocutionist."""

import os
from pathlib import Path
import sys

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
from e11ocutionist.e11ocutionist import (
    E11ocutionistPipeline,
    PipelineConfig,
    ProcessingStep,
)

VALID_MODELS = ["gpt-4", "gpt-3.5-turbo"]


def _configure_logging(debug: bool = False, verbose: bool = False) -> None:
    """Configure logging based on debug and verbose flags."""
    logger.remove()  # Remove default handler

    log_format = (
        "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
        "<level>{level: <8}</level> | "
        "<level>{message}</level>"
    )

    kwargs = {
        "sink": sys.stdout,
        "format": log_format,
        "level": "DEBUG" if debug else "INFO" if verbose else "WARNING",
    }
    logger.add(**kwargs)


def _validate_temperature(temperature: float) -> None:
    """Validate temperature value."""
    if not 0 <= temperature <= 1:
        msg = "Temperature must be between 0 and 1"
        raise ValueError(msg)


def _validate_model(model: str) -> None:
    """Validate model name."""
    if model not in VALID_MODELS:
        msg = f"Invalid model name. Must be one of: {', '.join(VALID_MODELS)}"
        raise ValueError(msg)


def _validate_min_distance(min_distance: int) -> None:
    """Validate minimum emphasis distance."""
    if min_distance <= 0:
        msg = "Minimum emphasis distance must be positive"
        raise ValueError(msg)


def _validate_output_dir(output_dir: str | Path) -> None:
    """Validate output directory."""
    output_dir = Path(output_dir)
    if output_dir.exists() and not output_dir.is_dir():
        msg = f"{output_dir} exists and is not a directory"
        raise NotADirectoryError(msg)
    if not output_dir.exists():
        output_dir.mkdir(parents=True, exist_ok=True)


def chunk(
    input_file: str,
    output_file: str,
    model: str = "gpt-4",
    temperature: float = 0.2,
    verbose: bool = False,
    debug: bool = False,
) -> str:
    """Run the chunking step."""
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
            verbose=verbose,
        )
        logger.info(f"Chunking completed: {output_file}")
        return output_file
    except Exception as e:
        logger.error(f"Processing failed: {e}")
        raise


def entitize(
    input_file: str,
    output_file: str,
    model: str = "gpt-4",
    temperature: float = 0.1,
    backup: bool = False,
    verbose: bool = False,
    debug: bool = False,
) -> str:
    """Run the entitizing step."""
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
            backup=backup,
            verbose=verbose,
        )
        logger.info(f"Entitizing completed: {output_file}")
        return output_file
    except Exception as e:
        logger.error(f"Processing failed: {e}")
        raise


def orate(
    input_file: str,
    output_file: str,
    model: str = "gpt-4",
    temperature: float = 0.7,
    all_steps: bool = True,
    sentences: bool = False,
    words: bool = False,
    punctuation: bool = False,
    emotions: bool = False,
    backup: bool = False,
    verbose: bool = False,
    debug: bool = False,
) -> str:
    """Run the orate step."""
    _configure_logging(debug, verbose)
    _validate_model(model)
    _validate_temperature(temperature)

    if not (all_steps or sentences or words or punctuation or emotions):
        msg = "At least one processing step must be selected"
        raise ValueError(msg)

    steps = []
    if all_steps:
        steps.append("--all_steps")
    else:
        if sentences:
            steps.append("--sentences")
        if words:
            steps.append("--words")
        if punctuation:
            steps.append("--punctuation")
        if emotions:
            steps.append("--emotions")

    logger.info("Running orating step:")
    logger.info(f"Input: {input_file}")
    logger.info(f"Output: {output_file}")
    logger.info(f"Steps: {steps}")

    try:
        result = orator.process_document(
            input_file=input_file,
            output_file=output_file,
            model=model,
            temperature=temperature,
            steps=steps,
            backup=backup,
            verbose=verbose,
        )
        logger.info(f"Orating completed: {result}")
        return result
    except Exception as e:
        logger.error(f"Processing failed: {e}")
        raise


def tone_down(
    input_file: str,
    output_file: str,
    model: str = "gpt-4",
    temperature: float = 0.1,
    min_em_distance: int = 5,
    backup: bool = False,
    verbose: bool = False,
    debug: bool = False,
) -> str:
    """Run the tone down step."""
    _configure_logging(debug, verbose)
    _validate_model(model)
    _validate_temperature(temperature)
    _validate_min_distance(min_em_distance)

    logger.info("Running tone down step:")
    logger.info(f"Input: {input_file}")
    logger.info(f"Output: {output_file}")

    try:
        result = tonedown.process_document(
            input_file=input_file,
            output_file=output_file,
            model=model,
            temperature=temperature,
            min_em_distance=min_em_distance,
            verbose=verbose,
        )
        logger.info(f"Tone down completed: {result}")
        return result
    except Exception as e:
        logger.error(f"Processing failed: {e}")
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
        msg = "Cannot enable both dialog and plaintext modes"
        raise ValueError(msg)

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
        logger.error(f"Error: {e!s}")
        raise


def say(
    text: str | None = None,
    input_file: str | None = None,
    output_dir: str = "output",
    api_key: str | None = None,
    model_id: str = "eleven_multilingual_v2",
    output_format: str = "mp3_44100_128",
    verbose: bool = False,
    debug: bool = False,
) -> str:
    """Synthesize text using ElevenLabs voices."""
    _configure_logging(debug, verbose)
    _validate_output_dir(output_dir)

    if text is not None and input_file is not None:
        msg = "Cannot provide both text and input file"
        raise ValueError(msg)

    if text is None and input_file is not None:
        with open(input_file, encoding="utf-8") as f:
            text = f.read()
    elif text is None:
        msg = "Either 'text' or 'input_file' must be provided"
        raise ValueError(msg)

    if api_key is None:
        api_key = os.environ.get("ELEVENLABS_API_KEY")
        if not api_key:
            msg = "ElevenLabs API key not provided"
            raise ValueError(msg)

    try:
        result = synthesizer.synthesize_with_all_voices(
            text=text,
            output_dir=output_dir,
            api_key=api_key,
            model_id=model_id,
            output_format=output_format,
        )
        logger.info(f"Synthesis completed: {result}")
        return result
    except Exception as e:
        logger.error(f"Processing failed: {e}")
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
        msg = "Input and output files must be different"
        raise ValueError(msg)

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
        logger.error(f"Error: {e!s}")
        raise


def process(  # noqa: PLR0913
    input_file: str,
    output_dir: str | None = None,
    start_step: str = "chunking",
    force_restart: bool = False,
    backup: bool = False,
    chunker_model: str = "gpt-4",
    chunker_temperature: float = 0.2,
    entitizer_model: str = "gpt-4",
    entitizer_temperature: float = 0.1,
    orator_model: str = "gpt-4",
    orator_temperature: float = 0.7,
    orator_all_steps: bool = True,
    orator_sentences: bool = False,
    orator_words: bool = False,
    orator_punctuation: bool = False,
    orator_emotions: bool = False,
    tonedown_model: str = "gpt-4",
    tonedown_temperature: float = 0.1,
    min_em_distance: int | None = 5,
    dialog_mode: bool = True,
    plaintext_mode: bool = False,
    verbose: bool = False,
    debug: bool = False,
) -> str:
    """Run the complete e11ocutionist pipeline.

    Args:
        input_file: Path to input file
        output_dir: Path to output directory (optional)
        start_step: Step to start from (chunking, entitizing, orating, toning_down, elevenlabs_conversion)
        force_restart: Whether to force restart from the specified step
        backup: Whether to create backups of files before modifying them
        chunker_model: Model to use for chunking
        chunker_temperature: Temperature for chunking model
        entitizer_model: Model to use for entitizing
        entitizer_temperature: Temperature for entitizing model
        orator_model: Model to use for orating
        orator_temperature: Temperature for orating model
        orator_all_steps: Whether to run all orator steps
        orator_sentences: Whether to run sentence enhancement
        orator_words: Whether to run word enhancement
        orator_punctuation: Whether to run punctuation enhancement
        orator_emotions: Whether to run emotion enhancement
        tonedown_model: Model to use for toning down
        tonedown_temperature: Temperature for toning down model
        min_em_distance: Minimum distance between emphasis marks
        dialog_mode: Whether to process as dialog
        plaintext_mode: Whether to process as plain text
        verbose: Whether to enable verbose logging
        debug: Whether to enable debug logging

    Returns:
        Path to final output file
    """
    _configure_logging(debug, verbose)

    # Validate input file exists
    input_path = Path(input_file)
    if not input_path.exists():
        msg = "Input file not found"
        raise FileNotFoundError(msg)

    # Map step name to enum
    try:
        start_step_enum = ProcessingStep[start_step.upper()]
    except KeyError as e:
        msg = f"Invalid start_step: {start_step}. Must be one of: {', '.join(step.name.lower() for step in ProcessingStep)}"
        raise ValueError(msg) from e

    # Create pipeline config
    config = PipelineConfig(
        input_file=input_path,
        output_dir=Path(output_dir) if output_dir else None,
        start_step=start_step_enum,
        force_restart=force_restart,
        backup=backup,
        chunker_model=chunker_model,
        chunker_temperature=chunker_temperature,
        entitizer_model=entitizer_model,
        entitizer_temperature=entitizer_temperature,
        orator_model=orator_model,
        orator_temperature=orator_temperature,
        orator_all_steps=orator_all_steps,
        orator_sentences=orator_sentences,
        orator_words=orator_words,
        orator_punctuation=orator_punctuation,
        orator_emotions=orator_emotions,
        tonedown_model=tonedown_model,
        tonedown_temperature=tonedown_temperature,
        min_em_distance=min_em_distance,
        dialog_mode=dialog_mode,
        plaintext_mode=plaintext_mode,
        verbose=verbose,
        debug=debug,
    )

    # Run pipeline
    pipeline = E11ocutionistPipeline(config)
    result = pipeline.run()

    # Return path to final output file
    return str(
        result["final_output_file"]
        if "final_output_file" in result
        else config.output_dir
    )
