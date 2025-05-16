#!/usr/bin/env python3
# this_file: src/e11ocutionist/cli.py
"""Command-line interface for e11ocutionist."""

import os
from pathlib import Path

import fire
from loguru import logger
from dotenv import load_dotenv

from .e11ocutionist import E11ocutionistPipeline, PipelineConfig, ProcessingStep


# Load environment variables from .env file if it exists
load_dotenv()


def process(
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
    tonedown_model: str = "gpt-4",
    tonedown_temperature: float = 0.1,
    orator_all_steps: bool = True,
    orator_sentences: bool = False,
    orator_words: bool = False,
    orator_punctuation: bool = False,
    orator_emotions: bool = False,
    min_em_distance: int | None = None,
    dialog: bool = True,
    plaintext: bool = False,
    verbose: bool = False,
    debug: bool = False,
) -> str:
    """Process a text document through the e11ocutionist pipeline.

    Args:
        input_file: Path to the input file
        output_dir: Path to the output directory (default: auto-generated)
        start_step: Which step to start processing from (default: chunking)
        force_restart: Force restart processing from start_step (default: False)
        backup: Create backups during processing (default: False)
        chunker_model: Model for chunking step (default: gpt-4)
        chunker_temperature: Temperature for chunking step (default: 0.2)
        entitizer_model: Model for entitizing step (default: gpt-4)
        entitizer_temperature: Temperature for entitizing step (default: 0.1)
        orator_model: Model for orating step (default: gpt-4)
        orator_temperature: Temperature for orating step (default: 0.7)
        tonedown_model: Model for toning down step (default: gpt-4)
        tonedown_temperature: Temperature for toning down step (default: 0.1)
        orator_all_steps: Run all orator steps (default: True)
        orator_sentences: Run orator sentences step (default: False)
        orator_words: Run orator words step (default: False)
        orator_punctuation: Run orator punctuation step (default: False)
        orator_emotions: Run orator emotions step (default: False)
        min_em_distance: Minimum distance between emphasis tags (default: None)
        dialog: Process dialog in ElevenLabs conversion (default: True)
        plaintext: Treat input as plaintext in ElevenLabs conversion (default: False)
        verbose: Enable verbose logging (default: False)
        debug: Enable debug logging (default: False)

    Returns:
        Path to the output directory
    """
    # Convert input file and output directory to Path objects
    input_path = Path(input_file)
    output_path = Path(output_dir) if output_dir else None

    # Verify input file exists
    if not input_path.exists():
        msg = f"Input file not found: {input_path}"
        raise FileNotFoundError(msg)

    # Map string start_step to ProcessingStep enum
    try:
        start_step_enum = ProcessingStep[start_step.upper()]
    except KeyError:
        valid_steps = [step.name.lower() for step in ProcessingStep]
        msg = (
            f"Invalid start_step: {start_step}. Valid values: {', '.join(valid_steps)}"
        )
        raise ValueError(msg)

    # Create pipeline configuration
    config = PipelineConfig(
        input_file=input_path,
        output_dir=output_path,
        start_step=start_step_enum,
        force_restart=force_restart,
        backup=backup,
        chunker_model=chunker_model,
        chunker_temperature=chunker_temperature,
        entitizer_model=entitizer_model,
        entitizer_temperature=entitizer_temperature,
        orator_model=orator_model,
        orator_temperature=orator_temperature,
        tonedown_model=tonedown_model,
        tonedown_temperature=tonedown_temperature,
        orator_all_steps=orator_all_steps,
        orator_sentences=orator_sentences,
        orator_words=orator_words,
        orator_punctuation=orator_punctuation,
        orator_emotions=orator_emotions,
        min_em_distance=min_em_distance,
        dialog_mode=dialog,
        plaintext_mode=plaintext,
        verbose=verbose,
        debug=debug,
    )

    # Run the pipeline
    pipeline = E11ocutionistPipeline(config)
    pipeline.run()

    return str(config.output_dir)


def chunk(
    input_file: str,
    output_file: str,
    model: str = "gpt-4",
    temperature: float = 0.2,
    verbose: bool = False,
    backup: bool = False,
    debug: bool = False,
) -> str:
    """Run only the chunking step of the pipeline.

    Args:
        input_file: Path to the input file
        output_file: Path to the output file
        model: Model to use (default: gpt-4)
        temperature: Temperature setting (default: 0.2)
        verbose: Enable verbose logging (default: False)
        backup: Create backups during processing (default: False)
        debug: Enable debug logging (default: False)

    Returns:
        Path to the output file
    """
    from .chunker import process_document

    if debug:
        logger.remove()
        logger.add(lambda msg: print(msg), level="DEBUG")
    elif verbose:
        logger.remove()
        logger.add(lambda msg: print(msg), level="INFO")

    logger.info(f"Running chunking step: {input_file} -> {output_file}")

    process_document(
        input_file=input_file,
        output_file=output_file,
        model=model,
        temperature=temperature,
        verbose=verbose,
        backup=backup,
    )

    logger.info(f"Chunking completed: {output_file}")
    return output_file


def entitize(
    input_file: str,
    output_file: str,
    model: str = "gpt-4",
    temperature: float = 0.1,
    verbose: bool = False,
    backup: bool = False,
    debug: bool = False,
) -> str:
    """Run only the entitizing step of the pipeline.

    Args:
        input_file: Path to the input file
        output_file: Path to the output file
        model: Model to use (default: gpt-4)
        temperature: Temperature setting (default: 0.1)
        verbose: Enable verbose logging (default: False)
        backup: Create backups during processing (default: False)
        debug: Enable debug logging (default: False)

    Returns:
        Path to the output file
    """
    from .entitizer import process_document

    if debug:
        logger.remove()
        logger.add(lambda msg: print(msg), level="DEBUG")
    elif verbose:
        logger.remove()
        logger.add(lambda msg: print(msg), level="INFO")

    logger.info(f"Running entitizing step: {input_file} -> {output_file}")

    process_document(
        input_file=input_file,
        output_file=output_file,
        model=model,
        temperature=temperature,
        verbose=verbose,
        backup=backup,
    )

    logger.info(f"Entitizing completed: {output_file}")
    return output_file


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
    verbose: bool = False,
    backup: bool = False,
    debug: bool = False,
) -> str:
    """Run only the orating step of the pipeline.

    Args:
        input_file: Path to the input file
        output_file: Path to the output file
        model: Model to use (default: gpt-4)
        temperature: Temperature setting (default: 0.7)
        all_steps: Run all sub-steps (default: True)
        sentences: Run sentences sub-step (default: False)
        words: Run words sub-step (default: False)
        punctuation: Run punctuation sub-step (default: False)
        emotions: Run emotions sub-step (default: False)
        verbose: Enable verbose logging (default: False)
        backup: Create backups during processing (default: False)
        debug: Enable debug logging (default: False)

    Returns:
        Path to the output file
    """
    from .orator import process_document

    if debug:
        logger.remove()
        logger.add(lambda msg: print(msg), level="DEBUG")
    elif verbose:
        logger.remove()
        logger.add(lambda msg: print(msg), level="INFO")

    logger.info(f"Running orating step: {input_file} -> {output_file}")

    # Determine which steps to run
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

    process_document(
        input_file=input_file,
        output_file=output_file,
        model=model,
        temperature=temperature,
        verbose=verbose,
        backup=backup,
        steps=steps,
    )

    logger.info(f"Orating completed: {output_file}")
    return output_file


def tonedown(
    input_file: str,
    output_file: str,
    model: str = "gpt-4",
    temperature: float = 0.1,
    min_em_distance: int | None = None,
    verbose: bool = False,
    debug: bool = False,
) -> str:
    """Run only the toning down step of the pipeline.

    Args:
        input_file: Path to the input file
        output_file: Path to the output file
        model: Model to use (default: gpt-4)
        temperature: Temperature setting (default: 0.1)
        min_em_distance: Minimum distance between emphasis tags (default: None)
        verbose: Enable verbose logging (default: False)
        debug: Enable debug logging (default: False)

    Returns:
        Path to the output file
    """
    from .tonedown import process_document

    if debug:
        logger.remove()
        logger.add(lambda msg: print(msg), level="DEBUG")
    elif verbose:
        logger.remove()
        logger.add(lambda msg: print(msg), level="INFO")

    logger.info(f"Running toning down step: {input_file} -> {output_file}")

    em_args = {"min_distance": min_em_distance} if min_em_distance else {}

    process_document(
        input_file=input_file,
        output_file=output_file,
        model=model,
        temperature=temperature,
        verbose=verbose,
        **em_args,
    )

    logger.info(f"Toning down completed: {output_file}")
    return output_file


def convert_11labs(
    input_file: str,
    output_file: str,
    dialog: bool = True,
    plaintext: bool = False,
    verbose: bool = False,
    debug: bool = False,
) -> str:
    """Run only the ElevenLabs conversion step of the pipeline.

    Args:
        input_file: Path to the input file
        output_file: Path to the output file
        dialog: Process dialog (default: True)
        plaintext: Treat input as plaintext (default: False)
        verbose: Enable verbose logging (default: False)
        debug: Enable debug logging (default: False)

    Returns:
        Path to the output file
    """
    from .elevenlabs_converter import process_document

    if debug:
        logger.remove()
        logger.add(lambda msg: print(msg), level="DEBUG")
    elif verbose:
        logger.remove()
        logger.add(lambda msg: print(msg), level="INFO")

    logger.info(f"Running ElevenLabs conversion step: {input_file} -> {output_file}")

    process_document(
        input_file=input_file,
        output_file=output_file,
        dialog=dialog,
        plaintext=plaintext,
        verbose=verbose,
    )

    logger.info(f"ElevenLabs conversion completed: {output_file}")
    return output_file


def say(
    text: str | None = None,
    input_file: str | None = None,
    output_dir: str = "output_audio",
    api_key: str | None = None,
    model_id: str = "eleven_multilingual_v2",
    output_format: str = "mp3_44100_128",
    verbose: bool = False,
    debug: bool = False,
) -> str:
    """Synthesize text using ElevenLabs voices.

    Args:
        text: Text to synthesize (exclusive with input_file)
        input_file: File containing text to synthesize (exclusive with text)
        output_dir: Directory to save audio files (default: output_audio)
        api_key: ElevenLabs API key (default: from ELEVENLABS_API_KEY env var)
        model_id: Model ID to use (default: eleven_multilingual_v2)
        output_format: Output format (default: mp3_44100_128)
        verbose: Enable verbose logging (default: False)
        debug: Enable debug logging (default: False)

    Returns:
        Path to the output directory
    """
    from .elevenlabs_synthesizer import synthesize_with_all_voices

    if debug:
        logger.remove()
        logger.add(lambda msg: print(msg), level="DEBUG")
    elif verbose:
        logger.remove()
        logger.add(lambda msg: print(msg), level="INFO")

    # Check if text or input_file is provided
    if text is None and input_file is None:
        msg = "Either 'text' or 'input_file' must be provided"
        raise ValueError(msg)

    # If input_file is provided, read text from it
    if input_file is not None:
        with open(input_file, encoding="utf-8") as f:
            text = f.read()

    # Get API key from environment if not provided
    if api_key is None:
        api_key = os.environ.get("ELEVENLABS_API_KEY")
        if not api_key:
            msg = "ElevenLabs API key not provided and not found in environment"
            raise ValueError(msg)

    # Create output directory if it doesn't exist
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    logger.info("Synthesizing text using ElevenLabs voices")

    synthesize_with_all_voices(
        text=text,
        output_dir=str(output_path),
        api_key=api_key,
        model_id=model_id,
        output_format=output_format,
    )

    logger.info(f"Synthesis completed: {output_dir}")
    return output_dir


def neifix(
    input_file: str,
    output_file: str,
    verbose: bool = False,
    debug: bool = False,
) -> str:
    """Transform NEI tag content in an XML file.

    Args:
        input_file: Path to the input file
        output_file: Path to the output file
        verbose: Enable verbose logging (default: False)
        debug: Enable debug logging (default: False)

    Returns:
        Path to the output file
    """
    from .neifix import transform_nei_content

    if debug:
        logger.remove()
        logger.add(lambda msg: print(msg), level="DEBUG")
    elif verbose:
        logger.remove()
        logger.add(lambda msg: print(msg), level="INFO")

    logger.info(f"Running NEI fix: {input_file} -> {output_file}")

    transform_nei_content(input_file=input_file, output_file=output_file)

    logger.info(f"NEI fix completed: {output_file}")
    return output_file


def main():
    """Main entry point for the e11ocutionist CLI."""
    # Expose the CLI functions using python-fire
    fire.Fire(
        {
            "process": process,
            "chunk": chunk,
            "entitize": entitize,
            "orate": orate,
            "tonedown": tonedown,
            "convert-11labs": convert_11labs,
            "say": say,
            "neifix": neifix,
        }
    )


if __name__ == "__main__":
    main()
