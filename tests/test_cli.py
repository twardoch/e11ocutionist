#!/usr/bin/env python3
# this_file: tests/test_cli.py
"""Tests for the CLI module."""

import os
from pathlib import Path
from unittest.mock import MagicMock, patch, call

import pytest
from loguru import logger

from e11ocutionist.cli import (
    chunk,
    convert_11labs,
    entitize,
    neifix,
    orate,
    process,
    say,
    tonedown,
)


@pytest.fixture
def sample_input_file(tmp_path):
    """Create a sample input file for testing."""
    input_file = tmp_path / "input.txt"
    input_file.write_text("Sample text for testing")
    return str(input_file)


@pytest.fixture
def sample_output_file(tmp_path):
    """Create a sample output file path."""
    return str(tmp_path / "output.xml")


@pytest.fixture
def sample_output_dir(tmp_path):
    """Create a sample output directory for testing."""
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    return str(output_dir)


def test_process_invalid_input_file():
    """Test process function with non-existent input file."""
    with pytest.raises(FileNotFoundError, match="Input file not found"):
        process("nonexistent.txt")


def test_process_invalid_start_step(sample_input_file):
    """Test process function with invalid start step."""
    with pytest.raises(ValueError, match="Invalid start_step"):
        process(sample_input_file, start_step="invalid")


@patch("e11ocutionist.cli.E11ocutionistPipeline")
def test_process_basic(mock_pipeline, sample_input_file, sample_output_dir):
    """Test basic process function execution."""
    # Setup mock
    mock_pipeline_instance = MagicMock()
    mock_pipeline.return_value = mock_pipeline_instance

    # Call process function
    result = process(
        sample_input_file,
        sample_output_dir,
        start_step="chunking",
        force_restart=True,
        backup=True,
    )

    # Verify pipeline was created and run
    mock_pipeline.assert_called_once()
    mock_pipeline_instance.run.assert_called_once()
    assert isinstance(result, str)


@patch("e11ocutionist.chunker.process_document")
def test_chunk_command(
    mock_process_document,
    sample_input_file,
    sample_output_file,
):
    """Test chunk command execution."""
    # Call chunk function
    result = chunk(
        sample_input_file,
        sample_output_file,
        model="gpt-4",
        temperature=0.2,
        verbose=True,
        backup=True,
    )

    # Verify process_document was called with correct arguments
    mock_process_document.assert_called_once_with(
        input_file=sample_input_file,
        output_file=sample_output_file,
        model="gpt-4",
        temperature=0.2,
        verbose=True,
        backup=True,
    )
    assert result == sample_output_file


@patch("e11ocutionist.entitizer.process_document")
def test_entitize_command(
    mock_process_document,
    sample_input_file,
    sample_output_file,
):
    """Test entitize command execution."""
    # Call entitize function
    result = entitize(
        sample_input_file,
        sample_output_file,
        model="gpt-4",
        temperature=0.1,
        verbose=True,
        backup=True,
    )

    # Verify process_document was called with correct arguments
    mock_process_document.assert_called_once_with(
        input_file=sample_input_file,
        output_file=sample_output_file,
        model="gpt-4",
        temperature=0.1,
        verbose=True,
        backup=True,
    )
    assert result == sample_output_file


@patch("e11ocutionist.orator.process_document")
def test_orate_command(
    mock_process_document,
    sample_input_file,
    sample_output_file,
):
    """Test orate command execution."""
    # Call orate function
    result = orate(
        sample_input_file,
        sample_output_file,
        model="gpt-4",
        temperature=0.7,
        all_steps=True,
        sentences=True,
        words=True,
        punctuation=True,
        emotions=True,
        verbose=True,
        backup=True,
    )

    # Verify process_document was called with correct arguments
    mock_process_document.assert_called_once_with(
        input_file=sample_input_file,
        output_file=sample_output_file,
        model="gpt-4",
        temperature=0.7,
        all_steps=True,
        sentences=True,
        words=True,
        punctuation=True,
        emotions=True,
        verbose=True,
        steps=["--all_steps"],
    )
    assert result == sample_output_file


@patch("e11ocutionist.tonedown.process_document")
def test_tonedown_command(
    mock_process_document,
    sample_input_file,
    sample_output_file,
):
    """Test tonedown command execution."""
    # Call tonedown function
    result = tonedown(
        sample_input_file,
        sample_output_file,
        model="gpt-4",
        temperature=0.1,
        min_em_distance=5,
        verbose=True,
    )

    # Verify process_document was called with correct arguments
    mock_process_document.assert_called_once_with(
        input_file=sample_input_file,
        output_file=sample_output_file,
        model="gpt-4",
        temperature=0.1,
        min_em_distance=5,
        verbose=True,
    )
    assert result == sample_output_file


@patch("e11ocutionist.elevenlabs_converter.process_document")
def test_convert_11labs_command(
    mock_process_document,
    sample_input_file,
    sample_output_file,
):
    """Test convert_11labs command execution."""
    # Call convert_11labs function
    result = convert_11labs(
        sample_input_file,
        sample_output_file,
        dialog=True,
        plaintext=False,
        verbose=True,
    )

    # Verify process_document was called with correct arguments
    mock_process_document.assert_called_once_with(
        input_file=sample_input_file,
        output_file=sample_output_file,
        dialog=True,
        plaintext=False,
        verbose=True,
    )
    assert result == sample_output_file


@patch("e11ocutionist.elevenlabs_synthesizer.process_document")
def test_say_command(
    mock_process_document,
    sample_input_file,
    sample_output_dir,
):
    """Test say command execution."""
    # Call say function with text
    result = say(
        text="Hello world",
        output_dir=sample_output_dir,
        api_key="test_key",
        model_id="eleven_multilingual_v2",
        output_format="mp3_44100_128",
        verbose=True,
    )

    # Verify process_document was called with correct arguments
    mock_process_document.assert_called_once_with(
        text="Hello world",
        output_dir=sample_output_dir,
        api_key="test_key",
        model_id="eleven_multilingual_v2",
        output_format="mp3_44100_128",
        verbose=True,
    )
    assert result == sample_output_dir


@patch("e11ocutionist.elevenlabs_synthesizer.process_document")
def test_say_command_with_file(
    mock_process_document,
    sample_input_file,
    sample_output_dir,
):
    """Test say command execution with input file."""
    # Call say function with input file
    result = say(
        input_file=sample_input_file,
        output_dir=sample_output_dir,
        api_key="test_key",
        model_id="eleven_multilingual_v2",
        output_format="mp3_44100_128",
        verbose=True,
    )

    # Verify process_document was called with correct arguments
    mock_process_document.assert_called_once_with(
        input_file=sample_input_file,
        output_dir=sample_output_dir,
        api_key="test_key",
        model_id="eleven_multilingual_v2",
        output_format="mp3_44100_128",
        verbose=True,
    )
    assert result == sample_output_dir


@patch("e11ocutionist.neifix.process_document")
def test_neifix_command(
    mock_process_document,
    sample_input_file,
    sample_output_file,
):
    """Test neifix command execution."""
    # Call neifix function
    result = neifix(
        sample_input_file,
        sample_output_file,
        verbose=True,
    )

    # Verify process_document was called with correct arguments
    mock_process_document.assert_called_once_with(
        input_file=sample_input_file,
        output_file=sample_output_file,
    )
    assert result == sample_output_file


# New test cases for error handling and logging


def test_say_command_missing_input():
    """Test say command with no input provided."""
    with pytest.raises(
        ValueError, match="Either 'text' or 'input_file' must be provided"
    ):
        say(output_dir="output")


def test_say_command_missing_api_key(sample_output_dir):
    """Test say command with no API key provided."""
    with pytest.raises(ValueError, match="ElevenLabs API key not provided"):
        say(text="Hello", output_dir=sample_output_dir)


@patch.dict(os.environ, {"ELEVENLABS_API_KEY": "test_key"})
@patch("e11ocutionist.elevenlabs_synthesizer.process_document")
def test_say_command_env_api_key(mock_process_document, sample_output_dir):
    """Test say command using API key from environment."""
    result = say(text="Hello", output_dir=sample_output_dir)
    mock_process_document.assert_called_once()
    assert mock_process_document.call_args[1]["api_key"] == "test_key"
    assert result == sample_output_dir


@patch("loguru.logger.remove")
@patch("loguru.logger.add")
def test_debug_logging_configuration(mock_add, mock_remove):
    """Test debug logging configuration."""
    result = chunk("input.txt", "output.xml", debug=True)
    mock_remove.assert_called_once()
    mock_add.assert_called_once()
    assert mock_add.call_args[0][1]["level"] == "DEBUG"
    assert isinstance(result, str)


@patch("loguru.logger.remove")
@patch("loguru.logger.add")
def test_verbose_logging_configuration(mock_add, mock_remove):
    """Test verbose logging configuration."""
    result = chunk("input.txt", "output.xml", verbose=True)
    mock_remove.assert_called_once()
    mock_add.assert_called_once()
    assert mock_add.call_args[0][1]["level"] == "INFO"
    assert isinstance(result, str)


@patch("e11ocutionist.cli.E11ocutionistPipeline")
def test_process_pipeline_error(mock_pipeline, sample_input_file):
    """Test process function when pipeline raises an error."""
    mock_pipeline_instance = MagicMock()
    mock_pipeline_instance.run.side_effect = RuntimeError("Pipeline error")
    mock_pipeline.return_value = mock_pipeline_instance

    with pytest.raises(RuntimeError, match="Pipeline error"):
        process(sample_input_file)


@patch("e11ocutionist.orator.process_document")
def test_orate_selective_steps(
    mock_process_document, sample_input_file, sample_output_file
):
    """Test orate command with selective steps."""
    result = orate(
        sample_input_file,
        sample_output_file,
        all_steps=False,
        sentences=True,
        words=True,
    )

    mock_process_document.assert_called_once()
    assert "--sentences" in mock_process_document.call_args[1]["steps"]
    assert "--words" in mock_process_document.call_args[1]["steps"]
    assert "--punctuation" not in mock_process_document.call_args[1]["steps"]
    assert "--emotions" not in mock_process_document.call_args[1]["steps"]
    assert result == sample_output_file
