#!/usr/bin/env python3
# this_file: tests/test_cli.py
"""Tests for the CLI module."""

import os
from unittest.mock import MagicMock, patch

import pytest

from e11ocutionist.cli import (
    chunk,
    convert_11labs,
    entitize,
    fix_nei,
    orate,
    process,
    say,
    tone_down,
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
    )

    # Verify process_document was called with correct arguments
    mock_process_document.assert_called_once_with(
        input_file=sample_input_file,
        output_file=sample_output_file,
        model="gpt-4",
        temperature=0.2,
        verbose=True,
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
    # Set up mock return value
    mock_process_document.return_value = sample_output_file

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
    # Set up mock return value
    mock_process_document.return_value = sample_output_file

    # Call tonedown function
    result = tone_down(
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


@patch("e11ocutionist.elevenlabs_synthesizer.synthesize_with_all_voices")
def test_say_command(mock_synthesize, sample_output_dir):
    """Test say command execution."""
    # Set up mock return value
    mock_synthesize.return_value = sample_output_dir

    # Call say function with text
    result = say(
        text="Hello world",
        output_dir=sample_output_dir,
        api_key="test_key",
        model_id="eleven_multilingual_v2",
        output_format="mp3_44100_128",
        verbose=True,
    )

    # Verify synthesize_with_all_voices was called with correct arguments
    mock_synthesize.assert_called_once_with(
        text="Hello world",
        output_dir=sample_output_dir,
        api_key="test_key",
        model_id="eleven_multilingual_v2",
        output_format="mp3_44100_128",
    )
    assert result == sample_output_dir


@patch("e11ocutionist.elevenlabs_synthesizer.synthesize_with_all_voices")
def test_say_command_with_file(mock_synthesize, sample_input_file, sample_output_dir):
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

    # Verify synthesize_with_all_voices was called with correct arguments
    mock_synthesize.assert_called_once()
    assert result == sample_output_dir


@patch("e11ocutionist.neifix.transform_nei_content")
def test_neifix_command(mock_fix_nei, sample_input_file, sample_output_file):
    """Test neifix command execution."""
    # Call neifix function
    result = fix_nei(
        sample_input_file,
        sample_output_file,
        verbose=True,
    )

    # Verify fix_nei_content was called with correct arguments
    mock_fix_nei.assert_called_once_with(
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
@patch("e11ocutionist.elevenlabs_synthesizer.synthesize_with_all_voices")
def test_say_command_env_api_key(mock_synthesize, sample_output_dir):
    """Test say command using API key from environment."""
    result = say(text="Hello", output_dir=sample_output_dir)
    mock_synthesize.assert_called_once()
    assert mock_synthesize.call_args[1]["api_key"] == "test_key"
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


@patch("e11ocutionist.cli.E11ocutionistPipeline")
def test_process_with_invalid_output_dir(mock_pipeline, sample_input_file):
    """Test process function with invalid output directory."""
    # Create a file where the output directory should be
    with open("invalid_dir", "w") as f:
        f.write("This is a file")

    with pytest.raises(NotADirectoryError):
        process(sample_input_file, output_dir="invalid_dir")

    # Cleanup
    os.remove("invalid_dir")


@patch("e11ocutionist.chunker.process_document")
def test_chunk_command_with_invalid_model(mock_process_document, sample_input_file):
    """Test chunk command with invalid model name."""
    with pytest.raises(ValueError, match="Invalid model name"):
        chunk(sample_input_file, "output.xml", model="invalid-model")


@patch("e11ocutionist.entitizer.process_document")
def test_entitize_command_with_invalid_temperature(
    mock_process_document, sample_input_file
):
    """Test entitize command with invalid temperature value."""
    with pytest.raises(ValueError, match="Temperature must be between 0 and 1"):
        entitize(sample_input_file, "output.xml", temperature=1.5)


@patch("e11ocutionist.orator.process_document")
def test_orate_command_no_steps_selected(mock_process_document, sample_input_file):
    """Test orate command when no processing steps are selected."""
    with pytest.raises(
        ValueError, match="At least one processing step must be selected"
    ):
        orate(
            sample_input_file,
            "output.xml",
            all_steps=False,
            sentences=False,
            words=False,
            punctuation=False,
            emotions=False,
        )


@patch("e11ocutionist.tonedown.process_document")
def test_tonedown_command_invalid_min_em_distance(
    mock_process_document,
    sample_input_file,
):
    """Test tonedown command with invalid minimum emphasis distance."""
    with pytest.raises(
        ValueError,
        match="Minimum emphasis distance must be positive",
    ):
        tone_down(
            sample_input_file,
            "output.xml",
            min_em_distance=-1,
        )


@patch("e11ocutionist.elevenlabs_converter.process_document")
def test_convert_11labs_command_with_both_modes(
    mock_process_document, sample_input_file
):
    """Test convert_11labs command with both dialog and plaintext modes enabled."""
    with pytest.raises(
        ValueError, match="Cannot enable both dialog and plaintext modes"
    ):
        convert_11labs(sample_input_file, "output.xml", dialog=True, plaintext=True)


@patch("e11ocutionist.elevenlabs_synthesizer.synthesize_with_all_voices")
def test_say_command_with_both_inputs(mock_synthesize, sample_input_file):
    """Test say command with both text and input file provided."""
    with pytest.raises(ValueError, match="Cannot provide both text and input file"):
        say(text="Hello", input_file=sample_input_file, output_dir="output")


@patch("e11ocutionist.neifix.transform_nei_content")
def test_neifix_command_same_input_output(mock_fix_nei, sample_input_file):
    """Test neifix command with same input and output file."""
    with pytest.raises(ValueError, match="Input and output files must be different"):
        fix_nei(sample_input_file, sample_input_file)


@patch("e11ocutionist.cli.E11ocutionistPipeline")
def test_process_with_backup_error(mock_pipeline, sample_input_file):
    """Test process function when backup creation fails."""
    mock_pipeline_instance = MagicMock()
    mock_pipeline_instance.run.side_effect = PermissionError("Backup failed")
    mock_pipeline.return_value = mock_pipeline_instance

    with pytest.raises(PermissionError, match="Backup failed"):
        process(sample_input_file, backup=True)


@patch("e11ocutionist.cli.E11ocutionistPipeline")
def test_process_with_custom_config(mock_pipeline, sample_input_file):
    """Test process function with custom configuration."""
    result = process(
        sample_input_file,
        chunker_model="gpt-3.5-turbo",
        chunker_temperature=0.5,
        entitizer_model="gpt-3.5-turbo",
        entitizer_temperature=0.3,
        orator_model="gpt-3.5-turbo",
        orator_temperature=0.8,
        tonedown_model="gpt-3.5-turbo",
        tonedown_temperature=0.2,
    )

    mock_pipeline.assert_called_once()
    config = mock_pipeline.call_args[0][0]
    assert config.chunker_model == "gpt-3.5-turbo"
    assert config.chunker_temperature == 0.5
    assert config.entitizer_model == "gpt-3.5-turbo"
    assert config.entitizer_temperature == 0.3
    assert config.orator_model == "gpt-3.5-turbo"
    assert config.orator_temperature == 0.8
    assert config.tonedown_model == "gpt-3.5-turbo"
    assert config.tonedown_temperature == 0.2
    assert isinstance(result, str)
