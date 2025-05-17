#!/usr/bin/env python3
# this_file: tests/test_pipeline.py
"""Tests for the e11ocutionist pipeline configuration and integration."""

from pathlib import Path
from unittest.mock import patch

import pytest

from e11ocutionist.e11ocutionist import (
    E11ocutionistPipeline,
    PipelineConfig,
    ProcessingStep,
)


@pytest.fixture
def sample_input_file(tmp_path):
    """Create a sample input file for testing."""
    input_file = tmp_path / "input.txt"
    input_file.write_text("Sample text for testing")
    return input_file


@pytest.fixture
def sample_output_dir(tmp_path):
    """Create a sample output directory for testing."""
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    return output_dir


@pytest.fixture
def basic_config(sample_input_file, sample_output_dir):
    """Create a basic pipeline configuration."""
    return PipelineConfig(
        input_file=sample_input_file,
        output_dir=sample_output_dir,
        start_step=ProcessingStep.CHUNKING,
        force_restart=False,
        backup=True,
    )


def test_pipeline_config_defaults():
    """Test pipeline configuration defaults."""
    config = PipelineConfig(input_file=Path("test.txt"))
    assert config.start_step == ProcessingStep.CHUNKING
    assert config.force_restart is False
    assert config.backup is False
    assert config.chunker_model == "gpt-4"
    assert config.chunker_temperature == 0.2
    assert config.entitizer_model == "gpt-4"
    assert config.entitizer_temperature == 0.1
    assert config.orator_model == "gpt-4"
    assert config.orator_temperature == 0.7
    assert config.tonedown_model == "gpt-4"
    assert config.tonedown_temperature == 0.1
    assert config.orator_all_steps is True
    assert config.orator_sentences is False
    assert config.orator_words is False
    assert config.orator_punctuation is False
    assert config.orator_emotions is False
    assert config.min_em_distance is None
    assert config.dialog_mode is True
    assert config.plaintext_mode is False
    assert config.verbose is False
    assert config.debug is False


def test_pipeline_config_custom():
    """Test pipeline configuration with custom values."""
    config = PipelineConfig(
        input_file=Path("test.txt"),
        output_dir=Path("output"),
        start_step=ProcessingStep.ENTITIZING,
        force_restart=True,
        backup=True,
        chunker_model="gpt-3.5-turbo",
        chunker_temperature=0.5,
        entitizer_model="gpt-3.5-turbo",
        entitizer_temperature=0.3,
        orator_model="gpt-3.5-turbo",
        orator_temperature=0.8,
        tonedown_model="gpt-3.5-turbo",
        tonedown_temperature=0.2,
        orator_all_steps=False,
        orator_sentences=True,
        orator_words=True,
        orator_punctuation=True,
        orator_emotions=True,
        min_em_distance=10,
        dialog_mode=False,
        plaintext_mode=True,
        verbose=True,
        debug=True,
    )

    assert config.start_step == ProcessingStep.ENTITIZING
    assert config.force_restart is True
    assert config.backup is True
    assert config.chunker_model == "gpt-3.5-turbo"
    assert config.chunker_temperature == 0.5
    assert config.entitizer_model == "gpt-3.5-turbo"
    assert config.entitizer_temperature == 0.3
    assert config.orator_model == "gpt-3.5-turbo"
    assert config.orator_temperature == 0.8
    assert config.tonedown_model == "gpt-3.5-turbo"
    assert config.tonedown_temperature == 0.2
    assert config.orator_all_steps is False
    assert config.orator_sentences is True
    assert config.orator_words is True
    assert config.orator_punctuation is True
    assert config.orator_emotions is True
    assert config.min_em_distance == 10
    assert config.dialog_mode is False
    assert config.plaintext_mode is True
    assert config.verbose is True
    assert config.debug is True


def test_pipeline_output_directory_creation(basic_config):
    """Test that pipeline creates output directory if it doesn't exist."""
    E11ocutionistPipeline(basic_config)
    assert basic_config.output_dir.exists()
    assert (basic_config.output_dir / "progress.json").exists()


def test_pipeline_progress_tracking(basic_config):
    """Test pipeline progress tracking."""
    pipeline = E11ocutionistPipeline(basic_config)
    progress_file = basic_config.output_dir / "progress.json"

    # Initial progress should be empty
    with open(progress_file) as f:
        initial_progress = f.read()
        assert initial_progress == ""

    # Mock the chunking process
    with patch("e11ocutionist.chunker.process_document") as mock_chunker:
        mock_chunker.return_value = {"status": "success"}
        pipeline._run_chunking()

    # Progress should now contain chunking step
    with open(progress_file) as f:
        updated_progress = f.read()
        assert "chunking" in updated_progress
        assert '"completed": true' in updated_progress


def test_pipeline_backup_functionality(basic_config):
    """Test pipeline backup functionality."""
    pipeline = E11ocutionistPipeline(basic_config)
    output_file = basic_config.output_dir / "test_output.xml"

    # Create a file to backup
    output_file.write_text("Original content")

    # Create backup
    pipeline._create_backup(output_file)

    # Check backup file exists
    backup_path = output_file.with_suffix(output_file.suffix + ".bak")
    assert backup_path.exists()
    assert backup_path.read_text() == "Original content"


def test_pipeline_step_ordering(basic_config):
    """Test pipeline step execution ordering."""
    pipeline = E11ocutionistPipeline(basic_config)

    # Mock all processing steps
    with (
        patch("e11ocutionist.chunker.process_document") as mock_chunker,
        patch("e11ocutionist.entitizer.process_document") as mock_entitizer,
        patch("e11ocutionist.orator.process_document") as mock_orator,
        patch("e11ocutionist.tonedown.process_document") as mock_tonedown,
        patch("e11ocutionist.elevenlabs_converter.process_document") as mock_converter,
    ):
        # Run pipeline
        pipeline.run()

        # Verify steps were called in order
        mock_chunker.assert_called_once()
        mock_entitizer.assert_called_once()
        mock_orator.assert_called_once()
        mock_tonedown.assert_called_once()
        mock_converter.assert_called_once()


def test_pipeline_step_dependencies(basic_config):
    """Test pipeline step dependencies."""
    pipeline = E11ocutionistPipeline(basic_config)

    # Mock all processing steps
    with (
        patch("e11ocutionist.chunker.process_document") as mock_chunker,
        patch("e11ocutionist.entitizer.process_document") as mock_entitizer,
    ):
        # Make chunking step fail
        mock_chunker.side_effect = RuntimeError("Chunking failed")

        # Run pipeline should fail and not proceed to next step
        with pytest.raises(RuntimeError):
            pipeline.run()

        mock_chunker.assert_called_once()
        mock_entitizer.assert_not_called()


def test_pipeline_resume_from_step(basic_config):
    """Test pipeline resumption from a specific step."""
    # Set start step to entitizing
    basic_config.start_step = ProcessingStep.ENTITIZING

    # Create necessary input files
    input_file = basic_config.output_dir / "input_step4_toneddown.xml"
    input_file.write_text("<root>Test content</root>", encoding="utf-8")

    pipeline = E11ocutionistPipeline(basic_config)

    # Mock all processing steps
    with (
        patch("e11ocutionist.chunker.process_document") as mock_chunker,
        patch("e11ocutionist.entitizer.process_document") as mock_entitizer,
    ):
        # Run pipeline
        pipeline.run()

        # Verify chunker was not called
        mock_chunker.assert_not_called()
        # Verify entitizer was called
        mock_entitizer.assert_called_once()


def test_pipeline_force_restart(basic_config):
    """Test pipeline force restart functionality."""
    # Set force restart and start from entitizing
    basic_config.force_restart = True
    basic_config.start_step = ProcessingStep.ENTITIZING

    pipeline = E11ocutionistPipeline(basic_config)

    # Create necessary input files
    input_file = basic_config.output_dir / "input_step4_toneddown.xml"
    input_file.write_text("<root>Test content</root>", encoding="utf-8")

    # Create fake progress
    progress_file = basic_config.output_dir / "progress.json"
    fake_progress = {
        "chunking": {
            "completed": True,
            "output_file": "test.xml",
        },
        "entitizing": {
            "completed": True,
            "output_file": "test.xml",
        },
    }
    progress_file.write_text(str(fake_progress), encoding="utf-8")

    # Mock all processing steps
    with (
        patch("e11ocutionist.entitizer.process_document") as mock_entitizer,
        patch("e11ocutionist.orator.process_document") as mock_orator,
    ):
        # Run pipeline
        pipeline.run()

        # Verify entitizer was called (force restart)
        mock_entitizer.assert_called_once()
        # Verify orator was called
        mock_orator.assert_called_once()
