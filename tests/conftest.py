#!/usr/bin/env python3
"""Common test fixtures for e11ocutionist tests."""

from pathlib import Path

import pytest
from loguru import logger

# Disable logging during tests unless explicitly enabled
logger.remove()
logger.add(lambda _: None)


@pytest.fixture
def test_data_dir() -> Path:
    """Return the path to the test data directory."""
    return Path(__file__).parent / "fixtures"


@pytest.fixture
def temp_workspace(tmp_path) -> Path:
    """Create a temporary workspace with input and output directories."""
    workspace = tmp_path / "workspace"
    (workspace / "input").mkdir(parents=True)
    (workspace / "output").mkdir(parents=True)
    return workspace


@pytest.fixture
def sample_xml() -> str:
    """Return a sample XML string for testing."""
    return """<?xml version="1.0" encoding="UTF-8"?>
<document>
    <metadata>
        <title>Test Document</title>
        <author>Test Author</author>
    </metadata>
    <content>
        <paragraph>This is a test paragraph with some text.</paragraph>
        <dialog speaker="John">
            "Hello," said John, looking at the
            <nei type="person">Named Entity</nei>.
        </dialog>
        <paragraph>
            Another paragraph with
            <nei type="person" new="true">New Entity</nei>.
        </paragraph>
    </content>
</document>"""


@pytest.fixture
def complex_dialog_xml() -> str:
    """Return a complex dialog XML string for testing."""
    return """<?xml version="1.0" encoding="UTF-8"?>
<document>
    <content>
        <dialog speaker="Alice">
            "Hi there!" Alice waved enthusiastically.
        </dialog>
        <dialog speaker="Bob">
            "Hello," Bob replied, "how are you today?"
        </dialog>
        <dialog>
            "Fine, thanks," she said. "And you?"
        </dialog>
        <paragraph>There was a brief pause.</paragraph>
        <dialog speaker="Bob">
            "I'm doing great," he said with a smile.
        </dialog>
    </content>
</document>"""


@pytest.fixture
def mock_config() -> dict:
    """Return a mock configuration dictionary."""
    return {
        "api_key": "test_key",
        "model": "test_model",
        "start_step": None,
        "force_restart": False,
        "verbose": True,
        "backup": True,
    }
