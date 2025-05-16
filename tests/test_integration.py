"""Test suite for integration between e11ocutionist modules."""

import os
import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path

from e11ocutionist.e11ocutionist import (
    process_to_chunks,
    process_chunks,
    process_to_text,
    process_to_speech,
)


@pytest.fixture
def sample_xml_content():
    """Sample XML content for testing."""
    return """<?xml version="1.0" encoding="UTF-8"?>
<doc>
    <item id="000000-123456">This is the first paragraph.</item>
    <item id="123456-234567">This is the second paragraph.</item>
    <item id="234567-345678">This is the third paragraph.</item>
</doc>
"""


@pytest.fixture
def sample_chunked_xml_content():
    """Sample chunked XML content for testing."""
    return """<?xml version="1.0" encoding="UTF-8"?>
<chunks>
    <chunk id="chunk-001">
        <unit type="normal">
            <item id="000000-123456">This is the first paragraph.</item>
            <item id="123456-234567">This is the second paragraph.</item>
        </unit>
    </chunk>
    <chunk id="chunk-002">
        <unit type="normal">
            <item id="234567-345678">This is the third paragraph.</item>
        </unit>
    </chunk>
</chunks>
"""


@pytest.fixture
def sample_processed_xml_content():
    """Sample processed XML content for testing."""
    return """<?xml version="1.0" encoding="UTF-8"?>
<chunks>
    <chunk id="chunk-001">
        <unit type="normal">
            <item id="000000-123456" cues="emph(first),pause(.)">
                This is the first paragraph.
            </item>
            <item id="123456-234567" cues="emph(second),pause(.)">
                This is the second paragraph.
            </item>
        </unit>
    </chunk>
    <chunk id="chunk-002">
        <unit type="normal">
            <item id="234567-345678" cues="emph(third),pause(.)">
                This is the third paragraph.
            </item>
        </unit>
    </chunk>
</chunks>
"""


@pytest.fixture
def tmp_files(tmp_path):
    """Create temporary files for testing."""
    input_file = tmp_path / "input.xml"
    chunked_file = tmp_path / "chunked.xml"
    processed_file = tmp_path / "processed.xml"
    text_file = tmp_path / "output.txt"
    audio_file = tmp_path / "output.mp3"

    return {
        "input_file": input_file,
        "chunked_file": chunked_file,
        "processed_file": processed_file,
        "text_file": text_file,
        "audio_file": audio_file,
    }


@patch("e11ocutionist.chunker.semantic_analysis")
def test_process_to_chunks(mock_semantic_analysis, sample_xml_content, tmp_files):
    """Test processing a document to chunks."""
    # Set up mock for semantic analysis
    mock_semantic_analysis.return_value = [
        ("This is the first paragraph.", "normal"),
        ("This is the second paragraph.", "normal"),
        ("This is the third paragraph.", "normal"),
    ]

    # Create the input file
    input_file = tmp_files["input_file"]
    input_file.write_text(sample_xml_content)

    # Run the chunking process
    result = process_to_chunks(
        str(input_file),
        str(tmp_files["chunked_file"]),
        chunk_size=1000,
        model="mock-model",
        temperature=0.1,
        verbose=True,
    )

    # Check result has the expected structure
    assert isinstance(result, dict)
    assert "output_file" in result
    assert "stats" in result

    # Check the output file was created
    assert os.path.exists(tmp_files["chunked_file"])

    # Check content contains chunk tags
    output_content = Path(tmp_files["chunked_file"]).read_text()
    assert "<chunks>" in output_content
    assert "<chunk id=" in output_content
    assert "</chunk>" in output_content
    assert "</chunks>" in output_content


@patch("e11ocutionist.orator.run_llm_processing")
def test_process_chunks(mock_llm_processing, sample_chunked_xml_content, tmp_files):
    """Test processing chunks with LLM."""
    # Set up mock for LLM processing
    mock_llm_processing.return_value = {
        "processed_text": (
            'This is the first paragraph with "emph(first)" and "pause(.)" cues.'
        ),
        "tokens_used": 100,
    }

    # Create the input file
    chunked_file = tmp_files["chunked_file"]
    chunked_file.write_text(sample_chunked_xml_content)

    # Run the processing
    result = process_chunks(
        str(chunked_file),
        str(tmp_files["processed_file"]),
        model="mock-model",
        temperature=0.1,
        verbose=True,
    )

    # Check result has the expected structure
    assert isinstance(result, dict)
    assert "output_file" in result
    assert "stats" in result

    # Check the output file was created
    assert os.path.exists(tmp_files["processed_file"])


@patch("e11ocutionist.elevenlabs_converter.process_document")
def test_process_to_text(
    mock_process_document, sample_processed_xml_content, tmp_files
):
    """Test converting processed XML to text for synthesis."""
    # Set up mock for document processing
    mock_process_document.return_value = "Converted text for synthesis"

    # Create the input file
    processed_file = tmp_files["processed_file"]
    processed_file.write_text(sample_processed_xml_content)

    # Run the text conversion
    result = process_to_text(
        str(processed_file),
        str(tmp_files["text_file"]),
    )

    # Check result has the expected structure
    assert isinstance(result, dict)
    assert "output_file" in result

    # Check the output file was created
    assert os.path.exists(tmp_files["text_file"])

    # Verify content
    output_content = Path(tmp_files["text_file"]).read_text()
    assert output_content == "Converted text for synthesis"


@patch("e11ocutionist.elevenlabs_synthesizer.synthesize_with_all_voices")
@patch("e11ocutionist.elevenlabs_synthesizer.get_personal_voices")
def test_process_to_speech(mock_get_voices, mock_synthesize, tmp_files):
    """Test converting text to speech with ElevenLabs."""
    # Set up mocks
    mock_get_voices.return_value = [
        {"voice_id": "voice1", "name": "Voice 1"},
        {"voice_id": "voice2", "name": "Voice 2"},
    ]
    mock_synthesize.return_value = ["output1.mp3", "output2.mp3"]

    # Create the input file
    text_file = tmp_files["text_file"]
    text_file.write_text("Text for synthesis")

    # Run the speech synthesis
    with patch.dict(os.environ, {"ELEVENLABS_API_KEY": "mock-api-key"}):
        result = process_to_speech(
            str(text_file),
            str(tmp_files["audio_file"]),
            voice_filter=None,
            model="eleven_monolingual_v1",
            backup=False,
        )

    # Check result has the expected structure
    assert isinstance(result, dict)
    assert "output_files" in result
    assert "stats" in result

    # Check mock was called
    mock_synthesize.assert_called_once()


def test_full_pipeline_integration(
    tmp_files,
    sample_xml_content,
    sample_chunked_xml_content,
    sample_processed_xml_content,
):
    """
    Test the full pipeline integration with mocks.

    This test verifies that all pipeline stages can be connected
    and executed in the correct order.
    """
    # Create all necessary files
    tmp_files["input_file"].write_text(sample_xml_content)

    # Mock all the intermediate processing steps
    with (
        patch("e11ocutionist.e11ocutionist.process_to_chunks") as mock_to_chunks,
        patch("e11ocutionist.e11ocutionist.process_chunks") as mock_process_chunks,
        patch("e11ocutionist.e11ocutionist.process_to_text") as mock_to_text,
        patch("e11ocutionist.e11ocutionist.process_to_speech") as mock_to_speech,
    ):
        # Configure mocks with appropriate return values
        mock_to_chunks.return_value = {
            "output_file": str(tmp_files["chunked_file"]),
            "stats": {"chunks": 2, "items": 3},
        }
        tmp_files["chunked_file"].write_text(sample_chunked_xml_content)

        mock_process_chunks.return_value = {
            "output_file": str(tmp_files["processed_file"]),
            "stats": {"processed_items": 3, "tokens": 300},
        }
        tmp_files["processed_file"].write_text(sample_processed_xml_content)

        mock_to_text.return_value = {
            "output_file": str(tmp_files["text_file"]),
            "stats": {"characters": 150},
        }
        tmp_files["text_file"].write_text("Converted text")

        mock_to_speech.return_value = {
            "output_files": [str(tmp_files["audio_file"])],
            "stats": {"duration": 30, "voices": 1},
        }

        # Verify the pipeline can be executed in sequence
        from e11ocutionist.e11ocutionist import process_document

        result = process_document(
            str(tmp_files["input_file"]),
            output_dir=str(tmp_files["input_file"].parent),
            chunk_size=1000,
            llm_model="mock-model",
            llm_temperature=0.1,
            tts_model="eleven_monolingual_v1",
            voice_filter=None,
            verbose=True,
        )

        # Check the process completed and produced expected results
        assert isinstance(result, dict)
        assert "stages" in result
        assert "chunks" in result["stages"]
        assert "process" in result["stages"]
        assert "text" in result["stages"]
        assert "speech" in result["stages"]

        # Check all mocks were called in the correct sequence
        mock_to_chunks.assert_called_once()
        mock_process_chunks.assert_called_once()
        mock_to_text.assert_called_once()
        mock_to_speech.assert_called_once()
