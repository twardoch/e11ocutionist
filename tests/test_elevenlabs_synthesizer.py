"""Test suite for e11ocutionist elevenlabs_synthesizer module."""

from unittest.mock import patch, MagicMock

import pytest

from e11ocutionist.elevenlabs_synthesizer import (
    sanitize_filename,
    get_personal_voices,
    synthesize_with_voice,
    synthesize_with_all_voices,
)


def test_sanitize_filename():
    """Test sanitization of filenames."""
    # Test with various invalid characters
    invalid_chars = 'file/with\\invalid*chars?"<>|'
    assert sanitize_filename(invalid_chars) == "filewithinvalidchars"

    # Test with spaces
    assert sanitize_filename("file with spaces") == "file_with_spaces"

    # Test with other problematic characters
    assert sanitize_filename("file-with-hyphens.mp3") == "file-with-hyphens.mp3"

    # Test truncation
    long_name = "a" * 200
    assert len(sanitize_filename(long_name)) == 100


@patch("e11ocutionist.elevenlabs_synthesizer.set_api_key")
@patch("e11ocutionist.elevenlabs_synthesizer.voices")
def test_get_personal_voices(mock_voices, mock_set_api_key):
    """Test getting personal voices from ElevenLabs."""
    # Create mock voices
    mock_voice1 = MagicMock()
    mock_voice1.category = "cloned"
    mock_voice1.name = "Voice 1"
    mock_voice1.voice_id = "voice1_id"

    mock_voice2 = MagicMock()
    mock_voice2.category = "generated"
    mock_voice2.name = "Voice 2"
    mock_voice2.voice_id = "voice2_id"

    mock_voice3 = MagicMock()
    mock_voice3.category = "premade"  # Not a personal voice
    mock_voice3.name = "Voice 3"
    mock_voice3.voice_id = "voice3_id"

    # Set up the mock to return our test voices
    mock_voices.return_value = [mock_voice1, mock_voice2, mock_voice3]

    # Call the function
    result = get_personal_voices("fake_api_key")

    # Verify the API key was set
    mock_set_api_key.assert_called_once_with("fake_api_key")

    # Verify we got only the personal voices
    assert len(result) == 2
    assert result[0].name == "Voice 1"
    assert result[1].name == "Voice 2"
    # Voice 3 should be filtered out as it's not a personal voice


@patch("e11ocutionist.elevenlabs_synthesizer.generate")
@patch("os.path.exists")
@patch("builtins.open", new_callable=MagicMock)
def test_synthesize_with_voice(mock_open, mock_path_exists, mock_generate):
    """Test synthesizing text with a specific voice."""
    # Set up mocks
    mock_path_exists.return_value = False
    mock_generate.return_value = b"fake_audio_data"

    # Create a mock file handle
    mock_file = MagicMock()
    mock_open.return_value.__enter__.return_value = mock_file

    # Create a mock voice
    mock_voice = MagicMock()
    mock_voice.name = "Test Voice"
    mock_voice.voice_id = "test_voice_id"

    # Call the function
    with patch("os.makedirs") as mock_makedirs:
        result = synthesize_with_voice(
            "Test text",
            mock_voice,
            "output_dir",
            model_id="test_model",
            output_format="mp3",
        )

    # Verify directories were created
    mock_makedirs.assert_called_once_with("output_dir", exist_ok=True)

    # Verify audio was generated with correct parameters
    mock_generate.assert_called_once_with(
        text="Test text", voice=mock_voice, model="test_model", output_format="mp3"
    )

    # Verify audio was saved
    mock_file.write.assert_called_once_with(b"fake_audio_data")

    # Verify the returned path is correct
    assert "test_voice_id--Test_Voice.mp3" in result


@patch("e11ocutionist.elevenlabs_synthesizer.get_personal_voices")
@patch("e11ocutionist.elevenlabs_synthesizer.synthesize_with_voice")
@patch("e11ocutionist.elevenlabs_synthesizer.set_api_key")
def test_synthesize_with_all_voices(mock_set_api_key, mock_synthesize, mock_get_voices):
    """Test synthesizing text with all personal voices."""
    # Create mock voices
    mock_voice1 = MagicMock()
    mock_voice1.name = "Voice 1"
    mock_voice1.voice_id = "voice1_id"

    mock_voice2 = MagicMock()
    mock_voice2.name = "Voice 2"
    mock_voice2.voice_id = "voice2_id"

    # Set up mocks
    mock_get_voices.return_value = [mock_voice1, mock_voice2]
    mock_synthesize.side_effect = [
        "output_dir/voice1_id--Voice_1.mp3",
        Exception("Synthesis failed"),  # Simulate one failure
    ]

    # Call the function with explicit API key
    result = synthesize_with_all_voices(
        text="Test text",
        output_dir="output_dir",
        api_key="fake_api_key",
        model_id="test_model",
        output_format="mp3",
        verbose=True,
    )

    # Verify API key was set
    mock_set_api_key.assert_called_once_with("fake_api_key")

    # Verify personal voices were retrieved
    mock_get_voices.assert_called_once()

    # Verify synthesis was attempted for each voice
    assert mock_synthesize.call_count == 2


@patch("os.environ.get")
def test_synthesize_with_all_voices_missing_api_key(mock_environ_get):
    """Test that an error is raised when no API key is provided."""
    # Simulate missing environment variable
    mock_environ_get.return_value = None

    # Call the function without API key and expect an error
    with pytest.raises(ValueError, match="ElevenLabs API key not provided"):
        synthesize_with_all_voices(
            text="Test text",
            output_dir="output_dir",
            api_key=None,
            verbose=True,
        )
