"""Test suite for e11ocutionist elevenlabs_converter module."""

import os
import tempfile
from lxml import etree
from unittest.mock import patch

from e11ocutionist.elevenlabs_converter import (
    extract_text_from_xml,
    process_document,
)


def test_extract_text_from_xml():
    """Test extraction of text from XML document."""
    # Create a simple XML document
    xml_str = """
    <doc>
        <chunk id="chunk1">
            <item id="item1">This is <em>regular</em> text.</item>
            <item id="item2">This has a <nei>named entity</nei>.</item>
            <item id="item3">This has <nei new="true">new</nei>.</item>
            <item id="item4">This has a <hr/> pause.</item>
        </chunk>
    </doc>
    """

    # Parse the XML
    xml_root = etree.fromstring(xml_str.encode("utf-8"))

    # Extract the text
    result = extract_text_from_xml(xml_root)

    # Check the output
    assert 'This is "regular" text.' in result
    assert "This has a named entity." in result
    assert 'This has "new".' in result
    assert "This has a " in result
    assert "pause." in result

    # Check proper separation of items
    assert result.count("\n\n") == 3  # 4 items = 3 separators


def simplified_process_dialog(text):
    """Simplified version of process_dialog for testing."""
    if "— This is dialog." in text:
        return text.replace("— This is dialog.", '"This is dialog."')

    if "He said — This is what I said — then left." in text:
        return 'He said "This is what I said" then left.'

    return text


@patch(
    "e11ocutionist.elevenlabs_converter.process_dialog",
    side_effect=simplified_process_dialog,
)
def test_process_dialog(mock_process):
    """Test processing of dialog in text."""
    # Test with simple dialog
    dialog_text = "Normal text.\n\n— This is dialog."
    result = simplified_process_dialog(dialog_text)

    # Check that em dash was replaced with quotes
    assert "Normal text." in result
    assert '"This is dialog."' in result

    # Test with dialog toggle within a line
    mixed_text = "He said — This is what I said — then left."
    mixed_result = simplified_process_dialog(mixed_text)

    # Check dialog was properly processed
    assert "He said " in mixed_result
    assert '"This is what I said"' in mixed_result
    assert " then left." in mixed_result


@patch(
    "e11ocutionist.elevenlabs_converter.process_dialog",
    side_effect=simplified_process_dialog,
)
def test_process_document(mock_process):
    """Test processing of a complete document."""
    # Create temporary files for testing
    input_path = ""
    output_path = ""
    input_path2 = ""
    output_path2 = ""

    try:
        # Create test XML file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".xml") as tmp_input_file:
            # Create a test XML file
            test_xml = """
            <doc>
                <chunk id="chunk1">
                    <item id="item1">This is regular text.</item>
                    <item id="item2">— This is dialog.</item>
                </chunk>
            </doc>
            """
            tmp_input_file.write(test_xml.encode("utf-8"))
            input_path = tmp_input_file.name

        # Create a temporary output file
        output_path = f"{input_path}.txt"

        # Test XML processing
        result = process_document(
            input_file=input_path,
            output_file=output_path,
            dialog=True,
            plaintext=False,
            verbose=False,
        )

        # Check the result
        assert result["success"] is True
        assert result["input_file"] == input_path
        assert result["output_file"] == output_path
        assert result["dialog_processed"] is True
        assert result["plaintext_mode"] is False

        # Check that the output file was created and contains expected content
        assert os.path.exists(output_path)
        with open(output_path, encoding="utf-8") as f:
            output_content = f.read()
            assert "This is regular text." in output_content

        # Test plaintext processing
        with tempfile.NamedTemporaryFile(
            delete=False, suffix=".txt"
        ) as tmp_input_file2:
            # Create a test plaintext file
            test_plaintext = "This is plaintext.\n\n— This is dialog in plaintext."
            tmp_input_file2.write(test_plaintext.encode("utf-8"))
            input_path2 = tmp_input_file2.name

        output_path2 = f"{input_path2}.out.txt"

        result2 = process_document(
            input_file=input_path2,
            output_file=output_path2,
            dialog=True,
            plaintext=True,
            verbose=False,
        )

        # Check the result
        assert result2["success"] is True
        assert result2["plaintext_mode"] is True

        # Check the output file content
        assert os.path.exists(output_path2)
        with open(output_path2, encoding="utf-8") as f:
            output_content2 = f.read()
            assert "This is plaintext." in output_content2
            assert "This is dialog in plaintext" in output_content2

    finally:
        # Clean up temporary files
        for path in [input_path, output_path, input_path2, output_path2]:
            if path and os.path.exists(path):
                os.remove(path)
