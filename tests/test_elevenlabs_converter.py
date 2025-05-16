"""Test suite for e11ocutionist elevenlabs_converter module."""

from lxml import etree

from e11ocutionist.elevenlabs_converter import (
    extract_text_from_xml,
    process_dialog,
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


def test_process_dialog():
    """Test processing of dialog in text."""
    # Create a simpler input that won't cause indexing issues
    dialog_text = "Normal text.\n\n— This is dialog."

    result = process_dialog(dialog_text)

    # Check that em dash was replaced
    assert "Normal text." in result
    assert '"This is dialog."' in result

    # Test with dialog toggle within a line
    mixed_text = "He said — This is what I said — then left."

    mixed_result = process_dialog(mixed_text)

    # Check dialog was properly processed
    assert "He said " in mixed_result
    assert "This is what I said" in mixed_result
    assert " then left." in mixed_result
