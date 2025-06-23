#!/usr/bin/env python3
"""Tests for the orator module."""


from lxml import etree

from e11ocutionist.orator import (
    extract_chunk_items,
    restructure_sentences,
    normalize_words,
    enhance_punctuation,
    add_emotional_emphasis,
    merge_processed_items,
    process_document,
)


def test_extract_chunk_items(sample_xml):
    """Test extraction of items from XML document."""
    items = extract_chunk_items(sample_xml)
    assert isinstance(items, list)
    for item_id, content, xml in items:
        assert isinstance(item_id, str)
        assert isinstance(content, str)
        assert isinstance(xml, str)
        assert content.strip()  # Content should not be empty
        assert "<item" in xml and "</item>" in xml


def test_restructure_sentences():
    """Test sentence restructuring."""
    items = [
        (
            "id1",
            "This is a very long sentence that could be split into two. "
            "And here's another one.",
            "<item>content</item>",
        ),
        ("id2", "Short sentence.", "<item>content</item>"),
    ]

    result = restructure_sentences(items, "gpt-4o", 0.2)
    assert isinstance(result, list)
    assert len(result) == len(items)
    for item_id, content in result:
        assert isinstance(item_id, str)
        assert isinstance(content, str)
        assert content.strip()


def test_normalize_words():
    """Test word normalization."""
    items = [
        ("id1", "There are 42 items and $100."),
        ("id2", "The temperature is 98.6°F."),
    ]

    result = normalize_words(items, "gpt-4o", 0.2)
    assert isinstance(result, list)
    assert len(result) == len(items)
    for item_id, content in result:
        assert isinstance(item_id, str)
        assert isinstance(content, str)
        assert content.strip()


def test_enhance_punctuation():
    """Test punctuation enhancement."""
    items = [
        ("id1", "This needs a period"),
        ("id2", "Is this a question"),
        ("id3", "Add some emphasis here"),
    ]

    result = enhance_punctuation(items)
    assert isinstance(result, list)
    assert len(result) == len(items)
    for item_id, content in result:
        assert isinstance(item_id, str)
        assert isinstance(content, str)
        assert content.strip()
        assert content[-1] in ".!?"  # Should end with punctuation


def test_add_emotional_emphasis():
    """Test adding emotional emphasis."""
    items = [
        ("id1", "This is a happy moment."),
        ("id2", "This is a sad story."),
    ]

    result = add_emotional_emphasis(items, "gpt-4o", 0.2)
    assert isinstance(result, list)
    assert len(result) == len(items)
    for item_id, content in result:
        assert isinstance(item_id, str)
        assert isinstance(content, str)
        assert content.strip()


def test_merge_processed_items():
    """Test merging processed items back into XML."""
    xml_content = """<?xml version="1.0" encoding="UTF-8"?>
<document>
    <content>
        <item id="id1">Original content 1</item>
        <item id="id2">Original content 2</item>
    </content>
</document>"""

    processed_items = [
        ("id1", "Processed content 1"),
        ("id2", "Processed content 2"),
    ]

    result = merge_processed_items(xml_content, processed_items)
    assert isinstance(result, str)

    # Parse and verify the merged XML
    root = etree.fromstring(result.encode())
    items = root.findall(".//item")
    assert len(items) == len(processed_items)

    # Check that content was updated
    for item, (_, content) in zip(items, processed_items, strict=False):
        assert item.text == content


def test_process_document(temp_workspace):
    """Test the complete document processing pipeline."""
    # Create test input file
    input_file = temp_workspace / "input" / "test.xml"
    output_file = temp_workspace / "output" / "test.xml"

    test_content = """<?xml version="1.0" encoding="UTF-8"?>
<document>
    <content>
        <item id="id1">This needs processing.</item>
        <item id="id2">This also needs work.</item>
    </content>
</document>"""

    input_file.write_text(test_content)

    # Process the document
    result = process_document(
        str(input_file),
        str(output_file),
        steps=["restructure", "normalize", "punctuate", "emphasize"],
        model="gpt-4o",
        temperature=0.2,
        verbose=True,
        backup=True,
    )

    assert isinstance(result, dict)
    assert "processed_items" in result
    assert output_file.exists()

    # Verify the processed content
    processed_content = output_file.read_text()
    root = etree.fromstring(processed_content.encode())
    items = root.findall(".//item")
    assert len(items) == 2


def test_error_handling():
    """Test error handling in orator functions."""
    # Test with invalid XML
    invalid_xml = "This is not XML"
    items = extract_chunk_items(invalid_xml)
    assert isinstance(items, list)
    assert len(items) == 0

    # Test with empty items list
    result = restructure_sentences([], "gpt-4o", 0.2)
    assert isinstance(result, list)
    assert len(result) == 0

    result = normalize_words([], "gpt-4o", 0.2)
    assert isinstance(result, list)
    assert len(result) == 0

    result = enhance_punctuation([])
    assert isinstance(result, list)
    assert len(result) == 0

    result = add_emotional_emphasis([], "gpt-4o", 0.2)
    assert isinstance(result, list)
    assert len(result) == 0

    # Test merge with invalid XML
    result = merge_processed_items(invalid_xml, [])
    assert result == invalid_xml  # Should return input unchanged
