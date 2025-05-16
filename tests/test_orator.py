"""Test suite for e11ocutionist orator module."""

from e11ocutionist.orator import (
    extract_chunk_items,
    merge_processed_items,
    enhance_punctuation,
)


def test_extract_chunk_items():
    """Test extraction of items from XML chunks."""
    test_xml = """
    <doc>
        <chunk id="chunk1">
            <item id="item1">This is content for item 1.</item>
            <item id="item2">This is <em>content</em> for item 2.</item>
        </chunk>
    </doc>
    """

    result = extract_chunk_items(test_xml)

    assert len(result) == 2

    # Check the first item
    assert result[0][0] == "item1"  # item_id
    assert "This is content for item 1." in result[0][1]  # item_content
    assert '<item id="item1">' in result[0][2]  # item_xml

    # Check the second item with embedded tags
    assert result[1][0] == "item2"  # item_id
    assert "<em>content</em>" in result[1][1]  # item_content
    assert '<item id="item2">' in result[1][2]  # item_xml


def test_merge_processed_items():
    """Test merging processed items back into the original XML document."""
    # Original XML with items
    original_xml = """
    <doc>
        <chunk id="chunk1">
            <item id="item1">Original content 1.</item>
            <item id="item2">Original content 2.</item>
        </chunk>
    </doc>
    """

    # Processed items with modified content
    processed_items = [
        ("item1", "Modified content 1."),
        ("item2", "Modified content 2."),
    ]

    result = merge_processed_items(original_xml, processed_items)

    # Check that the document structure is preserved
    assert "<doc>" in result
    assert '<chunk id="chunk1">' in result

    # Check that item content has been updated
    assert '<item id="item1">Modified content 1.</item>' in result
    assert '<item id="item2">Modified content 2.</item>' in result

    # Make sure original content is replaced
    assert "Original content 1." not in result
    assert "Original content 2." not in result


def test_enhance_punctuation():
    """Test enhancement of punctuation in text items."""
    # Test items with various punctuation needs
    test_items = [
        ("item1", "Text with parenthetical(word) needs commas"),
        ("item2", 'End quote with period." Next sentence.'),
        ("item3", "Text with colon: needs pause"),
        ("item4", "Text with numbered list 1. item; 2. item;"),
    ]

    result = enhance_punctuation(test_items)

    # Check that results have expected format
    assert len(result) == 4
    assert isinstance(result[0], tuple)
    assert len(result[0]) == 2

    # Check specific enhancements based on actual implementation
    assert result[0][0] == "item1"  # item_id preserved
    assert "parenthetical, (word)" in result[0][1]  # added commas

    assert result[1][0] == "item2"
    assert '." <hr/>' in result[1][1]  # added pause after quoted speech

    assert result[2][0] == "item3"
    assert "colon: <hr/>" in result[2][1]  # added pause after colon

    assert result[3][0] == "item4"
    assert "1. item, " in result[3][1]  # replaced semicolon with comma
