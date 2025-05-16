"""Test suite for e11ocutionist entitizer module."""

from lxml import etree

from e11ocutionist.entitizer import (
    extract_item_elements,
    extract_nei_from_tags,
    merge_tagged_items,
)


def test_extract_item_elements():
    """Test the extraction of item elements from XML."""
    test_xml = """
    <chunk id="chunk1">
        <item id="item1" tok="10">Test content 1</item>
        <item id="item2" tok="15">Test content 2</item>
    </chunk>
    """

    result = extract_item_elements(test_xml)

    assert len(result) == 2
    assert "item1" in result
    assert "item2" in result
    assert '<item id="item1" tok="10">Test content 1</item>' in result["item1"]
    assert '<item id="item2" tok="15">Test content 2</item>' in result["item2"]


def test_extract_nei_from_tags():
    """Test extraction of NEIs from tags in text."""
    test_xml = """
    <doc>
        <item id="item1">This contains <nei>John Smith</nei> as a name.</item>
        <item id="item2">Another <nei pronunciation="acro-nim">NEI</nei> example.</item>
        <item id="item3"><nei orig="doctor">Dr.</nei> is an abbreviation.</item>
    </doc>
    """

    result = extract_nei_from_tags(test_xml)

    assert len(result) == 3
    assert "john smith" in result
    assert "nei" in result
    assert "dr." in result

    assert result["john smith"]["text"] == "John Smith"
    assert result["john smith"]["count"] == 1
    assert result["john smith"]["new"] is True

    assert result["nei"]["text"] == "NEI"
    assert result["nei"]["pronunciation"] == "acro-nim"

    assert result["dr."]["text"] == "Dr."
    assert result["dr."]["orig"] == "doctor"


def test_merge_tagged_items():
    """Test merging tagged items back into the original chunk."""
    # Create original chunk
    original_xml = """
    <chunk id="chunk1">
        <item id="item1">Original content 1</item>
        <item id="item2">Original content 2</item>
    </chunk>
    """
    parser = etree.XMLParser(remove_blank_text=False, recover=True)
    original_chunk = etree.fromstring(original_xml.encode("utf-8"), parser)

    # Create tagged items
    tagged_items = {
        "item1": '<item id="item1">Tagged <nei>content 1</nei></item>',
        "item2": '<item id="item2">Tagged <nei>content 2</nei></item>',
    }

    # Merge
    result = merge_tagged_items(original_chunk, tagged_items)

    # Assert the result contains the tagged content
    items = result.xpath(".//item")
    assert len(items) == 2

    item1_text = etree.tostring(items[0], encoding="utf-8", method="text").decode(
        "utf-8"
    )
    item2_text = etree.tostring(items[1], encoding="utf-8", method="text").decode(
        "utf-8"
    )

    assert "Tagged content 1" in item1_text
    assert "Tagged content 2" in item2_text

    # Check that <nei> tags were preserved in the structure
    nei_tags = result.xpath(".//nei")
    assert len(nei_tags) == 2
    assert nei_tags[0].text == "content 1"
    assert nei_tags[1].text == "content 2"


def test_nei_identification_mock():
    """Test NEI identification with a mock response."""
    # Mock the identify_entities function to avoid LLM API calls

    # Create a mock XML with some potential named entities

    # Expected response with NEI tags added
    expected_response = """
    <chunk id="chunk1">
        <item id="item1">This document mentions <nei>John Smith</nei> and <nei>Dr. Watson</nei>.</item>
        <item id="item2">They work at <nei>ACME Corp</nei> in <nei>New York City</nei>.</item>
    </chunk>
    """

    # Test that the extract_item_elements function correctly handles the response
    items_result = extract_item_elements(expected_response)
    assert len(items_result) == 2
    assert "item1" in items_result
    assert "item2" in items_result

    # Test that extract_nei_from_tags correctly identifies the NEIs
    nei_result = extract_nei_from_tags(expected_response)
    assert len(nei_result) == 4
    assert "john smith" in nei_result
    assert "dr. watson" in nei_result
    assert "acme corp" in nei_result
    assert "new york city" in nei_result
