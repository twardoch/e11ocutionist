#!/usr/bin/env python3
"""Tests for the tonedown module."""

from lxml import etree

from e11ocutionist.tonedown import (
    extract_neis_from_document,
    detect_language,
    update_nei_tags,
    reduce_emphasis,
    process_document,
)


def test_extract_neis_from_document(sample_xml):
    """Test NEI extraction from XML document."""
    # Test with sample XML containing NEIs
    result = extract_neis_from_document(sample_xml)
    assert isinstance(result, dict)
    assert "named entity" in result
    assert result["named entity"]["text"] == "Named Entity"
    assert result["named entity"]["count"] == 1
    assert result["named entity"]["new"] is False

    # Test with XML containing multiple instances of same NEI
    xml_with_duplicates = """<?xml version="1.0" encoding="UTF-8"?>
<document>
    <content>
        <p>First <nei type="person">John Smith</nei> mention.</p>
        <p>Second <nei type="person" new="true">John Smith</nei> 
        mention.</p>
        <p>Third <nei type="person" orig="Jon Smyth">John Smith</nei> 
        mention.</p>
    </content>
</document>"""

    result = extract_neis_from_document(xml_with_duplicates)
    assert "john smith" in result
    assert result["john smith"]["count"] == 3
    assert result["john smith"]["new"] is True
    assert result["john smith"]["orig"] == "Jon Smyth"


def test_detect_language(sample_xml):
    """Test language detection."""
    # Test with default model and temperature
    lang, conf = detect_language(sample_xml, "gpt-4o", 0.2)
    assert isinstance(lang, str)
    assert isinstance(conf, float)
    assert lang == "en"  # Default/mock response
    assert 0 <= conf <= 1


def test_update_nei_tags():
    """Test updating NEI tags in XML."""
    xml_content = """<?xml version="1.0" encoding="UTF-8"?>
<document>
    <content>
        <p><nei type="person">John Smith</nei> is here.</p>
    </content>
</document>"""

    nei_dict = {
        "john smith": {
            "text": "John Smith",
            "new": True,
            "orig": "Jon Smyth",
            "count": 1,
        }
    }

    result = update_nei_tags(xml_content, nei_dict)
    assert isinstance(result, str)

    # Parse and verify the updated XML
    root = etree.fromstring(result.encode())
    nei = root.find(".//nei")
    assert nei is not None
    assert nei.get("new") == "true"
    assert nei.get("orig") == "Jon Smyth"


def test_reduce_emphasis():
    """Test emphasis reduction in XML."""
    xml_content = """<?xml version="1.0" encoding="UTF-8"?>
<document>
    <content>
        <p>This is <em>very</em> important and this is also <em>very</em> 
        important.</p>
    </content>
</document>"""

    # Test with default minimum distance
    result = reduce_emphasis(xml_content)
    assert isinstance(result, str)

    # Parse and count remaining emphasis tags
    root = etree.fromstring(result.encode())
    em_tags = root.findall(".//em")
    assert len(em_tags) < 2  # At least one emphasis should be removed

    # Test with large minimum distance (should preserve all emphasis)
    result = reduce_emphasis(xml_content, min_distance=1000)
    root = etree.fromstring(result.encode())
    em_tags = root.findall(".//em")
    assert len(em_tags) == 2


def test_process_document(temp_workspace):
    """Test the complete document processing pipeline."""
    # Create test input file
    input_file = temp_workspace / "input" / "test.xml"
    output_file = temp_workspace / "output" / "test.xml"

    test_content = """<?xml version="1.0" encoding="UTF-8"?>
<document>
    <content>
        <p><nei type="person">John Smith</nei> said this is <em>very</em> 
        important.</p>
        <p>And this is also <em>very</em> important, said 
        <nei type="person">John Smith</nei>.</p>
    </content>
</document>"""

    input_file.write_text(test_content)

    # Process the document
    result = process_document(
        str(input_file),
        str(output_file),
        model="gpt-4o",
        temperature=0.2,
        min_em_distance=5,
        backup=True,
        verbose=True,
    )

    assert isinstance(result, str)
    assert output_file.exists()

    # Verify the processed content
    processed_content = output_file.read_text()
    root = etree.fromstring(processed_content.encode())

    # Check NEI processing
    neis = root.findall(".//nei")
    assert len(neis) == 2

    # Check emphasis reduction
    em_tags = root.findall(".//em")
    assert len(em_tags) < 2  # At least one emphasis should be removed


def test_error_handling():
    """Test error handling in tonedown functions."""
    # Test with invalid XML
    invalid_xml = "This is not XML"
    result = extract_neis_from_document(invalid_xml)
    assert isinstance(result, dict)
    assert len(result) == 0

    # Test language detection with invalid XML
    lang, conf = detect_language(invalid_xml, "gpt-4o", 0.2)
    assert lang == "en"  # Should return default
    assert conf == 0.5  # Should return default confidence

    # Test with empty XML
    empty_xml = """<?xml version="1.0" encoding="UTF-8"?><document></document>"""
    result = extract_neis_from_document(empty_xml)
    assert isinstance(result, dict)
    assert len(result) == 0

    # Test update_nei_tags with invalid XML
    result = update_nei_tags(invalid_xml, {})
    assert result == invalid_xml  # Should return input unchanged

    # Test reduce_emphasis with invalid XML
    result = reduce_emphasis(invalid_xml)
    assert result == invalid_xml  # Should return input unchanged
