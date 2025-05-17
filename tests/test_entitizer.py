#!/usr/bin/env python3
"""Tests for the entitizer module."""

from pathlib import Path
from copy import deepcopy

import pytest
from lxml import etree

from e11ocutionist.entitizer import (
    get_chunks,
    get_chunk_text,
    save_current_state,
    extract_item_elements,
    merge_tagged_items,
    identify_entities,
    extract_response_text,
    extract_nei_from_tags,
    process_chunks,
    process_document,
)


def test_get_chunks(sample_xml):
    """Test extraction of chunks from XML document."""
    root = etree.fromstring(sample_xml.encode())
    chunks = get_chunks(root)
    assert isinstance(chunks, list)
    assert all(isinstance(chunk, etree._Element) for chunk in chunks)


def test_get_chunk_text():
    """Test extraction of text from a chunk."""
    chunk_xml = """
    <chunk id="test">
        <item id="1">First item</item>
        <item id="2">Second item with <nei>entity</nei></item>
    </chunk>
    """
    chunk = etree.fromstring(chunk_xml.encode())
    text = get_chunk_text(chunk)
    assert isinstance(text, str)
    assert "First item" in text
    assert "Second item" in text
    assert "<nei>entity</nei>" in text


def test_save_current_state(temp_workspace):
    """Test saving current state to files."""
    # Create test data
    xml_content = """<?xml version="1.0" encoding="UTF-8"?>
    <document><content><chunk id="test">Test content</chunk></content></document>
    """
    root = etree.fromstring(xml_content.encode())
    nei_dict = {"test_entity": {"text": "Test", "count": 1, "new": False}}

    # Set up output files
    output_file = temp_workspace / "output" / "test_output.xml"
    nei_dict_file = temp_workspace / "output" / "test_nei_dict.json"

    # Test saving state
    save_current_state(
        root,
        nei_dict,
        str(output_file),
        str(nei_dict_file),
        chunk_id="test",
        backup=True,
    )

    assert output_file.exists()
    assert nei_dict_file.exists()

    # Verify content
    saved_xml = output_file.read_text()
    assert "Test content" in saved_xml

    import json

    saved_dict = json.loads(nei_dict_file.read_text())
    assert saved_dict == nei_dict


def test_extract_item_elements():
    """Test extraction of item elements from XML text."""
    xml_text = """
    <chunk>
        <item id="1" type="normal">First item</item>
        <item id="2" type="dialog">Second item with <nei>entity</nei></item>
    </chunk>
    """

    items = extract_item_elements(xml_text)
    assert isinstance(items, dict)
    assert len(items) == 2
    assert "1" in items
    assert "2" in items
    assert 'type="normal"' in items["1"]
    assert 'type="dialog"' in items["2"]
    assert "<nei>entity</nei>" in items["2"]


def test_merge_tagged_items():
    """Test merging tagged items back into chunk."""
    # Create original chunk
    chunk_xml = """
    <chunk id="test">
        <item id="1">Original content 1</item>
        <item id="2">Original content 2</item>
    </chunk>
    """
    original_chunk = etree.fromstring(chunk_xml.encode())

    # Create tagged items
    tagged_items = {
        "1": '<item id="1">Updated content 1 with <nei>entity</nei></item>',
        "2": '<item id="2">Updated content 2</item>',
    }

    # Merge items
    result = merge_tagged_items(original_chunk, tagged_items)
    assert isinstance(result, etree._Element)

    # Verify updates
    items = result.xpath(".//item")
    assert len(items) == 2
    assert "Updated content 1" in etree.tostring(items[0], encoding="utf-8").decode()
    assert "<nei>entity</nei>" in etree.tostring(items[0], encoding="utf-8").decode()
    assert "Updated content 2" in etree.tostring(items[1], encoding="utf-8").decode()


def test_identify_entities():
    """Test entity identification in text."""
    chunk_text = """
    <chunk>
        <item id="1">John Smith visited New York.</item>
        <item id="2">He met Sarah Johnson at Central Park.</item>
    </chunk>
    """

    nei_dict = {
        "John Smith": {"text": "John Smith", "count": 1, "new": False},
        "New York": {"text": "New York", "count": 1, "new": False},
    }

    result = identify_entities(chunk_text, nei_dict, "gpt-4o", 0.1)
    assert isinstance(result, str)
    assert "<nei" in result
    assert "John Smith" in result
    assert "New York" in result


def test_extract_response_text():
    """Test extraction of response text."""
    response = {
        "choices": [
            {
                "message": {
                    "content": "Test response content",
                },
            },
        ],
    }

    text = extract_response_text(response)
    assert isinstance(text, str)
    assert text == "Test response content"


def test_extract_nei_from_tags():
    """Test extraction of NEIs from tagged text."""
    tagged_text = """
    <item>
        <nei type="person">John Smith</nei> visited 
        <nei type="location">New York</nei>.
    </item>
    """

    result = extract_nei_from_tags(tagged_text)
    assert isinstance(result, dict)
    assert "John Smith" in result
    assert "New York" in result
    assert result["John Smith"]["type"] == "person"
    assert result["New York"]["type"] == "location"


def test_process_chunks():
    """Test processing of document chunks."""
    # Create test chunks
    chunk_xml = """
    <chunk id="test">
        <item id="1">John Smith visited New York.</item>
        <item id="2">He met Sarah Johnson at Central Park.</item>
    </chunk>
    """
    chunk = etree.fromstring(chunk_xml.encode())
    chunks = [chunk]

    # Initial NEI dictionary
    nei_dict = {
        "John Smith": {"text": "John Smith", "count": 1, "new": False},
        "New York": {"text": "New York", "count": 1, "new": False},
    }

    # Process chunks
    root, updated_dict = process_chunks(
        chunks,
        model="gpt-4o",
        temperature=0.1,
        nei_dict=nei_dict,
    )

    assert isinstance(root, etree._Element)
    assert isinstance(updated_dict, dict)
    assert len(updated_dict) >= len(nei_dict)


def test_process_document(temp_workspace):
    """Test complete document processing."""
    # Create test input file
    input_file = temp_workspace / "input" / "test.xml"
    output_file = temp_workspace / "output" / "test_processed.xml"
    nei_dict_file = temp_workspace / "output" / "test_nei_dict.json"

    input_content = """<?xml version="1.0" encoding="UTF-8"?>
    <document>
        <content>
            <chunk id="1">
                <item id="1">John Smith visited New York.</item>
                <item id="2">He met Sarah Johnson at Central Park.</item>
            </chunk>
        </content>
    </document>
    """

    input_file.write_text(input_content)

    # Process document
    result = process_document(
        str(input_file),
        str(output_file),
        str(nei_dict_file),
        model="gpt-4o",
        temperature=0.1,
        verbose=True,
        backup=True,
    )

    assert isinstance(result, dict)
    assert "success" in result
    assert result["success"] is True
    assert output_file.exists()
    assert nei_dict_file.exists()


def test_error_handling():
    """Test error handling in entitizer functions."""
    # Test with invalid XML
    with pytest.raises(etree.XMLSyntaxError):
        etree.fromstring("<invalid>")

    # Test with empty input
    assert extract_item_elements("") == {}
    assert extract_nei_from_tags("") == {}

    # Test with malformed XML
    malformed_xml = "<chunk><item>No closing tags"
    items = extract_item_elements(malformed_xml)
    assert len(items) == 0
