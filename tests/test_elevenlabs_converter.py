#!/usr/bin/env python3
"""Tests for the elevenlabs_converter module."""

from pathlib import Path

import pytest
from lxml import etree

from e11ocutionist.elevenlabs_converter import (
    extract_text_from_xml,
    process_dialog,
    process_document,
)


def test_extract_text_from_xml(sample_xml):
    """Test extraction of text from XML document."""
    root = etree.fromstring(sample_xml.encode())
    result = extract_text_from_xml(root)

    assert isinstance(result, str)
    assert "This is a test paragraph" in result
    assert "Hello" in result
    assert "Named Entity" in result
    assert "New Entity" in result
    assert '"New Entity"' in result  # new=true entities should be quoted


def test_extract_text_from_xml_with_formatting():
    """Test extraction of text with various formatting elements."""
    xml = """<?xml version="1.0" encoding="UTF-8"?>
    <document>
        <content>
            <chunk>
                <item>
                    This is <em>emphasized</em> text with a 
                    <nei>Named Entity</nei> and a
                    <nei new="true">New Entity</nei>.
                    Here's a break: <hr/> and some more text.
                </item>
            </chunk>
        </content>
    </document>
    """

    root = etree.fromstring(xml.encode())
    result = extract_text_from_xml(root)

    assert '"emphasized"' in result  # <em> converted to quotes
    assert "Named Entity" in result  # regular NEI preserved
    assert '"New Entity"' in result  # new NEI converted to quotes
    assert '<break time="0.5s" />' in result  # <hr/> converted to break


def test_process_dialog():
    """Test dialog processing."""
    text = """Regular text.

— Hello, said John.
— Hi there! — he continued — How are you?
— I'm fine, thanks.

More regular text.

— Another dialog starts.
— Yes, it does."""

    result = process_dialog(text)

    assert '"Hello, said John."' in result
    assert (
        '"Hi there!" CLOSE_Q; OPEN_Q "he continued" CLOSE_Q; OPEN_Q "How are you?"'
        in result
    )
    assert '"I\'m fine, thanks."' in result
    assert '"Another dialog starts."' in result
    assert '"Yes, it does."' in result


def test_process_dialog_with_breaks():
    """Test dialog processing with break tags."""
    text = """— First line...
— Second line with <break time="1s" /> a pause.
— Third line."""

    result = process_dialog(text)

    assert '"First line..."' in result
    assert '"Second line ... a pause."' in result
    assert '"Third line."' in result


def test_process_document_xml(temp_workspace):
    """Test document processing in XML mode."""
    # Create test input file
    input_file = temp_workspace / "input" / "test.xml"
    output_file = temp_workspace / "output" / "test.txt"

    input_content = """<?xml version="1.0" encoding="UTF-8"?>
    <document>
        <content>
            <chunk>
                <item>Regular paragraph.</item>
                <item>
                    — Hello, said John.
                    — Hi there! — he continued — How are you?
                </item>
            </chunk>
        </content>
    </document>
    """

    input_file.write_text(input_content)

    # Process document
    result = process_document(
        str(input_file),
        str(output_file),
        dialog=True,
        plaintext=False,
        verbose=True,
    )

    assert isinstance(result, dict)
    assert result["success"] is True
    assert output_file.exists()

    # Check output content
    output_text = output_file.read_text()
    assert "Regular paragraph" in output_text
    assert '"Hello, said John."' in output_text
    assert '"Hi there!"' in output_text
    assert "he continued" in output_text
    assert '"How are you?"' in output_text


def test_process_document_plaintext(temp_workspace):
    """Test document processing in plaintext mode."""
    # Create test input file
    input_file = temp_workspace / "input" / "test.txt"
    output_file = temp_workspace / "output" / "test.txt"

    input_content = """Regular text.

— Hello, said John.
— Hi there! — he continued — How are you?
— I'm fine, thanks.

More regular text."""

    input_file.write_text(input_content)

    # Process document
    result = process_document(
        str(input_file),
        str(output_file),
        dialog=True,
        plaintext=True,
        verbose=True,
    )

    assert isinstance(result, dict)
    assert result["success"] is True
    assert output_file.exists()

    # Check output content
    output_text = output_file.read_text()
    assert "Regular text" in output_text
    assert '"Hello, said John."' in output_text
    assert '"Hi there!"' in output_text
    assert "he continued" in output_text
    assert '"How are you?"' in output_text
    assert '"I\'m fine, thanks."' in output_text
    assert "More regular text" in output_text


def test_error_handling(temp_workspace):
    """Test error handling in document processing."""
    # Test with invalid XML
    input_file = temp_workspace / "input" / "invalid.xml"
    output_file = temp_workspace / "output" / "invalid.txt"

    input_file.write_text("<invalid>")

    result = process_document(
        str(input_file),
        str(output_file),
        dialog=True,
        plaintext=False,
        verbose=True,
    )

    assert isinstance(result, dict)
    assert result["success"] is False
    assert "error" in result
    assert "XML parsing failed" in result["error"]

    # Test with missing input file
    missing_file = temp_workspace / "input" / "missing.xml"
    with pytest.raises(FileNotFoundError):
        process_document(
            str(missing_file),
            str(output_file),
            dialog=True,
            plaintext=False,
        )
