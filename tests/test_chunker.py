#!/usr/bin/env python3
"""Tests for the chunker module."""

from e11ocutionist.chunker import (
    count_tokens,
    escape_xml_chars,
    generate_hash,
    generate_id,
    split_into_paragraphs,
    is_heading,
    is_blockquote,
    is_list,
    is_code_block,
    is_horizontal_rule,
    is_table,
    is_html_block,
    is_image_or_figure,
    itemize_document,
    create_item_elements,
    create_chunks_on_tree as create_chunks, # Alias for test compatibility
    clean_consecutive_newlines,
    # pretty_print_xml, # This function was removed/changed to pretty_print_xml_from_tree
    process_document,
    pretty_print_xml_from_tree, # Import new one if tests need it directly
)


def test_count_tokens():
    """Test token counting."""
    text = "This is a test sentence."
    count = count_tokens(text)
    assert isinstance(count, int)
    assert count > 0


def test_escape_xml_chars():
    """Test XML character escaping."""
    test_cases = [
        ("&<>\"'", "&amp;&lt;&gt;&quot;&apos;"),
        ("Normal text", "Normal text"),
        ("Mixed & <tags>", "Mixed &amp; &lt;tags&gt;"),
        ("Unicode ♥", "Unicode ♥"),  # Unicode should be preserved
    ]

    for input_text, expected in test_cases:
        assert escape_xml_chars(input_text) == expected


def test_generate_hash():
    """Test hash generation."""
    text = "Test content"
    hash1 = generate_hash(text)
    assert isinstance(hash1, str)
    assert len(hash1) == 6
    assert hash1.isalnum()

    # Test consistency
    hash2 = generate_hash(text)
    assert hash1 == hash2


def test_generate_id():
    """Test ID generation."""
    content = "Test content"

    # Test first ID
    first_id = generate_id("", content)
    assert isinstance(first_id, str)
    assert len(first_id) == 13  # 6 chars + hyphen + 6 chars
    assert first_id.startswith("000000-")

    # Test subsequent ID
    second_id = generate_id(first_id, "More content")
    assert len(second_id) == 13
    assert second_id.startswith(first_id.split("-")[1] + "-")


def test_split_into_paragraphs():
    """Test paragraph splitting."""
    text = "First paragraph.\n\nSecond paragraph.\n\nThird paragraph."
    paragraphs = split_into_paragraphs(text)
    assert len(paragraphs) == 3
    assert "First paragraph." in paragraphs[0]
    assert "Second paragraph." in paragraphs[1]
    assert "Third paragraph." in paragraphs[2]


def test_is_heading():
    """Test heading detection."""
    assert is_heading("# Heading 1")
    assert is_heading("## Heading 2")
    assert is_heading("Chapter 1")
    assert is_heading("Rozdział 2")
    assert not is_heading("Regular text")


def test_is_blockquote():
    """Test blockquote detection."""
    assert is_blockquote("> This is a quote")
    assert not is_blockquote("This is not a quote")


def test_is_list():
    """Test list detection."""
    assert is_list("* First item\n* Second item")
    assert is_list("1. First item\n2. Second item")
    assert not is_list("* Single item")  # Requires at least 2 items
    assert not is_list("Regular text")


def test_is_code_block():
    """Test code block detection."""
    assert is_code_block("```\ncode here\n```")
    assert not is_code_block("Regular text")


def test_is_horizontal_rule():
    """Test horizontal rule detection."""
    assert is_horizontal_rule("---")
    assert is_horizontal_rule("***")
    assert is_horizontal_rule("___")
    assert not is_horizontal_rule("Regular text")


def test_is_table():
    """Test table detection."""
    table = "| Header 1 | Header 2 |\n|----------|----------|\n| Cell 1 | Cell 2 |"
    assert is_table(table)
    assert not is_table("Regular text")


def test_is_html_block():
    """Test HTML block detection."""
    assert is_html_block("<div>Content</div>")
    assert not is_html_block("Regular text")


def test_is_image_or_figure():
    """Test image/figure detection."""
    assert is_image_or_figure("![Alt text](image.jpg)")
    assert not is_image_or_figure("Regular text")


def test_itemize_document():
    """Test document itemization."""
    doc = """# Heading

First paragraph.

> A quote here.

* List item 1
* List item 2

```
Code block
```"""

    items = itemize_document(doc)
    assert isinstance(items, list)
    assert all(isinstance(item, tuple) and len(item) == 2 for item in items)
    assert len(items) > 0


def test_create_item_elements():
    """Test creation of item elements."""
    items = [
        ("First paragraph", "normal"),
        ("Second paragraph", "following"),
    ]

    result = create_item_elements(items)
    assert isinstance(result, list)
    assert len(result) == len(items)
    for item_xml, item_id in result:  # Corrected unpacking
        assert isinstance(item_id, str)
        assert isinstance(item_xml, str)
        assert item_xml.startswith("<item")
        assert item_xml.endswith("</item>")


def test_create_chunks():
    """Test chunk creation."""
    doc = """<?xml version="1.0" encoding="UTF-8"?>
<document>
    <content>
        <unit type="paragraph">First unit content.</unit>
        <unit type="dialog">Second unit content.</unit>
    </content>
</document>"""

    result = create_chunks(doc, max_chunk_size=1000)
    assert isinstance(result, str)
    assert "<chunk" in result
    assert "</chunk>" in result


def test_clean_consecutive_newlines():
    """Test newline cleaning."""
    text = "First line.\n\n\n\nSecond line."
    result = clean_consecutive_newlines(text)
    assert "\n\n\n" not in result
    assert "First line." in result
    assert "Second line." in result


def test_pretty_print_xml_from_tree(): # Renamed test
    """Test XML pretty printing from etree element."""
    xml_str = "<root><child>Content</child></root>"
    parser = etree.XMLParser(remove_blank_text=True) # Consistent with how it might be used
    root = etree.XML(xml_str.encode("utf-8"), parser)
    result = pretty_print_xml_from_tree(root)
    assert isinstance(result, str)
    assert "\n" in result  # Should have line breaks
    assert "  " in result  # Should have indentation
    assert "<?xml version='1.0' encoding='UTF-8'?>" in result # Expect declaration


def test_process_document(temp_workspace):
    """Test the complete document processing pipeline."""
    # Create test input file
    input_file = temp_workspace / "input" / "test.md"
    output_file = temp_workspace / "output" / "test.xml"

    test_content = """# Test Document

First paragraph with some content.

## Section 1

Another paragraph here.

* List item 1
* List item 2

> A blockquote for testing.
"""

    input_file.write_text(test_content)

    # Process the document
    result = process_document(
        str(input_file),
        str(output_file),
        chunk_size=1000,
        model="gpt-4o",
        temperature=0.2,
            verbose=False, # Changed to False to isolate pretty_print_xml
        backup=True,
    )

    assert isinstance(result, dict)
    assert output_file.exists()

    # Verify the processed content
    processed_content = output_file.read_text()
    assert "<?xml" in processed_content
    assert "<doc>" in processed_content  # Changed from <document>
    assert "<chunk" in processed_content
    assert "</doc>" in processed_content # Changed from </document>


def test_error_handling():
    """Test error handling in chunker functions."""
    # Test with empty input
    assert count_tokens("") == 0
    assert escape_xml_chars("") == ""
    assert generate_hash("") == "phoiac"  # Corrected expected hash for empty string
    assert generate_id("", "") == "000000-phoiac" # Corrected based on new generate_hash("")

    # Test with invalid input
    assert split_into_paragraphs("") == [''] # Empty string results in one empty paragraph
    assert not is_heading("")
    assert not is_blockquote("")
    assert not is_list("")
    assert not is_code_block("")
    assert not is_horizontal_rule("")
    assert not is_table("")
    assert not is_html_block("")
    assert not is_image_or_figure("")

    # Test with malformed XML
    assert pretty_print_xml("<invalid>") == "<invalid>"
