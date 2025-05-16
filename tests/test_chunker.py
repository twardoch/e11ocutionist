"""Test suite for e11ocutionist chunker module."""

from e11ocutionist.chunker import (
    split_into_paragraphs,
    is_heading,
    is_blockquote,
    is_list,
    is_code_block,
    is_horizontal_rule,
    is_table,
    is_html_block,
    is_image_or_figure,
    count_tokens,
    generate_hash,
    generate_id,
    escape_xml_chars,
)


def test_split_into_paragraphs():
    """Test splitting text into paragraphs."""
    test_text = "This is paragraph 1.\n\nThis is paragraph 2.\n\nThis is paragraph 3."

    paragraphs = split_into_paragraphs(test_text)

    assert len(paragraphs) == 3
    assert "This is paragraph 1." in paragraphs[0]
    assert "This is paragraph 2." in paragraphs[1]
    assert "This is paragraph 3." in paragraphs[2]

    # Test with Windows line endings
    windows_text = "Paragraph 1.\r\n\r\nParagraph 2."
    windows_paragraphs = split_into_paragraphs(windows_text)
    assert len(windows_paragraphs) == 2


def test_paragraph_classification():
    """Test functions that classify paragraphs by type."""
    # Test heading detection
    assert is_heading("# Heading 1") is True
    assert is_heading("## Heading 2") is True
    assert is_heading("Regular paragraph") is False
    assert is_heading("Chapter 1: Introduction") is True
    assert is_heading("Rozdział 2. Wstęp") is True

    # Test blockquote detection
    assert is_blockquote("> This is a blockquote") is True
    assert is_blockquote("This is not a blockquote") is False

    # Test list detection
    assert is_list("* Item 1\n* Item 2") is True
    assert is_list("- Item 1\n- Item 2\n- Item 3") is True
    assert is_list("1. First item\n2. Second item") is True
    # Not considered a list (needs 2+ items)
    assert is_list("* Single item") is False
    assert is_list("Regular paragraph") is False

    # Test code block detection
    code_block = "```python\ndef hello():\n    print('Hello')\n```"
    assert is_code_block(code_block) is True
    assert is_code_block("Regular paragraph") is False

    # Test horizontal rule detection
    assert is_horizontal_rule("---") is True
    assert is_horizontal_rule("***") is True
    assert is_horizontal_rule("___") is True
    assert is_horizontal_rule("Regular paragraph") is False

    # Test table detection
    table_text = "| Header1 | Header2 |\n|---------|---------|"
    assert is_table(table_text) is True
    assert is_table("Regular paragraph") is False

    # Test HTML block detection
    assert is_html_block("<div>This is HTML</div>") is True
    assert is_html_block("Regular paragraph") is False

    # Test image/figure detection
    assert is_image_or_figure("![Alt text](image.jpg)") is True
    assert is_image_or_figure("Regular paragraph") is False


def test_count_tokens():
    """Test token counting functionality."""
    # Simple test with predictable token counts
    text1 = "This is a simple sentence."
    assert count_tokens(text1) > 0

    # Check that longer text has more tokens
    text2 = "This is a simple sentence. " * 10
    assert count_tokens(text2) > count_tokens(text1)


def test_generate_hash():
    """Test hash generation from text."""
    # Test basic hash generation
    hash1 = generate_hash("test text")
    assert isinstance(hash1, str)
    assert len(hash1) == 6  # Should be 6 characters

    # Test consistent hashing
    hash2 = generate_hash("test text")
    assert hash1 == hash2  # Same input should give same hash

    # Test different inputs give different hashes
    hash3 = generate_hash("different text")
    assert hash1 != hash3  # Different input should give different hash


def test_generate_id():
    """Test ID generation for items."""
    # Test first item ID generation
    first_id = generate_id("", "First item content")
    assert first_id.startswith("000000-")
    assert len(first_id) == 13  # "000000-" + 6 chars

    # Test subsequent item ID generation
    second_id = generate_id(first_id, "Second item content")
    # Second ID should start with first ID's suffix
    prefix = first_id.split("-")[1]
    assert second_id.startswith(f"{prefix}-")
    assert len(second_id) == 13


def test_escape_xml_chars():
    """Test escaping of XML special characters."""
    # Test all special characters
    input_text = "Text with <tags> & \"quotes\" and 'apostrophes'"
    expected = (
        "Text with &lt;tags&gt; &amp; &quot;quotes&quot; and &apos;apostrophes&apos;"
    )
    assert escape_xml_chars(input_text) == expected

    # Test text with no special characters
    normal_text = "Normal text without special chars"
    assert escape_xml_chars(normal_text) == normal_text
