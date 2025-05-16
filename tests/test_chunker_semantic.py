"""Test suite for semantic analysis functions in the chunker module."""

import pytest
from unittest.mock import patch, MagicMock
from e11ocutionist.chunker import (
    semantic_analysis,
    extract_and_parse_llm_response,
    add_unit_tags,
    itemize_document,
    create_chunks,
    DEFAULT_MODEL,
    DEFAULT_TEMPERATURE,
)


@pytest.fixture
def mock_llm_response():
    """Mock LLM response for testing."""
    return """
    I've analyzed the text and identified the semantic units:

    ```json
    [
        ["This is the first paragraph.", "normal"],
        ["This is the second paragraph, which continues the thought.", "normal"],
        ["## New Section Heading", "heading"],
        ["This is content under the new section.", "normal"],
        [
            "This is more content that belongs with the previous paragraph.",
            "preceding"
        ]
    ]
    ```
    """


def test_itemize_document():
    """Test converting a document into itemized paragraphs."""
    test_doc = """This is the first paragraph.

This is the second paragraph.

# Heading

This is content under the heading.

* List item 1
* List item 2

> This is a blockquote."""

    items = itemize_document(test_doc)

    # Check the correct number of items were created
    assert len(items) == 5

    # Check item types
    assert items[0][0].strip() == "This is the first paragraph."
    assert items[1][0].strip() == "This is the second paragraph."
    assert items[2][0].strip() == "# Heading"
    assert items[3][0].strip() == "This is content under the heading."
    assert "* List item 1" in items[4][0]


@patch("e11ocutionist.chunker.litellm.completion")
def test_semantic_analysis(mock_completion, mock_llm_response):
    """Test semantic analysis using mock LLM response."""
    # Configure the mock
    mock_completion.return_value = MagicMock(
        choices=[MagicMock(message=MagicMock(content=mock_llm_response))]
    )

    doc_text = """This is the first paragraph.
    
This is the second paragraph, which continues the thought.

## New Section Heading

This is content under the new section.

This is more content that belongs with the previous paragraph."""

    result = semantic_analysis(doc_text, DEFAULT_MODEL, DEFAULT_TEMPERATURE)

    # Verify the LLM was called
    mock_completion.assert_called_once()

    # Check the result
    assert len(result) == 5
    assert result[0][0] == "This is the first paragraph."
    assert result[0][1] == "normal"
    assert result[2][0] == "## New Section Heading"
    assert result[2][1] == "heading"
    long_text = "This is more content that belongs with the previous paragraph."
    assert result[4][0] == long_text
    assert result[4][1] == "preceding"


def test_extract_and_parse_llm_response(mock_llm_response):
    """Test parsing of LLM response."""
    llm_response = MagicMock()
    llm_response.choices = [MagicMock()]
    llm_response.choices[0].message.content = mock_llm_response

    result = extract_and_parse_llm_response(llm_response)

    # Check the correct data was extracted
    assert len(result) == 5
    assert result[0][0] == "This is the first paragraph."
    assert result[0][1] == "normal"
    long_text = "This is more content that belongs with the previous paragraph."
    assert result[4][0] == long_text
    assert result[4][1] == "preceding"


def test_add_unit_tags():
    """Test adding unit tags to document."""
    itemized_doc = """<doc>
    <item id="000000-abc123">This is paragraph one.</item>
    <item id="abc123-def456">This is paragraph two.</item>
    <item id="def456-ghi789">This is a heading.</item>
    <item id="ghi789-jkl012">This follows the heading.</item>
    </doc>"""

    semantic_boundaries = [
        ("This is paragraph one.", "normal"),
        ("This is paragraph two.", "normal"),
        ("This is a heading.", "heading"),
        ("This follows the heading.", "normal"),
    ]

    result = add_unit_tags(itemized_doc, semantic_boundaries)

    # Check that unit tags were added correctly
    assert '<unit type="normal">' in result
    assert '<unit type="heading">' in result
    assert "</unit>" in result

    # Check that items are wrapped in unit tags
    item_pattern = (
        '<unit type="normal"><item id="000000-abc123">This is paragraph one.</item>'
    )
    heading_pattern = (
        '<unit type="heading"><item id="def456-ghi789">This is a heading.</item>'
    )
    assert item_pattern in result
    assert heading_pattern in result


@patch("e11ocutionist.chunker.add_unit_tags")
@patch("e11ocutionist.chunker.semantic_analysis")
def test_create_chunks(mock_semantic_analysis, mock_add_unit_tags):
    """Test creating chunks from document with units."""
    # Set up mocks
    mock_semantic_analysis.return_value = [
        ("This is paragraph one.", "normal"),
        ("This is paragraph two.", "normal"),
        ("This is a heading.", "heading"),
        ("This follows the heading.", "normal"),
    ]

    # Create a mock document with units
    doc_with_units = """<doc>
    <unit type="normal">
    <item id="000000-abc123">This is paragraph one.</item>
    <item id="abc123-def456">This is paragraph two.</item>
    </unit>
    <unit type="heading">
    <item id="def456-ghi789">This is a heading.</item>
    </unit>
    <unit type="normal">
    <item id="ghi789-jkl012">This follows the heading.</item>
    </unit>
    </doc>"""

    mock_add_unit_tags.return_value = doc_with_units

    # Create a document that would be suitable for the mock result above
    sample_doc = """<doc>
    <item id="000000-abc123">This is paragraph one.</item>
    <item id="abc123-def456">This is paragraph two.</item>
    <item id="def456-ghi789">This is a heading.</item>
    <item id="ghi789-jkl012">This follows the heading.</item>
    </doc>"""

    # Call the create_chunks function with a small max_chunk_size
    result = create_chunks(sample_doc, 100)

    # Verify the result contains chunk tags
    assert "<chunks>" in result
    assert "<chunk id=" in result
    assert "</chunk>" in result
    assert "</chunks>" in result
