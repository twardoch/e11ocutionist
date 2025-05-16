"""Test suite for e11ocutionist neifix module."""

import os
import tempfile
from unittest.mock import patch

from e11ocutionist.neifix import transform_nei_content


def test_transform_nei_content():
    """Test transforming NEI tag content in an XML file."""
    # Create test XML content with various NEI scenarios
    test_xml = """
    <doc>
        <item id="i1">This has <nei>JOHN SMITH</nei> as a name.</item>
        <item id="i2">This has <nei>One-Word</nei> hyphenated.</item>
        <item id="i3">This has <nei>A B C</nei> single letters.</item>
        <item id="i4">This has <nei pronunciation="example">MIXED-case TEXT</nei> with attributes.</item>
        <item id="i5">This has <nei>M-C Hammer</nei> with hyphens.</item>
    </doc>
    """

    # Create temporary files for testing
    with tempfile.NamedTemporaryFile(delete=False, suffix=".xml") as input_file:
        input_file.write(test_xml.encode("utf-8"))
        input_path = input_file.name

    output_path = f"{input_path}.out.xml"

    try:
        # Process the file
        with patch("e11ocutionist.neifix.logger"):
            result = transform_nei_content(input_path, output_path)

        # Check the result statistics
        assert result["success"] is True
        assert result["input_file"] == input_path
        assert result["output_file"] == output_path
        assert result["nei_count"] == 5  # Number of NEI tags in the test XML

        # Verify the output file exists
        assert os.path.exists(output_path)

        # Read and check the transformed content
        with open(output_path, encoding="utf-8") as f:
            transformed_content = f.read()

        # Check transformations
        # 1. Case transformation: "JOHN SMITH" -> "John Smith"
        assert "<nei>John Smith</nei>" in transformed_content

        # 2. Hyphen removal: "One-Word" -> "Oneword" (lowercase after first letter)
        assert "<nei>Oneword</nei>" in transformed_content

        # 3. Single letter preservation: "A B C" -> "A B C"
        assert "<nei>A B C</nei>" in transformed_content

        # 4. Mixed case with attributes: "MIXED-case TEXT" -> "Mixedcase Text"
        assert (
            '<nei pronunciation="example">Mixedcase Text</nei>' in transformed_content
        )

        # 5. Complex case with hyphens: "M-C Hammer" -> "Mc Hammer"
        assert "<nei>Mc Hammer</nei>" in transformed_content

    finally:
        # Clean up temporary files
        for path in [input_path, output_path]:
            if os.path.exists(path):
                os.remove(path)


def test_transform_nei_content_return_content():
    """Test transform_nei_content without output file (returns content)."""
    # Create test XML content
    test_xml = """
    <doc>
        <item id="i1">This has <nei>JOHN SMITH</nei> as a name.</item>
    </doc>
    """

    # Create temporary input file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".xml") as input_file:
        input_file.write(test_xml.encode("utf-8"))
        input_path = input_file.name

    try:
        # Process without output file
        with patch("e11ocutionist.neifix.logger"):
            result = transform_nei_content(input_path, None)

        # Check the result
        assert result["success"] is True
        assert result["input_file"] == input_path
        assert result["output_file"] is None
        assert result["nei_count"] == 1

    finally:
        # Clean up temporary file
        if os.path.exists(input_path):
            os.remove(input_path)
