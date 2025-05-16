#!/usr/bin/env python3
# this_file: src/e11ocutionist/neifix.py
"""
NEI fix module for e11ocutionist.

This module provides utilities to transform the text content inside
Named Entity of Interest (NEI) tags according to specific formatting rules.
"""

import re
from typing import Any
from loguru import logger


def transform_nei_content(
    input_file: str,
    output_file: str | None = None,
) -> dict[str, Any]:
    """
    Transform the content of NEI tags in an XML file.

    This function applies specific capitalization and hyphenation rules
    to the text inside NEI tags. For each word in an NEI tag:
    - Single-letter words are preserved (likely parts of acronyms)
    - Hyphens are removed
    - The first letter of each word is preserved as is
    - All subsequent letters are converted to lowercase

    Args:
        input_file: Path to the input XML file
        output_file: Path to save the transformed XML output (if None, returns the content)

    Returns:
        Dictionary with processing statistics
    """
    logger.info(f"Processing NEI tags in {input_file}")

    # Read the input file
    with open(input_file, encoding="utf-8") as f:
        content = f.read()

    # Define a function to process each NEI tag match
    def process_nei_content(match):
        # Get the opening tag and content
        tag_start = match.group(1)
        content = match.group(2)

        # Process the content
        words = content.split()
        processed_words = []

        for word in words:
            # Preserve single-letter words (likely parts of acronyms)
            if len(word) == 1:
                processed_words.append(word)
                continue

            # Remove hyphens
            word_no_hyphens = word.replace("-", "")

            # Keep first letter as is, convert the rest to lowercase
            if len(word_no_hyphens) > 0:
                processed_word = word_no_hyphens[0] + word_no_hyphens[1:].lower()
                processed_words.append(processed_word)
            else:
                # Fallback for edge cases
                processed_words.append(word)

        # Join processed words back together
        processed_content = " ".join(processed_words)

        # Return the full NEI tag with processed content
        return f"{tag_start}{processed_content}</nei>"

    # Find and process all NEI tags
    nei_pattern = r"(<nei[^>]*>)(.*?)</nei>"
    transformed_content = re.sub(
        nei_pattern, process_nei_content, content, flags=re.DOTALL
    )

    # Count the number of NEI tags processed
    nei_count = len(re.findall(nei_pattern, content))
    logger.info(f"Processed {nei_count} NEI tags")

    # Write to output file if specified
    if output_file:
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(transformed_content)
        logger.info(f"Saved transformed content to {output_file}")

    return {
        "input_file": input_file,
        "output_file": output_file if output_file else None,
        "nei_count": nei_count,
        "success": True,
    }
