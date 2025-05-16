#!/usr/bin/env python3
# this_file: src/e11ocutionist/elevenlabs_converter.py
"""
ElevenLabs converter for e11ocutionist.

This module converts processed XML files to a text format compatible with ElevenLabs text-to-speech,
with options for processing dialog and handling plaintext input.
"""

import re
from pathlib import Path
from typing import Any
from lxml import etree
from loguru import logger

from .utils import clean_consecutive_newlines, parse_xml, unescape_xml_chars


def extract_text_from_xml(xml_root: etree._Element) -> str:
    """
    Extract text from XML document, processing each item element.

    Args:
        xml_root: XML root element

    Returns:
        Extracted and processed text
    """
    result = []

    for chunk in xml_root.xpath("//chunk"):
        for item in chunk.xpath(".//item"):
            # Get the inner XML of the item
            inner_xml = etree.tostring(item, encoding="utf-8", method="xml").decode(
                "utf-8"
            )
            # Extract just the content between the opening and closing tags
            item_text = re.sub(
                r"^<item[^>]*>(.*)</item>$", r"\1", inner_xml, flags=re.DOTALL
            )

            # Process item text
            # Convert <em>...</em> to "..."
            item_text = re.sub(r"<em>(.*?)</em>", r'"\1"', item_text)
            # Convert <nei new="true">...</nei> to "..."
            item_text = re.sub(
                r'<nei\s+new="true"[^>]*>(.*?)</nei>', r'"\1"', item_text
            )
            # Convert remaining <nei>...</nei> to its content
            item_text = re.sub(r"<nei[^>]*>(.*?)</nei>", r"\1", item_text)
            # Replace <hr/> with <break time="0.5s" />
            item_text = re.sub(r"<hr\s*/?>", r'<break time="0.5s" />', item_text)
            # Strip any remaining HTML/XML tags
            item_text = re.sub(r"<[^>]+>", "", item_text)

            # Unescape HTML entities
            item_text = unescape_xml_chars(item_text)

            result.append(item_text)

    # Join all items with double newlines
    return "\n\n".join(result)


def process_dialog(text: str) -> str:
    """
    Process dialog in text, converting em dashes to appropriate quotation marks.

    Args:
        text: Text with dialog lines

    Returns:
        Text with processed dialog
    """
    # Flag indicating whether we're inside dialog
    in_dialog = False

    # Process the text line by line
    lines = text.split("\n")
    processed_lines = []

    for line in lines:
        if line.strip().startswith("— "):
            # Line starts with em dash and space - it's a dialog line
            in_dialog = True
            line = "<q>" + line[2:]  # Replace em dash with <q> tag

        if in_dialog:
            # Process dialog toggles (— ) within the line
            parts = line.split(" — ")
            if len(parts) > 1:
                # There are dialog toggles in this line
                for i in range(len(parts)):
                    if i > 0:
                        # For toggles, close previous and open new
                        parts[i] = "</q><q>" + parts[i]
                line = " ".join(parts)

            # Check if line should end dialog
            if in_dialog and not any(
                next_line.strip().startswith("— ")
                for next_line in lines[lines.index(line) + 1 : lines.index(line) + 2]
                if next_line.strip()
            ):
                # No dialog in next line, close this one
                line += "</q>"
                in_dialog = False

        processed_lines.append(line)

    result = "\n".join(processed_lines)

    # Final post-processing for dialog
    # Replace <q> at start of line with opening smart quote
    result = re.sub(r"^\s*<q>", '"', result, flags=re.MULTILINE)
    # Replace </q> at end of line with closing smart quote
    result = re.sub(r"</q>\s*$", '"', result, flags=re.MULTILINE)
    # Replace other <q> and </q> tags with special markers for ElevenLabs
    result = re.sub(r"<q>", "; OPEN_Q", result)
    result = re.sub(r"</q>", "CLOSE_Q;", result)

    # Replace break tags with ellipsis
    result = re.sub(r'<break time="[^"]*"\s*/>', "...", result)

    # Normalize consecutive newlines
    result = clean_consecutive_newlines(result)

    # Add some spacing for readability
    result = re.sub(r"\n", "\n\n", result)

    return result


def process_document(
    input_file: str,
    output_file: str,
    dialog: bool = True,
    plaintext: bool = False,
    verbose: bool = False,
) -> dict[str, Any]:
    """
    Convert an XML document to a text format compatible with ElevenLabs.

    Args:
        input_file: Path to the input XML file
        output_file: Path to save the converted text output
        dialog: Whether to process dialog (default: True)
        plaintext: Whether to treat the input as plaintext (default: False)
        verbose: Enable verbose logging

    Returns:
        Dictionary with processing statistics
    """
    if verbose:
        logger.info(f"Processing {input_file} to {output_file}")
        logger.info(f"Dialog processing: {dialog}, Plaintext mode: {plaintext}")

    # Read the input file
    input_content = Path(input_file).read_text(encoding="utf-8")

    # Process the content
    output_text = ""

    if plaintext:
        # Plaintext mode - just process the content directly
        output_text = input_content
        if dialog:
            output_text = process_dialog(output_text)
    else:
        # XML mode - parse the XML and extract the text
        xml_root = parse_xml(input_content)
        if xml_root is None:
            logger.error(f"Failed to parse XML from {input_file}")
            return {"success": False, "error": "XML parsing failed"}

        output_text = extract_text_from_xml(xml_root)
        if dialog:
            output_text = process_dialog(output_text)

    # Write the output file
    Path(output_file).write_text(output_text, encoding="utf-8")

    if verbose:
        logger.info(f"Conversion completed: {output_file}")

    return {
        "input_file": input_file,
        "output_file": output_file,
        "dialog_processed": dialog,
        "plaintext_mode": plaintext,
        "success": True,
    }
