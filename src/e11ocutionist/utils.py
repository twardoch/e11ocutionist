#!/usr/bin/env python3
# this_file: src/e11ocutionist/utils.py
"""
Utility functions for the e11ocutionist package.

This module provides common functionality shared across multiple components.
"""

import re
import hashlib
import datetime
from pathlib import Path
from lxml import etree
import tiktoken
from functools import cache
from loguru import logger


@cache
def get_token_encoder():
    """Get the token encoder for a specific model."""
    return tiktoken.encoding_for_model("gpt-4o")


def count_tokens(text: str) -> int:
    """
    Count the number of tokens in a text string accurately using tiktoken.

    Args:
        text: Text string to count tokens for

    Returns:
        Integer token count
    """
    encoder = get_token_encoder()
    return len(encoder.encode(text))


def escape_xml_chars(text: str) -> str:
    """
    Escape special XML characters to prevent parsing errors.

    Only escapes the five special XML characters while preserving UTF-8 characters.

    Args:
        text: Input text that may contain XML special characters

    Returns:
        Text with XML special characters escaped
    """
    # Replace ampersands first (otherwise you'd double-escape the other entities)
    text = text.replace("&", "&amp;")
    # Replace other special characters
    text = text.replace("<", "&lt;")
    text = text.replace(">", "&gt;")
    text = text.replace('"', "&quot;")
    text = text.replace("'", "&apos;")
    return text


def unescape_xml_chars(text: str) -> str:
    """
    Unescape XML character entities to their original form.

    Args:
        text: Text with XML character entities

    Returns:
        Text with XML character entities unescaped
    """
    # Order is important - must unescape ampersands last to avoid double-unescaping
    text = text.replace("&lt;", "<")
    text = text.replace("&gt;", ">")
    text = text.replace("&quot;", '"')
    text = text.replace("&apos;", "'")
    text = text.replace("&amp;", "&")
    return text


def generate_hash(text: str) -> str:
    """
    Generate a 6-character base36 hash from text.

    Args:
        text: Text to hash

    Returns:
        6-character base36 hash
    """
    sha1 = hashlib.sha1(text.encode("utf-8")).hexdigest()
    # Convert to base36 (alphanumeric lowercase)
    base36 = int(sha1, 16)
    chars = "0123456789abcdefghijklmnopqrstuvwxyz"
    result = ""
    while base36 > 0:
        base36, remainder = divmod(base36, 36)
        result = chars[remainder] + result

    # Ensure we have at least 6 characters
    result = result.zfill(6)
    return result[:6]


def create_backup(file_path: str | Path, stage: str = "") -> Path | None:
    """
    Create a timestamped backup of a file.

    Args:
        file_path: Path to the file to back up
        stage: Optional stage name to include in the backup filename

    Returns:
        Path to the backup file, or None if backup failed
    """
    file_path = Path(file_path)
    if not file_path.exists():
        logger.warning(f"Cannot back up non-existent file: {file_path}")
        return None

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    if stage:
        backup_path = file_path.with_name(
            f"{file_path.stem}_{stage}_{timestamp}{file_path.suffix}"
        )
    else:
        backup_path = file_path.with_name(
            f"{file_path.stem}_{timestamp}{file_path.suffix}"
        )

    try:
        import shutil

        shutil.copy2(file_path, backup_path)
        logger.debug(f"Created backup: {backup_path}")
        return backup_path
    except Exception as e:
        logger.error(f"Failed to create backup: {e}")
        return None


def parse_xml(xml_content: str | bytes, recover: bool = True) -> etree._Element | None:
    """
    Parse XML content safely with error recovery.

    Args:
        xml_content: XML content as string or bytes
        recover: Whether to try to recover from parsing errors

    Returns:
        XML root element, or None if parsing failed
    """
    try:
        # Convert to bytes if string
        if isinstance(xml_content, str):
            xml_bytes = xml_content.encode("utf-8")
        else:
            xml_bytes = xml_content

        parser = etree.XMLParser(
            remove_blank_text=False, recover=recover, encoding="utf-8"
        )
        root = etree.XML(xml_bytes, parser)
        return root
    except Exception as e:
        logger.error(f"XML parsing failed: {e}")
        return None


def serialize_xml(root: etree._Element, pretty_print: bool = True) -> str:
    """
    Serialize an XML element to string.

    Args:
        root: XML root element
        pretty_print: Whether to format the output with indentation

    Returns:
        XML string
    """
    try:
        result = etree.tostring(
            root,
            encoding="utf-8",
            method="xml",
            xml_declaration=True,
            pretty_print=pretty_print,
        ).decode("utf-8")
        return result
    except Exception as e:
        logger.error(f"XML serialization failed: {e}")
        # Return empty string in case of error
        return ""


def pretty_print_xml(xml_str: str) -> str:
    """Format XML string with proper indentation for readability."""
    try:
        root = parse_xml(xml_str)
        if root is not None:
            return serialize_xml(root, pretty_print=True)
        return xml_str
    except Exception:
        # If parsing fails, return the original string
        return xml_str


def clean_consecutive_newlines(content: str) -> str:
    """Replace more than two consecutive newlines with exactly two."""
    return re.sub(r"\n{3,}", "\n\n", content)
