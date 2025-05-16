#!/usr/bin/env -S uv run -s
# /// script
# dependencies = ["fire", "litellm>=1.67.2", "tiktoken", "lxml", "backoff", "python-dotenv", "loguru"]
# ///
# this_file: malmo_chunker.py

import fire
import re
import hashlib
import os
import backoff
import tiktoken
import datetime
from pathlib import Path
from lxml import etree
from litellm import completion
from litellm._logging import _turn_on_debug
from dotenv import load_dotenv
from loguru import logger
from functools import cache

# Load environment variables from .env file
load_dotenv()

# Global variables
DEFAULT_MODEL = "openrouter/openai/gpt-4.1"
FALLBACK_MODEL = "openrouter/google/gemini-2.5-pro-preview-03-25"

DEFAULT_CHUNK_SIZE = 12288
DEFAULT_TEMPERATURE = 1.0

# Check if API key is set
if not os.getenv("OPENROUTER_API_KEY"):
    logger.warning("No API key found in environment variables. API calls may fail.")


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


@cache
def get_token_encoder():
    """Get the token encoder."""
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


def replace_tok_placeholder(text: str, placeholder: str = 'tok="____"') -> str:
    """
    Replace token count placeholder with actual count.

    This function accurately counts tokens in XML elements and ensures
    the tok attribute reflects the actual token count.
    """
    if placeholder not in text:
        return text

    # Temporarily replace placeholder with a unique string unlikely to appear in text
    temp_placeholder = "-987"
    text_with_temp = text.replace(placeholder, f'tok="{temp_placeholder}"')

    # Count actual tokens
    token_count = count_tokens(text_with_temp.replace(temp_placeholder, "0"))

    # Replace temp placeholder with actual count
    return text_with_temp.replace(temp_placeholder, str(token_count))


def generate_hash(text: str) -> str:
    """Generate a 6-character base36 hash from text."""
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


def generate_id(prev_id: str, content: str) -> str:
    """Generate a unique ID for an item based on previous ID and content."""
    if not prev_id:
        # First item
        return f"000000-{generate_hash(content)}"
    else:
        # Subsequent items
        prev_suffix = prev_id.split("-")[-1]
        return f"{prev_suffix}-{generate_hash(content)}"


def split_into_paragraphs(text: str) -> list[str]:
    """Split text into paragraphs based on blank lines while preserving all whitespace."""
    # Remove Windows line endings
    text = text.replace("\r\n", "\n")
    # Split on blank lines (one or more newlines) and capture the delimiters
    parts = re.split(r"(\n\s*\n)", text)
    # Join each paragraph with its delimiter and keep everything, no stripping
    return ["".join(parts[i : i + 2]) for i in range(0, len(parts), 2)]


def is_heading(paragraph: str) -> bool:
    """Check if paragraph is a markdown heading or chapter heading in Polish/English."""
    # Check for markdown heading format
    if bool(re.match(r"^#{1,6}\s+", paragraph)):
        return True

    # Check for chapter headings in Polish and English
    if bool(re.search(r'(?i)^[„"]?(?:rozdział|chapter)\s+\w+', paragraph.strip())):
        return True

    return False


def is_blockquote(paragraph: str) -> bool:
    """Check if paragraph is a markdown blockquote."""
    return paragraph.startswith("> ")


def is_list(paragraph: str) -> bool:
    """Check if paragraph is a markdown list of at least two items."""
    # Strip trailing blank line delimiters
    text = paragraph.rstrip("\n")
    lines = text.split("\n")
    bullet_pattern = re.compile(r"^[\*\-\+]\s+")
    number_pattern = re.compile(r"^\d+\.\s+")
    # Count lines that look like list items
    bullet_count = sum(
        1
        for line in lines
        if bullet_pattern.match(line.strip()) or number_pattern.match(line.strip())
    )
    # Treat as list only if two or more items
    return bullet_count >= 2


def is_code_block(paragraph: str) -> bool:
    """Check if paragraph is a markdown code block."""
    return paragraph.startswith("```") and paragraph.endswith("```")


def is_horizontal_rule(paragraph: str) -> bool:
    """Check if paragraph is a markdown horizontal rule."""
    return bool(re.match(r"^[\*\-_]{3,}\s*$", paragraph))


def is_table(paragraph: str) -> bool:
    """Check if paragraph is a markdown table."""
    lines = paragraph.split("\n")
    if len(lines) < 2:
        return False
    # Check for pipe character and separator line
    return "|" in lines[0] and bool(re.match(r"^[\|\s\-:]+$", lines[1].strip()))


def is_html_block(paragraph: str) -> bool:
    """Check if paragraph is an HTML block."""
    return paragraph.startswith("<") and paragraph.endswith(">")


def is_image_or_figure(paragraph: str) -> bool:
    """Check if paragraph is a markdown image or figure."""
    return bool(re.match(r"!\[.*\]\(.*\)", paragraph))


def itemize_document(doc_text: str) -> list[tuple[str, str]]:
    """
    Split document into items according to rules.

    Returns:
        List of (item_text, attachment_type) where attachment_type is:
        - "following" for elements that attach to the following item
        - "preceding" for elements that attach to the preceding item
        - "normal" for regular paragraphs
    """
    paragraphs = split_into_paragraphs(doc_text)
    items = []

    for paragraph in paragraphs:
        # Skip empty paragraphs
        if not paragraph.strip():
            continue

        if is_heading(paragraph):
            # Treat chapter and markdown headings as standalone items, not attached to following text
            items.append((paragraph, "normal"))
        elif (
            is_blockquote(paragraph)
            or is_list(paragraph)
            or is_code_block(paragraph)
            or is_table(paragraph)
            or is_horizontal_rule(paragraph)
            or is_image_or_figure(paragraph)
            or is_html_block(paragraph)
        ):
            items.append((paragraph, "preceding"))
        else:
            items.append((paragraph, "normal"))

    return items


def create_item_elements(items: list[tuple[str, str]]) -> list[tuple[str, str]]:
    """
    Create XML item elements with appropriate attachments.

    Returns:
        List of (item_xml, item_id) tuples
    """
    result = []
    prev_id = ""
    buffer = []
    buffer_type = None

    # If document starts with a "following" element, create an empty item
    if items and items[0][1] == "following":
        buffer.append(items[0][0])
        buffer_type = "following"
        items = items[1:]

    for item_text, attachment_type in items:
        # Escape special characters in item_text to avoid XML parsing errors
        item_text = escape_xml_chars(item_text)

        if attachment_type == "following":
            if buffer and buffer_type != "following":
                # Finalize previous item and start a new buffer
                content = "\n\n".join(buffer)
                item_id = generate_id(prev_id, content)
                item_xml = f'<item xml:space="preserve" tok="____" id="{item_id}">\n{content}\n</item>'
                result.append((item_xml, item_id))
                prev_id = item_id
                buffer = [item_text]
                buffer_type = "following"
            else:
                # Add to existing "following" buffer
                buffer.append(item_text)
        elif attachment_type == "preceding":
            if buffer:
                # Add to existing buffer
                buffer.append(item_text)
            else:
                # Create a new buffer
                buffer = [item_text]
                buffer_type = "preceding"
        elif buffer and buffer_type == "following":
            # Add to existing "following" buffer
            buffer.append(item_text)
            # Finalize this item
            content = "\n\n".join(buffer)
            item_id = generate_id(prev_id, content)
            item_xml = f'<item xml:space="preserve" tok="____" id="{item_id}">\n{content}\n</item>'
            result.append((item_xml, item_id))
            prev_id = item_id
            buffer = []
            buffer_type = None
        elif buffer and buffer_type == "preceding":
            # Add normal text to preceding elements and finalize
            buffer.append(item_text)
            content = "\n\n".join(buffer)
            item_id = generate_id(prev_id, content)
            item_xml = f'<item xml:space="preserve" tok="____" id="{item_id}">\n{content}\n</item>'
            result.append((item_xml, item_id))
            prev_id = item_id
            buffer = []
            buffer_type = None
        else:
            # Just a normal paragraph by itself
            content = item_text
            item_id = generate_id(prev_id, content)
            item_xml = f'<item xml:space="preserve" tok="____" id="{item_id}">\n{content}\n</item>'
            result.append((item_xml, item_id))
            prev_id = item_id

    # Process any remaining buffer
    if buffer:
        content = "\n\n".join(buffer)
        item_id = generate_id(prev_id, content)
        item_xml = (
            f'<item xml:space="preserve" tok="____" id="{item_id}">\n{content}\n</item>'
        )
        result.append((item_xml, item_id))

    return result


def update_token_counts(xml_text: str) -> str:
    """
    Update all tok attributes with accurate token counts.

    Processes the XML document to replace token placeholders with
    actual token counts based on the element content.

    Args:
        xml_text: XML document with tok="____" placeholders

    Returns:
        XML document with accurate token counts
    """
    # Try parsing the XML to maintain structure
    try:
        parser = etree.XMLParser(
            remove_blank_text=False, recover=True, encoding="utf-8"
        )
        root = etree.XML(xml_text.encode("utf-8"), parser)

        # Process each element with a tok attribute in a bottom-up order
        # This ensures nested elements are counted first
        for element in reversed(list(root.xpath("//*[@tok]"))):
            # Get the string representation of this element
            element_xml = etree.tostring(
                element, encoding="utf-8", method="xml"
            ).decode("utf-8")

            # Count tokens for the element with temp placeholder
            element_text = element_xml.replace('tok="____"', 'tok="0"')
            token_count = count_tokens(element_text)

            # Update the element's tok attribute
            element.set("tok", str(token_count))

        # Convert back to string with proper formatting
        result = etree.tostring(root, encoding="utf-8", method="xml").decode("utf-8")
        return result
    except Exception as e:
        logger.warning(f"XML parsing for token counting failed: {e}")
        # Fall back to regex replacement method
        while 'tok="____"' in xml_text:
            xml_text = replace_tok_placeholder(xml_text)
        return xml_text


@backoff.on_exception(backoff.expo, Exception, max_tries=3)
def semantic_analysis(
    doc_text: str, model: str, temperature: float
) -> list[tuple[str, str]]:
    """
    Use LLM to identify semantic boundaries.

    Returns:
        List of (item_id, boundary_type) tuples
    """
    # Check if API key is available
    if not os.getenv("OPENROUTER_API_KEY"):
        logger.warning("No API key available for LLM requests")
        # Return default semantic boundaries - first item gets a unit
        items_match = re.findall(r'<item\s+[^>]*?id="([^"]*)"', doc_text)
        if items_match:
            return [(items_match[0], "unit")]
        return []

    # Calculate total document length in tokens
    doc_token_count = count_tokens(doc_text)
    logger.info(f"Document total token count: {doc_token_count}")

    # Calculate minimum number of chunks based on document size and max chunk size
    min_chunks = max(
        1,
        (doc_token_count // DEFAULT_CHUNK_SIZE)
        + (1 if doc_token_count % DEFAULT_CHUNK_SIZE > 0 else 0),
    )

    # Calculate minimum number of units (5 * minimum chunks)
    min_units = 5 * min_chunks

    logger.info(
        f"Minimum chunks needed: {min_chunks}, minimum units required: {min_units}"
    )

    # Extract all item IDs from the document for validation
    all_item_ids = re.findall(r'<item\s+[^>]*?id="([^"]*)"', doc_text)

    prompt = f"""
You are analyzing a document with semantic markup. The document has been split into items with unique IDs.
Your task is to identify meaningful semantic boundaries within this document.

Follow these strict rules:
1. Identify the beginning of chapters, scenes, and other significant semantic units
2. Mark EXACTLY {min_units} OR MORE semantic units - this is a HARD REQUIREMENT
3. Always mark the first item as a "unit" if it's not a chapter or scene
4. Chapters are typically titled with patterns like:
   - "Chapter X" (English)
   - "Rozdział X" (Polish)
   - X can be a number (1, 2, 3) or a word (one, two, three, pierwszy, drugi, trzeci, czwarty, piąty, etc.)
5. Scenes are sections that represent a distinct setting, time, or viewpoint.
6. Units are other significant semantic divisions: sections, passages with a common theme or author or form (eg. poem vs prose). They may be separated by a line consisting only of non-letter characters like `●` or `***` or similar.

This document requires AT LEAST {min_units} semantic units due to its length ({doc_token_count} tokens).
Your analysis must identify at least this minimum number or your response will be rejected.

For each item where a new semantic unit begins, return the item's ID and the type of boundary:
- "chapter" - For major divisions like chapters
- "scene" - For scene breaks within chapters
- "unit" - For other significant semantic units

Return your response as a line-by-line list in the format:
item_id: boundary_type

Here is the document:

{doc_text}
"""

    # Function to extract LLM response
    def extract_and_parse_llm_response(response) -> list[tuple[str, str]]:
        try:
            result_text = ""
            if hasattr(response, "choices") and response.choices:
                try:
                    # Try to get content from message
                    result_text = response.choices[0].message.content
                except (AttributeError, IndexError):
                    try:
                        # Try to access text attribute safely without triggering type errors
                        choice = response.choices[0]
                        if hasattr(choice, "text"):
                            result_text = choice.text
                        else:
                            # Fall back to string representation
                            result_text = str(choice)
                    except (AttributeError, IndexError):
                        # Fall back to string representation
                        result_text = str(response.choices[0])
            elif hasattr(response, "content") and response.content:
                result_text = response.content
            else:
                # Attempt to convert the entire response to string
                result_text = str(response)

            # Ensure we have a string, not None
            result_text = result_text or ""

            logger.debug(f"LLM response: {result_text}")
            logger.info(f"LLM response: {result_text}")

            # Parse the response
            semantic_boundaries = []
            for line in result_text.strip().split("\n"):
                line = line.strip()
                if not line or ":" not in line:
                    continue

                parts = line.split(":", 1)
                if len(parts) != 2:
                    continue

                item_id = parts[0].strip().lower()
                boundary_type = parts[1].strip().lower()

                # Validate boundary type
                if boundary_type not in ["chapter", "scene", "unit"]:
                    logger.warning(
                        f"Unrecognized boundary type: {boundary_type} for ID: {item_id}"
                    )
                    continue

                semantic_boundaries.append((item_id, boundary_type))

            # Deduplicate while preserving first occurrence
            seen_ids = set()
            unique_boundaries = []
            for item_id, boundary_type in semantic_boundaries:
                if item_id not in seen_ids:
                    unique_boundaries.append((item_id, boundary_type))
                    seen_ids.add(item_id)

            return unique_boundaries
        except Exception as e:
            logger.error(f"Failed to extract and parse LLM response: {e}")
            return []

    try:
        logger.info(f"Sending request to LLM model: {model}")
        response = completion(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
        )

        # Extract and parse the LLM response
        unique_boundaries = extract_and_parse_llm_response(response)

    except Exception as e:
        logger.error(f"Error in semantic analysis with model {model}: {e}")

        # Try with fallback model if different from the current model
        if model != FALLBACK_MODEL:
            try:
                logger.warning(f"Attempting to use fallback model: {FALLBACK_MODEL}")
                fallback_response = completion(
                    model=FALLBACK_MODEL,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=temperature,
                )

                # Extract and parse the fallback response
                unique_boundaries = extract_and_parse_llm_response(fallback_response)

                if unique_boundaries:
                    logger.info(
                        f"Successfully analyzed with fallback model: {FALLBACK_MODEL}"
                    )
                else:
                    logger.warning("Fallback model did not return valid boundaries")
                    # Fall back to heuristic approach
                    if all_item_ids:
                        return [(all_item_ids[0], "unit")]
                    return []

            except Exception as fallback_error:
                logger.error(f"Fallback model also failed: {fallback_error}")
                # Fall back to heuristic approach
                if all_item_ids:
                    return [(all_item_ids[0], "unit")]
                return []
        else:
            # If model and fallback are the same, just fall back to heuristic approach
            if all_item_ids:
                return [(all_item_ids[0], "unit")]
            return []

    # Check if we have enough boundaries
    if len(unique_boundaries) < min_units:
        logger.warning(
            f"LLM identified only {len(unique_boundaries)} units, which is less than the required {min_units}. Will attempt to add more."
        )

        # Add additional units by choosing items at regular intervals
        available_ids = [
            item_id
            for item_id in all_item_ids
            if item_id not in {id for id, _ in unique_boundaries}
        ]

        if available_ids:
            # Calculate how many more units we need
            needed = min_units - len(unique_boundaries)

            # Choose items at regular intervals
            step = max(1, len(available_ids) // needed)
            additional_ids = [
                available_ids[i] for i in range(0, len(available_ids), step)
            ][:needed]

            # Add as units
            for item_id in additional_ids:
                unique_boundaries.append((item_id, "unit"))

            logger.info(
                f"Added {len(additional_ids)} additional units to meet minimum requirement"
            )

    # If we still don't have any boundaries, fall back to the default
    if not unique_boundaries and all_item_ids:
        return [(all_item_ids[0], "unit")]

    return unique_boundaries


def add_unit_tags(itemized_doc: str, semantic_boundaries: list[tuple[str, str]]) -> str:
    """
    Add unit tags to the document based on semantic boundaries.

    Args:
        itemized_doc: XML document with items
        semantic_boundaries: List of (item_id, boundary_type) tuples

    Returns:
        Document with unit tags added
    """
    # Convert to dict for easy lookup
    boundaries_dict = dict(semantic_boundaries)

    # Parse the document
    parser = etree.XMLParser(remove_blank_text=False, recover=True, encoding="utf-8")
    try:
        doc_root = etree.XML(itemized_doc.encode("utf-8"), parser)
    except Exception as e:
        logger.error(f"Error parsing document: {e}")
        # Fall back to regex-based approach
        return add_unit_tags_regex(itemized_doc, semantic_boundaries)

    # Get all item elements
    items = doc_root.xpath("//item")

    # If we have no semantic boundaries, use heuristics to create them
    if not semantic_boundaries:
        # Detect chapter boundaries based on text patterns
        for item in items:
            item_text = (
                etree.tostring(item, encoding="utf-8", method="text")
                .decode("utf-8")
                .strip()
            )
            # Match both numeric and word-based chapter numbers in Polish and English
            if re.search(r'(?i)^[„"]?(?:rozdział|chapter)\s+\w+', item_text):
                item_id = item.get("id", "")
                if item_id:
                    boundaries_dict[item_id] = "chapter"
            elif item.getprevious() is None or item.getparent().index(item) == 0:
                # First item gets a unit boundary
                item_id = item.get("id", "")
                if item_id:
                    boundaries_dict[item_id] = "unit"

    # Ensure we have at least one unit
    if not boundaries_dict and items:
        # Get first item's ID
        first_item_id = items[0].get("id", "")
        if first_item_id:
            boundaries_dict[first_item_id] = "unit"

    # Add unit tags
    current_unit = None
    item_parent_map = {child: parent for parent in doc_root.iter() for child in parent}

    # Group items into units based on boundaries
    for _i, item in enumerate(items):
        item_id = item.get("id", "")

        # Check if this item starts a new unit
        if item_id in boundaries_dict:
            # Close previous unit if exists
            if current_unit is not None:
                # Already in a unit, need to close and open new
                parent = item_parent_map[item]
                unit_idx = parent.index(item)

                # Create new unit element with proper semantic type
                new_unit = etree.Element("unit")
                new_unit.set("type", boundaries_dict[item_id])
                new_unit.set("tok", "____")  # Placeholder

                # Insert new unit before item
                parent.insert(unit_idx, new_unit)

                # Move item to new unit
                parent.remove(item)
                new_unit.append(item)

                current_unit = new_unit
            else:
                # First unit in document
                parent = item_parent_map[item]
                unit_idx = parent.index(item)

                # Create unit element
                unit = etree.Element("unit")
                unit.set("type", boundaries_dict[item_id])
                unit.set("tok", "____")  # Placeholder

                # Insert unit before item
                parent.insert(unit_idx, unit)

                # Move item to unit
                parent.remove(item)
                unit.append(item)

                current_unit = unit
        elif current_unit is not None:
            # Already in a unit, move this item to current unit
            parent = item_parent_map[item]
            parent.remove(item)
            current_unit.append(item)

    # Convert back to string
    result = etree.tostring(doc_root, encoding="utf-8", method="xml").decode("utf-8")

    # Update token counts
    result = update_token_counts(result)

    return result


def add_unit_tags_regex(
    itemized_doc: str, semantic_boundaries: list[tuple[str, str]]
) -> str:
    """
    Fallback method to add unit tags using regex.

    Args:
        itemized_doc: XML document with items
        semantic_boundaries: List of (item_id, boundary_type) tuples

    Returns:
        Document with unit tags added
    """
    # Convert to dict for easy lookup
    boundaries_dict = dict(semantic_boundaries)

    # Find all item IDs
    item_matches = list(re.finditer(r'<item tok="[^"]*" id="([^"]*)">', itemized_doc))

    if not item_matches:
        return itemized_doc

    # Process in reverse order to avoid invalidating match positions
    result = itemized_doc
    open_unit = False

    # Ensure first item has a unit if not already defined
    first_item_id = item_matches[0].group(1) if item_matches else None
    if first_item_id and first_item_id not in boundaries_dict:
        boundaries_dict[first_item_id] = "unit"

    for i in range(len(item_matches) - 1, -1, -1):
        match = item_matches[i]
        item_id = match.group(1)

        if item_id in boundaries_dict:
            # Add unit close/open tags
            if open_unit:
                # Close previous unit and open new one
                unit_tag = (
                    f'</unit>\n<unit type="{boundaries_dict[item_id]}" tok="____">\n'
                )
            else:
                # Just open a unit
                unit_tag = f'<unit type="{boundaries_dict[item_id]}" tok="____">\n'
                open_unit = True

            result = result[: match.start()] + unit_tag + result[match.start() :]

    # Close the last unit at document end
    if open_unit:
        result += "\n</unit>"

    # Ensure document starts with a unit if needed
    if not result.strip().startswith("<unit"):
        first_item_pos = item_matches[0].start() if item_matches else 0
        result = (
            result[:first_item_pos]
            + '<unit type="unit" tok="____">\n'
            + result[first_item_pos:]
        )
        result += "\n</unit>"

    # Update token counts
    result = update_token_counts(result)

    return result


def create_chunks(doc_with_units: str, max_chunk_size: int) -> str:
    """
    Create chunks based on token limit and unit boundaries.

    Args:
        doc_with_units: XML document with items and units
        max_chunk_size: Maximum chunk size in tokens

    Returns:
        Document with chunk tags added
    """
    try:
        # Parse document
        parser = etree.XMLParser(
            remove_blank_text=False, recover=True, encoding="utf-8"
        )
        doc_root = etree.XML(doc_with_units.encode("utf-8"), parser)

        # Get all unit elements
        units = doc_root.xpath("//unit")

        # Verify that all items are inside a unit
        all_items = doc_root.xpath("//item")
        items_in_units = doc_root.xpath("//unit//item")

        if len(all_items) != len(items_in_units):
            logger.warning(
                f"Found {len(all_items) - len(items_in_units)} items not in units. Adding them to a default unit."
            )
            # Find all items not in units and add them to a default unit
            for item in all_items:
                # Check if this item has a unit parent
                parent = item.getparent()
                if parent is not None and parent.tag != "unit":
                    # Create a new unit
                    default_unit = etree.Element("unit")
                    default_unit.set("type", "unit")
                    default_unit.set("tok", "____")

                    # Replace the item with the unit containing the item
                    item_idx = parent.index(item)
                    parent.remove(item)
                    default_unit.append(item)
                    parent.insert(item_idx, default_unit)

        # Get updated list of units
        units = doc_root.xpath("//unit")

        if not units:
            # No units, just wrap everything in one chunk
            chunk = etree.Element("chunk")
            chunk.set("id", "c0")
            chunk.set("tok", "____")

            # Move all direct children to the chunk
            for child in list(doc_root):
                doc_root.remove(child)
                chunk.append(child)

            doc_root.append(chunk)
        else:
            # Process units into chunks
            current_chunk = None
            current_chunk_id = 0
            current_chunk_tokens = 0

            # Find units of type "chapter" to ensure they start new chunks
            for unit in units:
                # Calculate unit tokens (use actual token count instead of placeholder)
                unit_tokens = int(unit.get("tok", "0"))
                is_chapter = unit.get("type") == "chapter"

                # Start a new chunk if:
                # 1. We don't have a current chunk
                # 2. This unit is a chapter (always starts a new chunk)
                # 3. Adding this unit would exceed the max chunk size
                if (
                    current_chunk is None
                    or is_chapter
                    or current_chunk_tokens + unit_tokens > max_chunk_size
                ):
                    # Create new chunk
                    current_chunk = etree.Element("chunk")
                    current_chunk.set("id", f"c{current_chunk_id}")
                    current_chunk.set("tok", "____")
                    current_chunk_id += 1

                    # Get parent of the unit
                    parent = unit.getparent()
                    if parent is not None:
                        # Move unit to chunk
                        idx = parent.index(unit)
                        parent.remove(unit)
                        current_chunk.append(unit)

                        # Add chunk to document at the same position
                        parent.insert(idx, current_chunk)
                    else:
                        # Unit is at root level
                        doc_root.remove(unit)
                        current_chunk.append(unit)
                        doc_root.append(current_chunk)

                    current_chunk_tokens = unit_tokens
                else:
                    # Add to current chunk
                    parent = unit.getparent()
                    if parent is not None:
                        parent.remove(unit)
                        current_chunk.append(unit)
                    current_chunk_tokens += unit_tokens

        # Convert back to string
        result = etree.tostring(doc_root, encoding="utf-8", method="xml").decode(
            "utf-8"
        )

        # Update token counts
        result = update_token_counts(result)

        return result

    except Exception as e:
        logger.error(f"Error creating chunks: {e}")
        # Fall back to regex approach
        return create_chunks_regex(doc_with_units, max_chunk_size)


def handle_oversized_unit(
    doc_root, unit, max_chunk_size: int, chunk_id_start: int
) -> None:
    """
    Handle a unit that exceeds the maximum chunk size by splitting it.

    Args:
        doc_root: XML document root
        unit: The oversized unit element
        max_chunk_size: Maximum chunk size in tokens
        chunk_id_start: Starting chunk ID number
    """
    # Get unit's parent
    parent = unit.getparent()
    if parent is None:
        parent = doc_root

    # Get unit's position
    unit_pos = parent.index(unit) if unit in parent else -1

    # Remove unit from parent
    if unit_pos >= 0:
        parent.remove(unit)

    # Get items in unit
    items = list(unit.xpath(".//item"))
    unit_type = unit.get("type", "unit")

    # Create chunks with split units
    current_chunk = None
    current_unit = None
    current_chunk_id = chunk_id_start
    current_unit_id = 0
    current_chunk_tokens = 0
    current_unit_tokens = 0

    for item in items:
        # Get item tokens
        item_tokens = int(item.get("tok", "0"))

        # Check if adding this item would exceed limits
        if current_unit is None:
            # Create new unit and chunk
            current_chunk = etree.Element("chunk")
            current_chunk.set("id", f"c{current_chunk_id}")
            current_chunk.set("tok", "____")

            current_unit = etree.Element("unit")
            current_unit.set("type", unit_type)
            current_unit.set("tok", "____")

            if unit.get("id"):
                current_unit.set("id", f"{unit.get('id')}-split-{current_unit_id}")

            current_chunk.append(current_unit)
            if unit_pos >= 0:
                parent.insert(
                    unit_pos + current_chunk_id - chunk_id_start, current_chunk
                )
            else:
                parent.append(current_chunk)

            current_chunk_tokens = 0
            current_unit_tokens = 0
            current_chunk_id += 1
            current_unit_id += 1

        # Remove item from original unit
        item_parent = item.getparent()
        if item_parent is not None:
            item_parent.remove(item)

        # Add item to current unit
        current_unit.append(item)
        current_unit_tokens += item_tokens
        current_chunk_tokens += item_tokens

        # Check if we need a new chunk/unit after adding this item
        if current_chunk_tokens >= max_chunk_size:
            # This chunk is full, create a new one
            current_chunk = None
            current_unit = None


def create_chunks_regex(doc_with_units: str, max_chunk_size: int) -> str:
    """
    Fallback method to create chunks using regex.

    Args:
        doc_with_units: XML document with items and units
        max_chunk_size: Maximum chunk size in tokens

    Returns:
        Document with chunk tags added
    """
    # Find all units with their token counts
    unit_matches = list(
        re.finditer(
            r'<unit type="([^"]*)" tok="(\d+)"[^>]*>(.*?)</unit>',
            doc_with_units,
            re.DOTALL,
        )
    )

    if not unit_matches:
        # No units found, wrap everything in one chunk
        result = (
            f'<doc>\n<chunk id="c0" tok="____">\n{doc_with_units}\n</chunk>\n</doc>'
        )
        return update_token_counts(result)

    # Extract document start/end
    doc_start_match = re.match(r"^(.*?)<unit", doc_with_units, re.DOTALL)
    doc_start = doc_start_match.group(1) if doc_start_match else "<doc>\n"

    doc_end_match = re.search(r"</unit>(.*?)$", doc_with_units, re.DOTALL)
    doc_end = doc_end_match.group(1) if doc_end_match else "\n</doc>"

    # Process units into chunks
    chunks = []
    current_chunk: list[str] = []
    current_chunk_id = 0
    current_chunk_tokens = 0

    for match in unit_matches:
        unit_type = match.group(1)
        unit_tokens = int(match.group(2))
        unit_content = match.group(3)

        if unit_tokens > max_chunk_size:
            # Handle oversized unit by splitting
            split_chunks = split_oversized_unit_regex(
                unit_type, unit_content, max_chunk_size, current_chunk_id
            )
            chunks.extend(split_chunks)
            current_chunk_id += len(split_chunks)
        elif current_chunk_tokens + unit_tokens > max_chunk_size:
            # Finalize current chunk and start a new one
            if current_chunk:
                chunk_xml = f'<chunk id="c{current_chunk_id}" tok="____">\n{"".join(current_chunk)}\n</chunk>'
                chunks.append(chunk_xml)
                current_chunk_id += 1
                current_chunk = []
                current_chunk_tokens = 0

            # Add unit to new chunk
            current_chunk.append(
                f'<unit type="{unit_type}" tok="{unit_tokens}">{unit_content}</unit>'
            )
            current_chunk_tokens += unit_tokens
        else:
            # Add to current chunk
            current_chunk.append(
                f'<unit type="{unit_type}" tok="{unit_tokens}">{unit_content}</unit>'
            )
            current_chunk_tokens += unit_tokens

    # Add remaining chunk
    if current_chunk:
        chunk_xml = f'<chunk id="c{current_chunk_id}" tok="____">\n{"".join(current_chunk)}\n</chunk>'
        chunks.append(chunk_xml)

    # Combine everything
    result = f"{doc_start}{''.join(chunks)}{doc_end}"

    # Update token counts
    return update_token_counts(result)


def split_oversized_unit_regex(
    unit_type: str, unit_content: str, max_chunk_size: int, chunk_id_start: int
) -> list[str]:
    """
    Split an oversized unit into multiple chunks.

    Args:
        unit_type: Type of the unit
        unit_content: Content of the unit
        max_chunk_size: Maximum chunk size in tokens
        chunk_id_start: Starting chunk ID

    Returns:
        List of chunk XML strings
    """
    # Find all items in the unit
    item_matches = list(
        re.finditer(
            r'<item tok="(\d+)" id="([^"]*)">(.*?)</item>', unit_content, re.DOTALL
        )
    )

    if not item_matches:
        # No items, just return the whole unit in one chunk
        return [
            f'<chunk id="c{chunk_id_start}" tok="____">\n<unit type="{unit_type}" tok="____">{unit_content}</unit>\n</chunk>'
        ]

    # Split into chunks
    chunks = []
    current_chunk_items: list[str] = []
    current_chunk_id = chunk_id_start
    current_unit_id = 0
    current_chunk_tokens = 0

    for match in item_matches:
        item_tokens = int(match.group(1))
        item_id = match.group(2)
        item_content = match.group(3)

        # Check if adding this item would exceed chunk size
        if current_chunk_tokens + item_tokens > max_chunk_size and current_chunk_items:
            # Finalize current chunk
            items_xml = "".join(current_chunk_items)
            chunk_xml = (
                f'<chunk id="c{current_chunk_id}" tok="____">\n'
                f'<unit type="{unit_type}" id="{item_id}-split-{current_unit_id}" tok="____">'
                f"{items_xml}</unit>\n</chunk>"
            )
            chunks.append(chunk_xml)

            # Reset for next chunk
            current_chunk_id += 1
            current_unit_id += 1
            current_chunk_items = []
            current_chunk_tokens = 0

        # Add item to current chunk
        current_chunk_items.append(
            f'<item xml:space="preserve" tok="{item_tokens}" id="{item_id}">{item_content}</item>'
        )
        current_chunk_tokens += item_tokens

    # Add remaining items
    if current_chunk_items:
        items_xml = "".join(current_chunk_items)
        chunk_xml = (
            f'<chunk id="c{current_chunk_id}" tok="____">\n'
            f'<unit type="{unit_type}" id="split-{current_unit_id}" tok="____">'
            f"{items_xml}</unit>\n</chunk>"
        )
        chunks.append(chunk_xml)

    return chunks


def clean_consecutive_newlines(content: str) -> str:
    """
    Clean up content by removing more than 2 consecutive newlines.

    Args:
        content: The XML document content

    Returns:
        The content with excessive newlines removed
    """
    # First, normalize all newlines to \n (convert Windows \r\n to \n)
    content = content.replace("\r\n", "\n")

    # Replace 3 or more consecutive newlines with just 2 newlines
    return re.sub(r"\n{3,}", "\n\n", content)


def pretty_print_xml(xml_str: str) -> str:
    """
    Return XML as is to preserve original whitespace and formatting.
    No pretty printing is performed to ensure exact whitespace preservation.
    """
    return xml_str


def process_document(
    input_file: str,
    output_file: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    model: str = DEFAULT_MODEL,
    temperature: float = DEFAULT_TEMPERATURE,
    encoding: str = "utf-8",
    verbose: bool = False,
    backup: bool = False,
) -> None:
    """
    Process a markdown document into a semantically chunked XML document.

    Args:
        input_file: Path to the input markdown file
        output_file: Path for the XML output
        chunk_size: Maximum chunk size in tokens
        model: LLM model identifier
        temperature: LLM temperature setting
        encoding: File encoding
        verbose: Enable detailed logging
        backup: Whether to create backup copies with timestamps
    """
    if verbose:
        logger.level("DEBUG")

    try:
        # Read input file
        logger.info(f"Reading input file: {input_file}")
        with open(input_file, encoding=encoding) as f:
            input_text = f.read()

        # Wrap in doc tag
        logger.debug("Wrapping document")

        # Itemize document
        logger.info("Itemizing document")
        items = itemize_document(input_text)
        item_elements = create_item_elements(items)
        itemized_doc = f"<doc>\n{''.join([item[0] for item in item_elements])}\n</doc>"

        # Update token counts for items
        logger.debug("Updating token counts")
        itemized_doc = update_token_counts(itemized_doc)

        # Create backup of initial itemized document if requested
        if backup:
            save_backup(output_file, itemized_doc.encode(encoding), "itemized")

        # Semantic analysis
        logger.info(f"Performing semantic analysis with model: {model}")
        try:
            semantic_boundaries = semantic_analysis(itemized_doc, model, temperature)
            logger.debug(f"Identified semantic boundaries: {semantic_boundaries}")

            # Create backup after semantic analysis if requested
            if backup:
                itemized_doc_with_boundaries = (
                    itemized_doc  # We don't modify the document yet
                )
                save_backup(
                    output_file,
                    itemized_doc_with_boundaries.encode(encoding),
                    "semantic_boundaries",
                )
        except Exception as e:
            logger.error(f"Semantic analysis failed: {e}")
            logger.warning("Using a single unit for the whole document")
            # Get first item ID
            first_item_id = item_elements[0][1] if item_elements else None
            semantic_boundaries = [(first_item_id, "unit")] if first_item_id else []

        # Add unit tags
        logger.info("Adding unit tags")
        doc_with_units = add_unit_tags(itemized_doc, semantic_boundaries)

        # Create backup after adding unit tags if requested
        if backup:
            save_backup(output_file, doc_with_units.encode(encoding), "units_added")

        # Create chunks
        logger.info(f"Creating chunks with max size: {chunk_size} tokens")
        chunked_doc = create_chunks(doc_with_units, chunk_size)

        # Create backup after creating chunks if requested
        if backup:
            save_backup(output_file, chunked_doc.encode(encoding), "chunks_created")

        # Format as nice XML
        logger.debug("Formatting XML")
        final_xml = pretty_print_xml(chunked_doc)

        # Clean up consecutive newlines
        logger.debug("Cleaning up consecutive newlines")
        final_xml = clean_consecutive_newlines(final_xml)

        # Write output file
        logger.info(f"Writing output to: {output_file}")
        with open(output_file, "wb") as f:
            f.write(final_xml.encode(encoding))

        # Create final backup if requested
        if backup:
            save_backup(output_file, final_xml.encode(encoding), "final")

        logger.success("Processing complete!")

    except Exception as e:
        logger.error(f"Error processing document: {e}")
        raise


def save_backup(output_file: str, content: bytes, stage: str) -> None:
    """
    Save a backup of the current document state with a timestamp.

    Args:
        output_file: Path to the output file
        content: Document content as bytes
        stage: Processing stage name
    """
    try:
        output_path = Path(output_file)
        backup_dir = output_path.parent / f"{output_path.stem}_backups"
        backup_dir.mkdir(exist_ok=True)

        # Generate timestamp
        timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

        # Create backup filename with stage and timestamp
        backup_filename = f"{output_path.stem}-{timestamp}-{stage}{output_path.suffix}"
        backup_path = backup_dir / backup_filename

        # Save backup
        with open(backup_path, "wb") as f:
            f.write(content)

        logger.debug(f"Created backup for {stage} stage at: {backup_path}")
    except Exception as e:
        logger.error(f"Error creating backup: {e}")


def main(
    input_file: str,
    output_file: str,
    chunk: int = DEFAULT_CHUNK_SIZE,
    model: str = DEFAULT_MODEL,
    temperature: float = DEFAULT_TEMPERATURE,
    encoding: str = "utf-8",
    verbose: bool = True,
    backup: bool = True,
) -> None:
    """
    Main entry point for malmo_chunker.

    Args:
        input_file: Path to the input markdown file
        output_file: Path for the XML output
        chunk: Maximum chunk size in tokens
        model: LLM model identifier
        temperature: LLM temperature setting
        encoding: File encoding
        verbose: Enable detailed logging
        backup: Whether to create backup copies with timestamps
    """
    if verbose:
        logger.level("DEBUG")
        _turn_on_debug()

    process_document(
        input_file, output_file, chunk, model, temperature, encoding, verbose, backup
    )


if __name__ == "__main__":
    fire.Fire(main)
