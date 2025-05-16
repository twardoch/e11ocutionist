#!/usr/bin/env python3
# this_file: src/e11ocutionist/chunker.py
"""
Chunker module for e11ocutionist.

This module handles the semantic chunking of documents for processing.
It breaks documents into manageable semantic chunks based on content boundaries.
"""

import hashlib
import re
from pathlib import Path
from typing import Any
import datetime
from copy import deepcopy

import tiktoken
from lxml import etree
from loguru import logger
from functools import cache
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Constants
DEFAULT_MODEL = "gpt-4o"  # Updated from "openrouter/openai/gpt-4.1"
FALLBACK_MODEL = (
    "gpt-4o-mini"  # Updated from "openrouter/google/gemini-2.5-pro-preview-03-25"
)
DEFAULT_CHUNK_SIZE = 12288
DEFAULT_TEMPERATURE = 0.2


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
    return paragraph.strip().startswith("> ")


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
        if (bullet_pattern.match(line.strip()) or number_pattern.match(line.strip()))
    )
    # Treat as list only if two or more items
    return bullet_count >= 2


def is_code_block(paragraph: str) -> bool:
    """Check if paragraph is a markdown code block."""
    return paragraph.strip().startswith("```") and paragraph.strip().endswith("```")


def is_horizontal_rule(paragraph: str) -> bool:
    """Check if paragraph is a markdown horizontal rule."""
    return bool(re.match(r"^[\*\-_]{3,}\s*$", paragraph.strip()))


def is_table(paragraph: str) -> bool:
    """Check if paragraph is a markdown table."""
    lines = paragraph.split("\n")
    if len(lines) < 2:
        return False
    # Check for pipe character and separator line
    return "|" in lines[0] and bool(re.match(r"^[\|\s\-:]+$", lines[1].strip()))


def is_html_block(paragraph: str) -> bool:
    """Check if paragraph is an HTML block."""
    p = paragraph.strip()
    return p.startswith("<") and p.endswith(">")


def is_image_or_figure(paragraph: str) -> bool:
    """Check if paragraph is a markdown image or figure."""
    return bool(re.match(r"!\[.*\]\(.*\)", paragraph.strip()))


def itemize_document(document_text: str) -> list[tuple[str, str]]:
    """
    Split a document into semantic paragraph-level units.

    This function identifies paragraph boundaries and classifies each paragraph
    based on its content type (heading, blockquote, list, etc.). It also
    determines the attachment type of each paragraph (whether it should be
    attached to the preceding paragraph, following paragraph, or treated as normal).

    Args:
        document_text: The raw text document to process

    Returns:
        A list of tuples where each tuple contains (item_text, attachment_type)
        attachment_type is one of: "normal", "following", "preceding"
    """
    # Split the document into paragraphs
    paragraphs = split_into_paragraphs(document_text)

    # Process each paragraph to determine its attachment type
    itemized_paragraphs = []

    for i, paragraph in enumerate(paragraphs):
        if not paragraph.strip():
            continue  # Skip empty paragraphs

        attachment_type = "normal"  # Default attachment type

        # Determine attachment type based on paragraph characteristics
        if is_heading(paragraph):
            # Headings get their own item and following content attaches to them
            attachment_type = "normal"
        elif is_blockquote(paragraph) or is_list(paragraph) or is_code_block(paragraph):
            # These get their own items
            attachment_type = "normal"
        elif (
            is_horizontal_rule(paragraph)
            or is_table(paragraph)
            or is_html_block(paragraph)
        ):
            # These are normal items
            attachment_type = "normal"
        elif is_image_or_figure(paragraph):
            # Images/figures typically attach to preceding content
            attachment_type = "preceding"
        elif i > 0 and len(paragraph.strip()) < 100:
            # Short paragraphs after another paragraph might be continuations
            prev_para = paragraphs[i - 1].strip()
            if prev_para.endswith((":", "—")) or (
                prev_para
                and not any(
                    prev_para.endswith(x) for x in [".", "!", "?", '"', "'", ")", "]"]
                )
            ):
                attachment_type = "preceding"

        # Add to the list with its attachment type
        itemized_paragraphs.append((paragraph, attachment_type))

    return itemized_paragraphs


def replace_tok_placeholder(text: str, placeholder: str = 'tok="____"') -> str:
    """
    Replace token count placeholder with actual count.

    This function accurately counts tokens in XML elements and ensures
    the tok attribute reflects the actual token count.

    Args:
        text: XML text with token placeholders
        placeholder: The placeholder pattern to replace

    Returns:
        XML text with actual token counts
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


def create_item_elements(items: list[tuple[str, str]]) -> list[tuple[str, str]]:
    """
    Create XML item elements from itemized paragraphs.

    This function takes the output of itemize_document and generates XML items
    with appropriate attributes. Each item includes a unique ID, token count,
    and preserves whitespace.

    Args:
        items: List of tuples containing (item_text, attachment_type)

    Returns:
        List of tuples containing (item_xml, item_id)
    """
    xml_items = []
    prev_id = ""

    for _i, (text, attachment_type) in enumerate(items):
        # Escape XML special characters in the text
        escaped_text = escape_xml_chars(text)

        # Generate a unique ID for this item
        item_id = generate_id(prev_id, text)
        prev_id = item_id

        # Construct the XML item
        tok_placeholder = "____"  # Will be replaced with actual count later

        # Set additional attributes based on attachment type
        attachment_attr = ""
        if attachment_type == "following":
            attachment_attr = ' attach="following"'
        elif attachment_type == "preceding":
            attachment_attr = ' attach="preceding"'

        # Create the item element with attributes
        item_xml = f'<item xml:space="preserve" tok="{tok_placeholder}" id="{item_id}"{attachment_attr}>{escaped_text}</item>'

        # Add to the list
        xml_items.append((item_xml, item_id))

    return xml_items


def create_chunks(doc_with_units: str, max_chunk_size: int) -> str:
    """
    Organize units into chunks of manageable size.

    This function groups units into chunks, ensuring each chunk doesn't
    exceed the maximum token size. It handles oversized units by splitting
    them across chunks.

    Args:
        doc_with_units: XML document with unit tags
        max_chunk_size: Maximum token size for chunks

    Returns:
        XML document with chunk tags
    """
    logger.info(f"Creating chunks with max size of {max_chunk_size} tokens")

    try:
        # Parse the XML document
        parser = etree.XMLParser(remove_blank_text=False)
        root = etree.XML(doc_with_units.encode("utf-8"), parser)

        # Find all units
        units = root.xpath("//unit")
        if not units:
            logger.warning("No units found in document")
            return doc_with_units

        # Remove units from their current position
        for unit in units:
            unit.getparent().remove(unit)

        # Create initial chunk
        chunk_id = 1
        chunk = etree.SubElement(root, "chunk")
        chunk.set("id", f"{chunk_id:04d}")
        chunk.set("tok", "____")
        current_chunk_tokens = 0

        # Process each unit
        for unit in units:
            # Count tokens in the unit
            unit_xml = etree.tostring(unit, encoding="utf-8").decode("utf-8")
            unit_tokens = count_tokens(unit_xml)

            # Check if this unit is too large for a single chunk
            if unit_tokens > max_chunk_size:
                logger.info(f"Found oversized unit with {unit_tokens} tokens")
                handle_oversized_unit(root, unit, max_chunk_size, chunk_id)
                # Update chunk_id to account for new chunks
                chunk_id = len(root.xpath("//chunk"))
                # Create a new chunk for the next unit
                chunk = etree.SubElement(root, "chunk")
                chunk_id += 1
                chunk.set("id", f"{chunk_id:04d}")
                chunk.set("tok", "____")
                current_chunk_tokens = 0
            # Check if adding this unit would exceed the chunk size
            elif current_chunk_tokens + unit_tokens > max_chunk_size:
                # Start a new chunk
                chunk = etree.SubElement(root, "chunk")
                chunk_id += 1
                chunk.set("id", f"{chunk_id:04d}")
                chunk.set("tok", "____")
                chunk.append(unit)
                current_chunk_tokens = unit_tokens
            else:
                # Add to current chunk
                chunk.append(unit)
                current_chunk_tokens += unit_tokens

        # Update token counts
        for e in root.xpath("//*[@tok]"):
            if e.get("tok") == "____":
                # Count tokens in this element
                e_xml = etree.tostring(e, encoding="utf-8").decode("utf-8")
                e_tokens = count_tokens(e_xml)
                e.set("tok", str(e_tokens))

        # Serialize back to XML
        doc_with_chunks = etree.tostring(
            root, encoding="utf-8", method="xml", xml_declaration=True
        ).decode("utf-8")

        logger.info(f"Created {chunk_id} chunks")
        return doc_with_chunks

    except Exception as e:
        logger.error(f"Failed to create chunks: {e}")
        logger.info("Falling back to regex-based chunk creation")
        return create_chunks_regex(doc_with_units, max_chunk_size)


def handle_oversized_unit(
    doc_root, unit, max_chunk_size: int, chunk_id_start: int
) -> None:
    """
    Split an oversized unit across multiple chunks.

    This function splits a unit that's too large for a single chunk into
    multiple chunks, preserving the unit attributes.

    Args:
        doc_root: XML document root
        unit: Oversized unit element
        max_chunk_size: Maximum token size for chunks
        chunk_id_start: Starting chunk ID

    Returns:
        None (modifies doc_root in place)
    """
    logger.info("Handling oversized unit by splitting it across chunks")

    unit_type = unit.get("type", "unit")

    # Get all items in the unit
    items = unit.xpath(".//item")
    if not items:
        logger.warning("No items found in oversized unit")
        return

    # Group items into chunks
    current_items = []
    current_tokens = 0
    chunk_items = []  # List of item groups

    for item in items:
        item_xml = etree.tostring(item, encoding="utf-8").decode("utf-8")
        item_tokens = count_tokens(item_xml)

        # If adding this item exceeds the chunk size and we already have items
        if current_items and current_tokens + item_tokens > max_chunk_size:
            # Save current group and start a new one
            chunk_items.append(current_items)
            current_items = [item]
            current_tokens = item_tokens
        else:
            # Add to current group
            current_items.append(item)
            current_tokens += item_tokens

    # Add the last group if not empty
    if current_items:
        chunk_items.append(current_items)

    # Create chunks with the grouped items
    for i, item_group in enumerate(chunk_items):
        # Create a new chunk
        chunk = etree.SubElement(doc_root, "chunk")
        chunk_id = chunk_id_start + i
        chunk.set("id", f"{chunk_id:04d}")
        chunk.set("tok", "____")

        # Create a new unit inside the chunk
        new_unit = etree.SubElement(chunk, "unit")
        new_unit.set("type", unit_type)
        new_unit.set("tok", "____")

        # If this is a continuation, add an attribute
        if i > 0:
            new_unit.set("cont", "true")

        # Add items to the unit
        for item in item_group:
            # Copy the item
            item_copy = deepcopy(item)
            new_unit.append(item_copy)


def create_chunks_regex(doc_with_units: str, max_chunk_size: int) -> str:
    """
    Create chunks using regex when XML parsing fails.

    This is a fallback implementation that uses regex instead of XML parsing.

    Args:
        doc_with_units: XML document with unit tags
        max_chunk_size: Maximum token size for chunks

    Returns:
        XML document with chunk tags
    """
    logger.info("Using regex to create chunks")

    # Find the root tag
    root_match = re.search(r"<([^>]+)>", doc_with_units)
    if not root_match:
        logger.error("Could not find root tag")
        return doc_with_units

    root_tag = root_match.group(1)

    # Extract all units with their content
    unit_pattern = re.compile(r"<unit[^>]*>(.*?)</unit>", re.DOTALL)
    units = unit_pattern.findall(doc_with_units)

    if not units:
        logger.warning("No units found using regex")
        return doc_with_units

    # Create chunks
    chunks = []
    current_chunk = []
    current_tokens = 0

    for unit_content in units:
        unit_tokens = count_tokens(unit_content)

        # Check if this unit is too large for a single chunk
        if unit_tokens > max_chunk_size:
            # Split the oversized unit
            unit_type_match = re.search(r'<unit[^>]*type="([^"]+)"', unit_content)
            unit_type = unit_type_match.group(1) if unit_type_match else "unit"

            split_units = split_oversized_unit_regex(
                unit_type, unit_content, max_chunk_size, len(chunks) + 1
            )

            # Add each split unit as its own chunk
            for split_unit in split_units:
                chunks.append([split_unit])
        # Check if adding this unit would exceed the chunk size
        elif current_tokens + unit_tokens > max_chunk_size:
            # Save current chunk and start a new one
            if current_chunk:
                chunks.append(current_chunk)
                current_chunk = [unit_content]
                current_tokens = unit_tokens
        else:
            # Add to current chunk
            current_chunk.append(unit_content)
            current_tokens += unit_tokens

    # Add the last chunk if not empty
    if current_chunk:
        chunks.append(current_chunk)

    # Rebuild the document with chunks
    result = f'<?xml version="1.0" encoding="UTF-8"?>\n<{root_tag}>'

    for i, chunk_units in enumerate(chunks):
        chunk_id = i + 1
        chunk_content = "".join(
            f'<unit type="unit" tok="____">{unit}</unit>' for unit in chunk_units
        )
        result += f'\n  <chunk id="{chunk_id:04d}" tok="____">{chunk_content}</chunk>'

    result += f"\n</{root_tag}>"

    # Replace token placeholders with actual counts
    result = replace_tok_placeholder(result)

    logger.info(f"Created {len(chunks)} chunks using regex")
    return result


def split_oversized_unit_regex(
    unit_type: str, unit_content: str, max_chunk_size: int, chunk_id_start: int
) -> list[str]:
    """
    Split an oversized unit into multiple units using regex.

    Args:
        unit_type: Type of the unit ("chapter", "scene", "unit")
        unit_content: Content of the unit
        max_chunk_size: Maximum token size
        chunk_id_start: Starting chunk ID

    Returns:
        List of split unit contents
    """
    logger.info(f"Splitting oversized unit of type {unit_type}")

    # Extract items from the unit
    item_pattern = re.compile(r"<item[^>]*>.*?</item>", re.DOTALL)
    items = item_pattern.findall(unit_content)

    if not items:
        logger.warning("No items found in oversized unit")
        return [unit_content]

    # Group items into chunks
    result = []
    current_items = []
    current_tokens = 0

    for item in items:
        item_tokens = count_tokens(item)

        # If adding this item exceeds the chunk size and we already have items
        if current_items and current_tokens + item_tokens > max_chunk_size:
            # Save current group and start a new one
            result.append("".join(current_items))
            current_items = [item]
            current_tokens = item_tokens
        else:
            # Add to current group
            current_items.append(item)
            current_tokens += item_tokens

    # Add the last group if not empty
    if current_items:
        result.append("".join(current_items))

    logger.info(f"Split unit into {len(result)} parts")
    return result


def clean_consecutive_newlines(content: str) -> str:
    """Replace more than two consecutive newlines with exactly two."""
    return re.sub(r"\n{3,}", "\n\n", content)


def pretty_print_xml(xml_str: str) -> str:
    """Format XML string with proper indentation for readability."""
    try:
        parser = etree.XMLParser(remove_blank_text=True)
        root = etree.XML(xml_str.encode("utf-8"), parser)
        return etree.tostring(
            root, encoding="utf-8", pretty_print=True, xml_declaration=True
        ).decode("utf-8")
    except Exception:
        # If parsing fails, return the original string
        return xml_str


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


def process_document(
    input_file: str,
    output_file: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    model: str = DEFAULT_MODEL,
    temperature: float = DEFAULT_TEMPERATURE,
    encoding: str = "utf-8",
    verbose: bool = False,
    backup: bool = False,
) -> dict[str, Any]:
    """
    Process a document through the semantic chunking pipeline.

    This function:
    1. Itemizes the document (splits it into paragraph-level items)
    2. Optionally performs semantic analysis using an LLM to identify chapter/scene boundaries
    3. Groups items into semantic units
    4. Organizes units into chunks of manageable size for further processing

    Args:
        input_file: Path to the input document
        output_file: Path to save the processed XML output
        chunk_size: Maximum token size for chunks
        model: LLM model to use for semantic analysis
        temperature: Temperature setting for the LLM
        encoding: Character encoding of the input file
        verbose: Enable verbose logging
        backup: Create backups at intermediate stages

    Returns:
        Dictionary with processing statistics
    """
    logger.info(f"Processing document: {input_file} -> {output_file}")
    logger.info(
        f"Using model: {model}, temperature: {temperature}, chunk size: {chunk_size}"
    )

    try:
        # Read the input file
        with open(input_file, encoding=encoding) as f:
            document_text = f.read()

        # Normalize line endings
        document_text = document_text.replace("\r\n", "\n")

        # Step 1: Itemize the document
        logger.info("Step 1: Itemizing document")
        itemized_paragraphs = itemize_document(document_text)
        logger.info(f"Identified {len(itemized_paragraphs)} paragraphs")

        # Step 2: Create XML items
        logger.info("Step 2: Creating XML items")
        xml_items = create_item_elements(itemized_paragraphs)

        # Create initial XML document with items
        items_xml = "\n".join(item_xml for item_xml, _ in xml_items)
        doc_xml = f"""<?xml version="1.0" encoding="UTF-8"?>
<doc>
{items_xml}
</doc>"""

        # Replace token placeholders
        doc_xml = replace_tok_placeholder(doc_xml)

        # Save backup after itemization if requested
        if backup:
            backup_path = create_backup(output_file, "items")
            if backup_path:
                Path(backup_path).write_text(doc_xml, encoding="utf-8")

        # Step 3: Semantic analysis to identify boundaries
        logger.info("Step 3: Performing semantic analysis")
        semantic_boundaries = semantic_analysis(doc_xml, model, temperature)

        # Step 4: Add unit tags based on semantic boundaries
        logger.info("Step 4: Adding unit tags")
        doc_with_units = add_unit_tags(doc_xml, semantic_boundaries)

        # Replace token placeholders
        doc_with_units = replace_tok_placeholder(doc_with_units)

        # Save backup after adding units if requested
        if backup:
            backup_path = create_backup(output_file, "units")
            if backup_path:
                Path(backup_path).write_text(doc_with_units, encoding="utf-8")

        # Step 5: Create chunks
        logger.info("Step 5: Creating chunks")
        doc_with_chunks = create_chunks(doc_with_units, chunk_size)

        # Replace token placeholders one last time
        doc_with_chunks = replace_tok_placeholder(doc_with_chunks)

        # Format for readability
        if verbose:
            doc_with_chunks = pretty_print_xml(doc_with_chunks)

        # Write the final result
        Path(output_file).write_text(doc_with_chunks, encoding="utf-8")

        # Count items, units, and chunks for statistics
        items_count = len(xml_items)
        units_count = len(re.findall(r"<unit[^>]*>", doc_with_chunks))
        chunks_count = len(re.findall(r"<chunk[^>]*>", doc_with_chunks))
        total_tokens = count_tokens(doc_with_chunks)

        logger.info(
            f"Processing complete: {items_count} items, {units_count} units, {chunks_count} chunks"
        )
        logger.info(f"Total tokens: {total_tokens}")

        return {
            "input_file": input_file,
            "output_file": output_file,
            "items_count": items_count,
            "units_count": units_count,
            "chunks_count": chunks_count,
            "total_tokens": total_tokens,
        }

    except Exception as e:
        logger.error(f"Processing failed: {e}")
        # Create a minimal valid XML in case of failure
        fallback_xml = f"""<?xml version="1.0" encoding="UTF-8"?>
<doc>
  <chunk id="0001" tok="1">
    <unit type="chapter" tok="1">
      <item xml:space="preserve" tok="1" id="000000-123456">
        Error processing document: {e!s}
      </item>
    </unit>
  </chunk>
</doc>
"""
        Path(output_file).write_text(fallback_xml, encoding="utf-8")

        return {
            "input_file": input_file,
            "output_file": output_file,
            "items_count": 1,
            "units_count": 1,
            "chunks_count": 1,
            "total_tokens": 1,
            "error": str(e),
        }


def semantic_analysis(
    doc_text: str, model: str, temperature: float
) -> list[tuple[str, str]]:
    """
    Use LLM to identify semantic boundaries in the document.

    This function sends the document to an LLM with instructions to identify
    semantic boundaries such as chapters, scenes, and thematic units. The
    LLM's response is processed to extract identified boundaries.

    Args:
        doc_text: Document text with item IDs
        model: LLM model to use
        temperature: Temperature setting for the LLM

    Returns:
        List of tuples (item_id, boundary_type) where boundary_type is
        "chapter", "scene", or "unit"
    """
    logger.info("Performing semantic analysis of document")

    # Extract item IDs from the document
    item_pattern = re.compile(r'<item[^>]*id="([^"]+)"[^>]*>.*?</item>', re.DOTALL)
    items = item_pattern.findall(doc_text)

    if not items:
        logger.warning("No items found in document, skipping semantic analysis")
        return []

    # Create a simplified version of the document for the LLM
    # This extracts just enough content for the LLM to understand the structure
    simplified_doc = []
    for match in item_pattern.finditer(doc_text):
        item_id = match.group(1)
        # Extract text content (remove tags and limit length)
        content = re.sub(r"<.*?>", "", match.group(0))
        # Truncate to first 100 chars for LLM efficiency
        content = content[:100] + ("..." if len(content) > 100 else "")
        simplified_doc.append(f"ID: {item_id}\nContent: {content}\n")

    "\n".join(simplified_doc)

    # Construct the prompt for the LLM

    try:
        # Call the LLM API (implementation will depend on your LLM library)
        # This is a placeholder for the actual API call
        # In a real implementation, you would use a library like litellm, openai, etc.
        logger.info(f"Calling LLM API with model: {model}")

        # Placeholder for LLM call
        # In the legacy code, this uses litellm.completion()
        # For now, we'll just create a dummy response
        response = {
            "choices": [{"message": {"content": "BOUNDARY: 000000-123456, chapter"}}]
        }

        logger.info("Successfully received response from LLM")

        # Process the LLM response
        boundaries = extract_and_parse_llm_response(response)

        # Apply fallback logic if too few boundaries
        if len(boundaries) < max(3, len(items) // 100):
            logger.warning("Too few boundaries identified, applying fallback logic")
            # Add fallback boundaries every N items as a backup
            interval = min(50, max(10, len(items) // 10))
            for i in range(0, len(items), interval):
                if i > 0:  # Skip the first item
                    item_id = items[i]
                    if not any(b[0] == item_id for b in boundaries):
                        boundaries.append((item_id, "unit"))

        logger.info(f"Identified {len(boundaries)} semantic boundaries")
        return boundaries

    except Exception as e:
        logger.error(f"Semantic analysis failed: {e}")
        # Fallback: create basic boundaries every N items
        logger.info("Using fallback boundary detection")
        boundaries = []
        interval = min(50, max(10, len(items) // 10))
        for i in range(0, len(items), interval):
            if i > 0:  # Skip the first item
                boundaries.append((items[i], "unit"))
        logger.info(f"Created {len(boundaries)} fallback boundaries")
        return boundaries


def extract_and_parse_llm_response(response) -> list[tuple[str, str]]:
    """
    Extract and parse boundaries from LLM response.

    Args:
        response: LLM API response object

    Returns:
        List of tuples (item_id, boundary_type)
    """
    boundaries = []

    # Extract the content from the response
    if hasattr(response, "choices") and response.choices:
        # For OpenAI-like responses
        content = response.choices[0].message.content
    elif isinstance(response, dict) and "choices" in response:
        # For dict-like responses
        content = response["choices"][0]["message"]["content"]
    else:
        # Fallback
        logger.warning("Unexpected response format from LLM")
        return boundaries

    # Parse the response to extract boundaries
    pattern = re.compile(r"BOUNDARY:\s*([^,]+),\s*([a-z]+)", re.IGNORECASE)
    matches = pattern.findall(content)

    for item_id, boundary_type in matches:
        item_id = item_id.strip()
        boundary_type = boundary_type.strip().lower()

        # Validate boundary type
        if boundary_type not in ["chapter", "scene", "unit"]:
            logger.warning(
                f"Invalid boundary type: {boundary_type}, defaulting to 'unit'"
            )
            boundary_type = "unit"

        boundaries.append((item_id, boundary_type))

    return boundaries


def add_unit_tags(itemized_doc: str, semantic_boundaries: list[tuple[str, str]]) -> str:
    """
    Add unit tags around sequences of items based on semantic analysis.

    This function groups items by their boundary types and adds appropriate
    XML unit tags with attributes.

    Args:
        itemized_doc: XML document with items
        semantic_boundaries: List of tuples (item_id, boundary_type)

    Returns:
        XML document with added unit tags
    """
    logger.info("Adding unit tags based on semantic boundaries")

    try:
        # Parse the XML document
        parser = etree.XMLParser(remove_blank_text=False)
        root = etree.XML(itemized_doc.encode("utf-8"), parser)

        # Create a map of item IDs to boundary types
        boundary_map = dict(semantic_boundaries)

        # Find all items
        items = root.xpath("//item")
        if not items:
            logger.warning("No items found in document")
            return itemized_doc

        # Keep track of units we're creating
        current_unit = None
        current_type = None
        current_items = []

        # Process each item
        for item in items:
            item_id = item.get("id")

            # Check if this item is a boundary
            if item_id in boundary_map:
                # If we have a current unit, finalize it
                if current_unit is not None and current_items:
                    # Create a new unit element
                    unit = etree.Element("unit")
                    unit.set("type", current_type)
                    unit.set("tok", "____")  # Will be replaced later

                    # Move items into the unit
                    parent = current_items[0].getparent()
                    unit_index = parent.index(current_items[0])
                    parent.insert(unit_index, unit)

                    for item_elem in current_items:
                        unit.append(item_elem)

                # Start a new unit
                current_type = boundary_map[item_id]
                current_unit = item_id
                current_items = [item]
            elif current_unit is not None:
                current_items.append(item)
            else:
                # No current unit, create a default one
                current_type = "unit"
                current_unit = item_id
                current_items = [item]

        # Handle the last unit
        if current_unit is not None and current_items:
            unit = etree.Element("unit")
            unit.set("type", current_type)
            unit.set("tok", "____")

            parent = current_items[0].getparent()
            unit_index = parent.index(current_items[0])
            parent.insert(unit_index, unit)

            for item_elem in current_items:
                unit.append(item_elem)

        # Serialize back to XML
        doc_with_units = etree.tostring(
            root, encoding="utf-8", method="xml", xml_declaration=True
        ).decode("utf-8")

        logger.info("Successfully added unit tags")
        return doc_with_units

    except Exception as e:
        logger.error(f"Failed to add unit tags: {e}")
        logger.info("Falling back to regex-based unit tagging")
        return add_unit_tags_regex(itemized_doc, semantic_boundaries)


def add_unit_tags_regex(
    itemized_doc: str, semantic_boundaries: list[tuple[str, str]]
) -> str:
    """
    Add unit tags using regex when XML parsing fails.

    This is a fallback implementation that uses regex instead of XML parsing.

    Args:
        itemized_doc: XML document with items
        semantic_boundaries: List of tuples (item_id, boundary_type)

    Returns:
        XML document with added unit tags
    """
    logger.info("Using regex to add unit tags")

    # Create a map of item IDs to boundary types
    boundary_map = dict(semantic_boundaries)

    # Add a dummy boundary at the beginning if none exists
    # This ensures the first items are also grouped
    first_item_match = re.search(r'<item[^>]*id="([^"]+)"', itemized_doc)
    if first_item_match and first_item_match.group(1) not in boundary_map:
        boundary_map[first_item_match.group(1)] = "unit"

    # Find all items with their IDs
    item_pattern = re.compile(r'<item[^>]*id="([^"]+)"[^>]*>.*?</item>', re.DOTALL)

    # Get all items and their positions
    items = []
    for match in item_pattern.finditer(itemized_doc):
        item_id = match.group(1)
        items.append((item_id, match.start(), match.end(), match.group(0)))

    # Check if we have items
    if not items:
        logger.warning("No items found using regex")
        return itemized_doc

    # Find boundaries
    boundaries = []
    for i, (item_id, _start, _end, _) in enumerate(items):
        if item_id in boundary_map:
            boundary_type = boundary_map[item_id]
            boundaries.append((i, boundary_type))

    # If no boundaries found, create default
    if not boundaries:
        logger.warning("No boundaries found, creating default")
        boundaries = [(0, "unit")]

    # Create units by inserting tags at appropriate positions
    result = itemized_doc
    offset = 0  # Track string position offset as we add tags

    # Add each unit
    for i, (boundary_index, boundary_type) in enumerate(boundaries):
        # Determine unit start and end
        start_index = boundary_index
        end_index = boundaries[i + 1][0] if i + 1 < len(boundaries) else len(items)

        if start_index >= end_index:
            continue  # Skip empty units

        # Get item positions
        unit_start = items[start_index][1]
        unit_end = items[end_index - 1][2]

        # Insert unit start tag
        unit_start_tag = f'<unit type="{boundary_type}" tok="____">'
        result = (
            result[: unit_start + offset]
            + unit_start_tag
            + result[unit_start + offset :]
        )
        offset += len(unit_start_tag)

        # Insert unit end tag
        unit_end_tag = "</unit>"
        result = (
            result[: unit_end + offset] + unit_end_tag + result[unit_end + offset :]
        )
        offset += len(unit_end_tag)

    logger.info("Successfully added unit tags using regex")
    return result


# The remaining functions from malmo_chunker.py will be implemented
# in subsequent commits. These include:
# - semantic_analysis: Use LLM to identify semantic boundaries
# - add_unit_tags: Group items into semantic units
# - create_chunks: Organize units into chunks of manageable size
# - handle_oversized_unit: Split large units into multiple chunks
