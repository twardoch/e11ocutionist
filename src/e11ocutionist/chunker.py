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
DEFAULT_MODEL = "gpt-4o"
FALLBACK_MODEL = "gpt-4o-mini"
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
    text = text.replace("&", "&amp;")
    text = text.replace("<", "&lt;")
    text = text.replace(">", "&gt;")
    text = text.replace('"', "&quot;")
    text = text.replace("'", "&apos;")
    return text


def generate_hash(text: str) -> str:
    """Generate a 6-character base36 hash from text."""
    sha1 = hashlib.sha1(text.encode("utf-8")).hexdigest()
    base36 = int(sha1, 16)
    chars = "0123456789abcdefghijklmnopqrstuvwxyz"
    result = ""
    while base36 > 0:
        base36, remainder = divmod(base36, 36)
        result = chars[remainder] + result
    result = result.zfill(6)
    return result[:6]


def generate_id(prev_id: str, content: str) -> str:
    """Generate a unique ID for an item based on previous ID and content."""
    if not prev_id:
        return f"000000-{generate_hash(content)}"
    else:
        prev_suffix = prev_id.split("-")[-1]
        return f"{prev_suffix}-{generate_hash(content)}"


def split_into_paragraphs(text: str) -> list[str]:
    text = text.replace("\r\n", "\n")
    parts = re.split(r"(\n\s*\n)", text)
    return ["".join(parts[i : i + 2]) for i in range(0, len(parts), 2)]


def is_heading(paragraph: str) -> bool:
    if bool(re.match(r"^#{1,6}\s+", paragraph)):
        return True
    if bool(re.search(r'(?i)^[„"]?(?:rozdział|chapter)\s+\w+', paragraph.strip())):
        return True
    return False


def is_blockquote(paragraph: str) -> bool:
    return paragraph.strip().startswith("> ")


def is_list(paragraph: str) -> bool:
    text = paragraph.rstrip("\n")
    lines = text.split("\n")
    bullet_pattern = re.compile(r"^[\*\-\+]\s+")
    number_pattern = re.compile(r"^\d+\.\s+")
    bullet_count = sum(
        1
        for line in lines
        if (bullet_pattern.match(line.strip()) or number_pattern.match(line.strip()))
    )
    return bullet_count >= 2


def is_code_block(paragraph: str) -> bool:
    return paragraph.strip().startswith("```") and paragraph.strip().endswith("```")


def is_horizontal_rule(paragraph: str) -> bool:
    return bool(re.match(r"^[\*\-_]{3,}\s*$", paragraph.strip()))


def is_table(paragraph: str) -> bool:
    lines = paragraph.split("\n")
    if len(lines) < 2:
        return False
    return "|" in lines[0] and bool(re.match(r"^[\|\s\-:]+$", lines[1].strip()))


def is_html_block(paragraph: str) -> bool:
    p = paragraph.strip()
    return p.startswith("<") and p.endswith(">")


def is_image_or_figure(paragraph: str) -> bool:
    return bool(re.match(r"!\[.*\]\(.*\)", paragraph.strip()))


def itemize_document(document_text: str) -> list[tuple[str, str]]:
    paragraphs = split_into_paragraphs(document_text)
    itemized_paragraphs = []
    for i, paragraph in enumerate(paragraphs):
        if not paragraph.strip():
            continue
        attachment_type = "normal"
        if is_heading(paragraph):
            attachment_type = "normal"
        elif is_blockquote(paragraph) or is_list(paragraph) or is_code_block(paragraph):
            attachment_type = "normal"
        elif (
            is_horizontal_rule(paragraph)
            or is_table(paragraph)
            or is_html_block(paragraph)
        ):
            attachment_type = "normal"
        elif is_image_or_figure(paragraph):
            attachment_type = "preceding"
        elif i > 0 and len(paragraph.strip()) < 100:
            prev_para = paragraphs[i - 1].strip()
            if prev_para.endswith((":", "—")) or (
                prev_para
                and not any(
                    prev_para.endswith(x) for x in [".", "!", "?", '"', "'", ")", "]"]
                )
            ):
                attachment_type = "preceding"
        itemized_paragraphs.append((paragraph, attachment_type))
    return itemized_paragraphs


def replace_tok_placeholder(text: str, placeholder: str = 'tok="____"') -> str:
    if placeholder not in text:
        return text
    temp_placeholder = "-987"
    text_with_temp = text.replace(placeholder, f'tok="{temp_placeholder}"')
    token_count = count_tokens(text_with_temp.replace(temp_placeholder, "0"))
    return text_with_temp.replace(temp_placeholder, str(token_count))

def update_tok_attributes_on_tree(root: etree._Element):
    """Recursively update tok attributes on the tree for placeholder or missing tok."""
    for element in root.xpath("//*[@tok='____' or not(@tok)]"):
        try:
            # Attempt to serialize only the element's text content and immediate children's text
            # This is a heuristic to get a 'content' token count rather than full XML structure.
            content_parts = []
            if element.text:
                content_parts.append(element.text)
            for child in element:
                if child.tail: # Text between child and next sibling
                    content_parts.append(child.tail)
            # For a more accurate 'content' token count, one might need to iterate deeper or use specific logic.
            # The current approach approximates by serializing the element if direct text is too simple.
            if not content_parts and len(element) > 0: # Element with children but no direct text
                 element_xml_for_tokens = etree.tostring(element, encoding="utf-8", method="xml").decode("utf-8")
            else:
                 element_xml_for_tokens = " ".join(content_parts).strip() if content_parts else ""

            token_count = count_tokens(element_xml_for_tokens)
            element.set("tok", str(token_count))
        except Exception as e:
            logger.warning(f"Could not update token for element {element.tag}: {e}")
            element.set("tok", "0") # Default to 0 on error

def create_item_elements(items: list[tuple[str, str]]) -> list[tuple[str, str]]:
    xml_items = []
    prev_id = ""
    for _i, (text, attachment_type) in enumerate(items):
        escaped_text = escape_xml_chars(text)
        item_id = generate_id(prev_id, text)
        prev_id = item_id
        tok_placeholder = "____"
        attachment_attr = ""
        if attachment_type == "following":
            attachment_attr = ' attach="following"'
        elif attachment_type == "preceding":
            attachment_attr = ' attach="preceding"'
        item_xml = f'<item xml:space="preserve" tok="{tok_placeholder}" id="{item_id}"{attachment_attr}>{escaped_text}</item>'
        xml_items.append((item_xml, item_id))
    return xml_items

def add_unit_tags_on_tree(root_element: etree._Element, semantic_boundaries: list[tuple[str, str]]) -> etree._Element:
    logger.info("Adding unit tags based on semantic boundaries (tree-based)")
    boundary_map = dict(semantic_boundaries)
    items = root_element.xpath("//item")

    if not items:
        logger.warning("No items found in document for unit tagging.")
        return root_element

    # Ensure first item starts a unit if no boundary is defined for it
    if items[0].get("id") not in boundary_map:
        boundary_map[items[0].get("id")] = "unit"

    new_children_for_root = []
    current_unit_items = []
    current_unit_type = "unit" # Default

    for item in items:
        item_id = item.get("id")
        if item_id in boundary_map: # This item starts a new unit
            if current_unit_items: # Finalize previous unit
                unit_element = etree.Element("unit")
                unit_element.set("type", current_unit_type)
                unit_element.set("tok", "____")
                for prev_item in current_unit_items:
                    unit_element.append(deepcopy(prev_item)) # Append copies
                new_children_for_root.append(unit_element)
            current_unit_items = [item]
            current_unit_type = boundary_map[item_id]
        else:
            current_unit_items.append(item)

    if current_unit_items: # Finalize the last unit
        unit_element = etree.Element("unit")
        unit_element.set("type", current_unit_type)
        unit_element.set("tok", "____")
        for prev_item in current_unit_items:
            unit_element.append(deepcopy(prev_item))
        new_children_for_root.append(unit_element)

    # Replace root's children with the new units
    for child in list(root_element): # Clear existing children (items)
        root_element.remove(child)
    for unit_child in new_children_for_root: # Add new units
        root_element.append(unit_child)

    logger.info(f"Created {len(new_children_for_root)} unit(s).")
    return root_element

def create_chunks_on_tree(root_doc_element: etree._Element, max_chunk_size: int) -> etree._Element:
    logger.info(f"Creating chunks on tree with max size of {max_chunk_size} tokens")

    units = list(root_doc_element.xpath("./unit"))

    if not units:
        logger.warning("No direct <unit> children found in <doc> to create chunks.")
        if list(root_doc_element) and not root_doc_element.xpath("./chunk"): # If <doc> has items but no chunks
            logger.info("Wrapping existing content of <doc> in a default chunk.")
            default_chunk = etree.Element("chunk")
            default_chunk.set("id", "0001")
            default_chunk.set("tok", "____")
            for child in list(root_doc_element):
                root_doc_element.remove(child)
                default_chunk.append(child)
            root_doc_element.append(default_chunk)
        return root_doc_element

    for unit_node in units: # Detach units from <doc>
        unit_node.getparent().remove(unit_node)

    for old_chunk in root_doc_element.xpath("./chunk"): # Clear any old chunks
        root_doc_element.remove(old_chunk)

    chunk_id_counter = 1
    current_chunk_node: etree._Element | None = None
    current_chunk_tokens = 0

    for unit_node in units:
        unit_xml_for_tokens = etree.tostring(unit_node, encoding="utf-8").decode("utf-8")
        unit_tokens = count_tokens(unit_xml_for_tokens)
        is_chapter_start = unit_node.get("type") == "chapter"

        if current_chunk_node is None: # Need to start a new chunk
            current_chunk_node = etree.SubElement(root_doc_element, "chunk")
            current_chunk_node.set("id", f"{chunk_id_counter:04d}")
            current_chunk_node.set("tok", "____")
            current_chunk_tokens = 0

        new_chunk_forced = False
        if is_chapter_start and current_chunk_tokens > 0 : # Chapters always start new chunks if current isn't empty
            new_chunk_forced = True
        elif unit_tokens > max_chunk_size and current_chunk_tokens > 0 : # Oversized unit needs new chunk if current isn't empty
             new_chunk_forced = True
        elif current_chunk_tokens + unit_tokens > max_chunk_size and current_chunk_tokens > 0: # Regular overflow
            new_chunk_forced = True

        if new_chunk_forced:
            chunk_id_counter += 1
            current_chunk_node = etree.SubElement(root_doc_element, "chunk")
            current_chunk_node.set("id", f"{chunk_id_counter:04d}")
            current_chunk_node.set("tok", "____")
            current_chunk_tokens = 0

        current_chunk_node.append(unit_node)
        current_chunk_tokens += unit_tokens

        # If this unit itself is oversized, this chunk is now oversized and done.
        if unit_tokens > max_chunk_size:
            chunk_id_counter += 1
            current_chunk_node = None

    logger.info(f"Organized units into {len(root_doc_element.xpath('./chunk'))} chunks.")
    return root_doc_element

def clean_consecutive_newlines(content: str) -> str:
    return re.sub(r"\n{3,}", "\n\n", content)


def pretty_print_xml_from_tree(root: etree._Element) -> str:
    """Format XML tree with proper indentation for readability."""
    return etree.tostring(
        root, encoding="utf-8", pretty_print=True, xml_declaration=True
    ).decode("utf-8")

def create_backup(file_path: str | Path, stage: str = "") -> Path | None:
    file_path = Path(file_path)
    if not file_path.exists():
        logger.warning(f"Cannot back up non-existent file: {file_path}")
        return None
    timestamp = datetime.datetime.now(datetime.UTC).strftime("%Y%m%d_%H%M%S")
    backup_name = f"{file_path.stem}_{stage}_{timestamp}{file_path.suffix}" if stage else f"{file_path.stem}_{timestamp}{file_path.suffix}"
    # Ensure backups are saved in the same directory as the output_file or a dedicated backup subdir
    backup_dir = file_path.parent / "backups"
    backup_dir.mkdir(parents=True, exist_ok=True)
    backup_path = backup_dir / backup_name
    try:
        import shutil
        shutil.copy2(file_path, backup_path)
        logger.debug(f"Created backup: {backup_path}")
        return backup_path
    except Exception as e:
        logger.error(f"Failed to create backup: {e}")
        return None

def save_tree_to_file(root: etree._Element, file_path: str | Path, pretty: bool = False):
    xml_string = etree.tostring(
        root, encoding="utf-8", method="xml",
        xml_declaration=True, pretty_print=pretty
    ).decode("utf-8")
    Path(file_path).write_text(xml_string, encoding="utf-8")


def process_document(
    input_file: str,
    output_file: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    model: str = DEFAULT_MODEL,
    temperature: float = DEFAULT_TEMPERATURE,
    encoding: str = "utf-8",
    verbose: bool = False, # verbose now controls pretty_print for final output
    backup: bool = False,
) -> dict[str, Any]:
    logger.info(f"Processing document: {input_file} -> {output_file}")
    logger.info(
        f"Using model: {model}, temperature: {temperature}, chunk size: {chunk_size}"
    )

    try:
        with open(input_file, encoding=encoding) as f:
            document_text = f.read().replace("\r\n", "\n")

        logger.info("Step 1: Itemizing document")
        itemized_paragraphs = itemize_document(document_text)

        logger.info("Step 2: Creating XML items and initial <doc> tree")
        items_xml_str_list = [item_tuple[0] for item_tuple in create_item_elements(itemized_paragraphs)]
        # Initial XML string for parsing. Ensure it's well-formed.
        doc_content_str = "\n".join(items_xml_str_list)
        initial_doc_str = f'<?xml version="1.0" encoding="UTF-8"?>\n<doc>\n{doc_content_str}\n</doc>'

        parser = etree.XMLParser(remove_blank_text=False, recover=True, encoding="utf-8")
        current_root = etree.XML(initial_doc_str.encode("utf-8"), parser)

        update_tok_attributes_on_tree(current_root)
        if backup:
            Path(output_file).parent.mkdir(parents=True, exist_ok=True) # Ensure output dir exists for backup
            save_tree_to_file(current_root, f"{output_file}.items.xml", pretty=verbose)
            create_backup(f"{output_file}.items.xml", "items")


        logger.info("Step 3: Performing semantic analysis")
        # Semantic analysis needs string representation of current tree
        current_doc_str_for_llm = etree.tostring(current_root, encoding="utf-8").decode("utf-8")
        semantic_boundaries = semantic_analysis(current_doc_str_for_llm, model, temperature)

        logger.info("Step 4: Adding unit tags to tree")
        current_root = add_unit_tags_on_tree(current_root, semantic_boundaries)
        update_tok_attributes_on_tree(current_root)
        if backup:
            save_tree_to_file(current_root, f"{output_file}.units.xml", pretty=verbose)
            create_backup(f"{output_file}.units.xml", "units")

        logger.info("Step 5: Creating chunks on tree")
        current_root = create_chunks_on_tree(current_root, chunk_size)
        update_tok_attributes_on_tree(current_root)
        if backup:
            save_tree_to_file(current_root, f"{output_file}.chunks.xml", pretty=verbose)
            create_backup(f"{output_file}.chunks.xml", "chunks")

        Path(output_file).parent.mkdir(parents=True, exist_ok=True) # Ensure output dir exists
        save_tree_to_file(current_root, output_file, pretty=verbose)

        items_count = len(current_root.xpath("//item"))
        units_count = len(current_root.xpath("//unit"))
        chunks_count = len(current_root.xpath("//chunk"))
        # total_tokens might need re-evaluation based on final serialized string if that's the definition
        final_xml_string_for_tokens = etree.tostring(current_root, encoding="utf-8").decode("utf-8")
        total_tokens = count_tokens(final_xml_string_for_tokens)

        logger.info(
            f"Processing complete: {items_count} items, {units_count} units, {chunks_count} chunks"
        )
        logger.info(f"Total tokens in final structure: {total_tokens}")

        return {
            "input_file": input_file, "output_file": output_file,
            "items_count": items_count, "units_count": units_count,
            "chunks_count": chunks_count, "total_tokens": total_tokens, "success": True
        }

    except Exception as e:
        logger.error(f"Processing failed: {e}", exc_info=True)
        fallback_xml = f"""<?xml version="1.0" encoding="UTF-8"?>
<doc><chunk id="error001" tok="1"><unit type="error" tok="1"><item xml:space="preserve" tok="1" id="error-item001">Error processing document: {escape_xml_chars(str(e))}</item></unit></chunk></doc>"""
        Path(output_file).parent.mkdir(parents=True, exist_ok=True)
        Path(output_file).write_text(fallback_xml, encoding="utf-8")
        return {"input_file": input_file, "output_file": output_file, "success": False, "error": str(e)}


def semantic_analysis( doc_text: str, model: str, temperature: float ) -> list[tuple[str, str]]:
    logger.info("Performing semantic analysis of document (tree-based path)")
    item_pattern = re.compile(r'<item[^>]*id="([^"]+)"[^>]*>.*?</item>', re.DOTALL)
    items_ids = item_pattern.findall(doc_text)

    if not items_ids:
        logger.warning("No items found in document for semantic analysis.")
        return []

    # This is a placeholder for actual LLM call logic
    # For testing, assume first item always starts a unit.
    # A real implementation would call an LLM.
    logger.warning("Using placeholder semantic analysis - first item is a 'unit'.")
    if items_ids:
        return [(items_ids[0], "unit")]
    return []

def extract_and_parse_llm_response(response) -> list[tuple[str, str]]:
    # This function is part of semantic_analysis; kept for modularity if LLM call is restored
    boundaries = []
    # Dummy implementation for now
    if hasattr(response, "choices") and response.choices:
        content = response.choices[0].message.content
        pattern = re.compile(r"BOUNDARY:\s*([^,]+),\s*([a-z]+)", re.IGNORECASE)
        matches = pattern.findall(content)
        for item_id, boundary_type in matches:
            item_id = item_id.strip()
            boundary_type = boundary_type.strip().lower()
            if boundary_type not in ["chapter", "scene", "unit"]:
                boundary_type = "unit"
            boundaries.append((item_id, boundary_type))
    return boundaries

# Removed old pretty_print_xml(xml_str) as pretty_print_xml_from_tree(etree_obj) is preferred.
# Removed regex-based fallbacks for add_unit_tags and create_chunks for now to simplify,
# focusing on the tree-based operations. If parsing fails early, it will be caught.
# Removed unused handle_oversized_unit and split_oversized_unit_regex
