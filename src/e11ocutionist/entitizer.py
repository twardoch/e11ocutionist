#!/usr/bin/env python3
# this_file: src/e11ocutionist/entitizer.py
"""
Entitizer module for e11ocutionist.

This module identifies and tags Named Entities of Interest (NEIs) in documents
for consistent pronunciation in speech synthesis.
"""

import json
import os
import re
from typing import Any
from copy import deepcopy

import backoff
from lxml import etree
from loguru import logger
from dotenv import load_dotenv

from e11ocutionist.utils import parse_xml, serialize_xml, create_backup

# Load environment variables from .env file
load_dotenv()

# Constants
DEFAULT_MODEL = "gpt-4o"
FALLBACK_MODEL = "gpt-4o-mini"
DEFAULT_TEMPERATURE = 0.1


def get_chunks(root: etree._Element) -> list[etree._Element]:
    """
    Extract chunks from the XML document.

    Args:
        root: Root XML element

    Returns:
        List of chunk elements

    Used in:
    - e11ocutionist/entitizer.py
    """
    return root.xpath("//chunk")


def get_chunk_text(chunk: etree._Element) -> str:
    """
    Extract the raw text content of a chunk while preserving whitespace.

    Args:
        chunk: XML chunk element

    Returns:
        Raw text content of the chunk with all whitespace preserved

    Used in:
    - e11ocutionist/entitizer.py
    """
    # Using etree.tostring to preserve all whitespace in the XML structure
    chunk_xml = etree.tostring(chunk, encoding="utf-8", method="xml").decode("utf-8")
    return chunk_xml


def save_current_state(
    root: etree._Element,
    nei_dict: dict[str, dict[str, Any]],
    output_file: str,
    nei_dict_file: str | None = None,
    chunk_id: str | None = None,
    backup: bool = False,
) -> None:
    """
    Save current state to the output file after processing a chunk.

    Args:
        root: Current XML root element
        nei_dict: Current NEI dictionary
        output_file: File to save output
        nei_dict_file: File to save NEI dictionary (defaults to output_file
                       with _nei_dict.json)
        chunk_id: ID of the chunk that was processed
        backup: Whether to create backup copies with timestamps

    Used in:
    - e11ocutionist/entitizer.py
    """
    try:
        # Save current XML state
        xml_string = serialize_xml(root)
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(xml_string)

        # Save current NEI dictionary to a JSON file
        if nei_dict_file is None:
            nei_dict_file = f"{os.path.splitext(output_file)[0]}_nei_dict.json"

        with open(nei_dict_file, "w", encoding="utf-8") as f:
            json.dump(nei_dict, f, ensure_ascii=False, indent=2)

        # Create backup if requested
        if backup:
            chunk_suffix = f"_{chunk_id}" if chunk_id else ""
            backup_file = create_backup(output_file, f"entitizer{chunk_suffix}")
            nei_backup_file = create_backup(nei_dict_file, f"nei{chunk_suffix}")

            if backup_file and nei_backup_file:
                logger.debug(f"Created backups at: {backup_file} and {nei_backup_file}")

        suffix = f" after processing chunk {chunk_id}" if chunk_id else ""
        logger.info(f"Saved current state{suffix}")
        logger.debug(f"Files saved to {output_file} and {nei_dict_file}")

    except Exception as e:
        logger.error(f"Error saving current state: {e}")


def extract_item_elements(xml_text: str) -> dict[str, str]:
    """
    Extract item elements from XML text and map them by ID.

    Args:
        xml_text: XML text containing item elements

    Returns:
        Dictionary mapping item IDs to their full XML content

    Used in:
    - e11ocutionist/entitizer.py
    """
    item_map = {}
    # Pattern to match complete <item> elements including their attributes and content
    pattern = r"<item\s+([^>]*)>(.*?)</item>"

    # Find all item elements
    for match in re.finditer(pattern, xml_text, re.DOTALL):
        # Extract attributes and content
        attributes = match.group(1)
        content = match.group(2)

        # Extract ID attribute
        id_match = re.search(r'id="([^"]*)"', attributes)
        if id_match:
            item_id = id_match.group(1)
            # Store the complete item element
            item_map[item_id] = f"<item {attributes}>{content}</item>"

    return item_map


def merge_tagged_items(
    original_chunk: etree._Element, tagged_items: dict[str, str]
) -> etree._Element:
    """
    Merge tagged items back into the original chunk.

    Args:
        original_chunk: Original XML chunk
        tagged_items: Dictionary of tagged items mapped by ID

    Returns:
        Updated chunk with merged tagged items

    Used in:
    - e11ocutionist/entitizer.py
    """
    # Create a deep copy of the original chunk to avoid modifying it
    updated_chunk = deepcopy(original_chunk)

    # Find all items in the updated chunk
    for item in updated_chunk.xpath(".//item"):
        item_id = item.get("id")
        if item_id in tagged_items:
            # Parse the tagged item
            try:
                tagged_item = etree.fromstring(
                    tagged_items[item_id].encode("utf-8"),
                    etree.XMLParser(remove_blank_text=False, recover=True),
                )

                # Replace the content of the original item with the tagged content
                # This preserves the attributes of the original item
                item.clear()

                # Copy attributes from tagged item
                for name, value in tagged_item.attrib.items():
                    item.set(name, value)

                # Copy children and text from tagged item
                if tagged_item.text:
                    item.text = tagged_item.text

                for child in tagged_item:
                    item.append(deepcopy(child))

                if tagged_item.tail:
                    item.tail = tagged_item.tail

            except Exception as e:
                logger.error(f"Error merging tagged item {item_id}: {e}")

    return updated_chunk


@backoff.on_exception(backoff.expo, Exception, max_tries=3)
def identify_entities(
    chunk_text: str, nei_dict: dict[str, dict[str, Any]], model: str, temperature: float
) -> str:
    """
    Identify Named Entities of Interest (NEIs) in a chunk of text using an LLM.

    Args:
        chunk_text: XML text of the chunk
        nei_dict: Current dictionary of NEIs
        model: LLM model to use
        temperature: Temperature setting for the LLM

    Returns:
        Chunk text with NEIs tagged

    Used in:
    - e11ocutionist/entitizer.py
    """
    try:
        # Import here to avoid circular imports
        from litellm import completion

        # Extract existing entities for the prompt
        existing_entities = []
        for _key, value in nei_dict.items():
            if "text" in value:
                entity_info = f"{value['text']}"
                if value.get("pronunciation"):
                    entity_info += f" (pronounced: {value['pronunciation']})"
                existing_entities.append(entity_info)

        # Construct the prompt
        prompt = f"""
You are an expert in Named Entity Recognition for text-to-speech applications.

Your task is to identify Named Entities of Interest (NEIs) in a document that might need special pronunciation guidance.
These entities should be tagged using the <nei> XML tag in the exact original text.

Types of entities to tag:
1. PROPER NAMES: People, places, organizations, brands, titles
2. TECHNICAL TERMS: Scientific, technical, or specialized terminology
3. FOREIGN WORDS: Non-English words or phrases
4. ABBREVIATIONS & ACRONYMS: Especially those that aren't obviously pronounced
5. UNUSUAL SPELLINGS: Words with non-standard spelling
6. NUMBERS & DATES: Complex numerical expressions that benefit from pronunciation guidance

DO NOT tag:
- Common words, even if capitalized at the start of sentences
- Regular English words with standard pronunciations
- Extremely common acronyms with obvious pronunciations (e.g., TV, DVD)

XML TAG FORMAT:
<nei>entity text</nei>

IMPORTANT RULES:
- Do NOT alter the original structure of the XML document
- Only add <nei> tags around the exact text of entities
- Preserve all existing XML tags and attributes
- Do NOT nest <nei> tags inside other tags or break existing tags
- Do NOT modify any other aspect of the text
- Only tag complete words or phrases, never partial words
- Maintain all whitespace exactly as in the original

{"PREVIOUSLY IDENTIFIED ENTITIES (already in the document elsewhere):\\n" + "\\n".join(existing_entities) if existing_entities else ""}

Here is the XML text to tag with NEIs:

{chunk_text}
"""

        # Call the LLM API
        logger.info(f"Calling LLM API with model: {model}")
        response = completion(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=None,  # Let the API determine the max tokens
        )

        # Extract the response text
        response_text = extract_response_text(response)
        return response_text

    except Exception as e:
        logger.error(f"Error in entity identification: {e}")
        # Return the original text if there's an error
        return chunk_text


def extract_response_text(response: Any) -> str:
    """
    Extract the text content from an LLM API response.

    Args:
        response: Response object from LLM API

    Returns:
        Extracted text content

    Used in:
    - e11ocutionist/entitizer.py
    """
    try:
        # Handle response object based on its structure
        if hasattr(response, "choices") and response.choices:
            if hasattr(response.choices[0], "message"):
                content = response.choices[0].message.content
            else:
                content = response.choices[0].text
        elif isinstance(response, dict):
            if "choices" in response:
                if "message" in response["choices"][0]:
                    content = response["choices"][0]["message"]["content"]
                else:
                    content = response["choices"][0]["text"]
            else:
                logger.error("Unexpected response format")
                content = str(response)
        else:
            logger.error("Unexpected response object")
            content = str(response)

        return content
    except Exception as e:
        logger.error(f"Error extracting response text: {e}")
        return ""


def extract_nei_from_tags(text: str) -> dict[str, dict[str, Any]]:
    """
    Extract NEIs from <nei> tags in text and build a dictionary.

    Args:
        text: XML text containing <nei> tags

    Returns:
        Dictionary of NEIs with their attributes

    Used in:
    - e11ocutionist/entitizer.py
    """
    nei_dict = {}
    # Pattern to match <nei> tags with optional attributes
    pattern = r"<nei(?:\s+([^>]*))?>(.*?)</nei>"

    for match in re.finditer(pattern, text, re.DOTALL):
        attributes_str = match.group(1) or ""
        content = match.group(2).strip() # Ensure content is stripped

        # Skip empty content
        if not content:
            continue

        key = content.lower()

        # Extract all attributes from the tag
        current_attributes = {}
        for attr_match in re.finditer(r'(\w+)="([^"]*)"', attributes_str):
            current_attributes[attr_match.group(1)] = attr_match.group(2)

        if key not in nei_dict:
            nei_dict[key] = {
                "text": content, # Store original cased content
                "count": 1,
                **current_attributes # Add all found attributes
            }
            # Explicitly set 'new' based on its presence in attributes,
            # rather than defaulting to True then False.
            # If 'new' attribute is literally "true", it's new. Otherwise, if attr not present, it's new.
            if "new" not in current_attributes:
                 nei_dict[key]["new"] = True
            elif current_attributes.get("new") != "true": # handles new="false" or other values
                 nei_dict[key]["new"] = False

        else:
            nei_dict[key]["count"] += 1
            # Update attributes if not already present or if they differ,
            # preferring existing ones unless a specific logic is needed.
            for attr_name, attr_value in current_attributes.items():
                if attr_name not in nei_dict[key]: # Add new attributes from this occurrence
                    nei_dict[key][attr_name] = attr_value

            # If it's seen again, it's not "new" overall for the dictionary,
            # unless this specific occurrence is marked new="true" AND we want to override.
            # Current logic: if an entity is re-encountered, its 'new' status in the dict becomes False.
            # unless this specific tag had new="true"
            if current_attributes.get("new") == "true":
                nei_dict[key]["new"] = True # This specific instance was marked new
            else:
                nei_dict[key]["new"] = False


    return nei_dict


def process_chunks(
    chunks: list[etree._Element],
    model: str,
    temperature: float,
    nei_dict: dict[str, dict[str, Any]] | None = None,
    output_file: str = "",
    nei_dict_file: str | None = None,
    backup: bool = False,
) -> tuple[etree._Element | None, dict[str, dict[str, Any]]]:
    """
    Process chunks to identify and tag NEIs.

    Args:
        chunks: List of XML chunk elements
        model: LLM model to use
        temperature: Temperature setting for the LLM
        nei_dict: Existing NEI dictionary (optional)
        output_file: File to save intermediate results (optional)
        nei_dict_file: File to save NEI dictionary (optional)
        backup: Whether to create backup copies with timestamps

    Returns:
        Tuple of (updated XML root, updated NEI dictionary)

    Used in:
    - e11ocutionist/entitizer.py
    """
    # Initialize the NEI dictionary if not provided
    if nei_dict is None:
        nei_dict = {}

    # Get the root element
    if chunks:
        root = chunks[0].getroottree().getroot()
    else:
        logger.warning("No chunks found in the document")
        return None, nei_dict

    # Process each chunk
    for i, chunk in enumerate(chunks):
        chunk_id = chunk.get("id", str(i))
        logger.info(f"Processing chunk {chunk_id} ({i + 1}/{len(chunks)})")

        # Get the text of the chunk
        chunk_text = get_chunk_text(chunk)

        # Identify entities in the chunk
        try:
            tagged_chunk_text = identify_entities(
                chunk_text, nei_dict, model, temperature
            )

            # Extract item elements from the tagged chunk
            tagged_items = extract_item_elements(tagged_chunk_text)

            # Merge tagged items back into the original chunk
            updated_chunk = merge_tagged_items(chunk, tagged_items)

            # Replace the original chunk with the updated one
            parent = chunk.getparent()
            if parent is not None:
                index = parent.index(chunk)
                parent.remove(chunk)
                parent.insert(index, updated_chunk)

            # Extract NEIs from the tagged chunk text and update the dictionary
            new_neis = extract_nei_from_tags(tagged_chunk_text)
            for key, value in new_neis.items():
                if key in nei_dict:
                    # Update existing NEI
                    nei_dict[key]["count"] += value["count"]
                    nei_dict[key]["new"] = False
                else:
                    # Add new NEI
                    nei_dict[key] = value

            # Save the current state if an output file is provided
            if output_file:
                save_current_state(
                    root, nei_dict, output_file, nei_dict_file, chunk_id, backup
                )

        except Exception as e:
            logger.error(f"Error processing chunk {chunk_id}: {e}")

    return root, nei_dict


def process_document(
    input_file: str,
    output_file: str,
    nei_dict_file: str | None = None,
    model: str = DEFAULT_MODEL,
    temperature: float = DEFAULT_TEMPERATURE,
    encoding: str = "utf-8",
    verbose: bool = False,
    backup: bool = False,
) -> dict[str, Any]:
    """
    Process a document to identify and tag Named Entities of Interest (NEIs).

    Args:
        input_file: Path to the input XML file
        output_file: Path to save the processed XML output
        nei_dict_file: Path to save/load the NEI dictionary
        model: LLM model to use
        temperature: Temperature setting for the LLM
        encoding: Character encoding of the input file
        verbose: Enable verbose logging
        backup: Create backups at intermediate stages

    Returns:
        Dictionary with processing statistics

    Used in:
    - e11ocutionist/__init__.py
    - e11ocutionist/cli.py
    - e11ocutionist/e11ocutionist.py
    - e11ocutionist/entitizer.py
    """
    logger.info(f"Processing document for NEI tagging: {input_file} -> {output_file}")

    try:
        # Configure logging level
        if verbose:
            logger.level("INFO")
        else:
            logger.level("WARNING")

        # Load existing NEI dictionary if available
        nei_dict = {}
        if nei_dict_file and os.path.exists(nei_dict_file):
            try:
                with open(nei_dict_file, encoding=encoding) as f:
                    nei_dict = json.load(f)
                logger.info(f"Loaded existing NEI dictionary from {nei_dict_file}")
            except Exception as e:
                logger.error(f"Error loading NEI dictionary: {e}")

        # Create a backup of the input file if requested
        if backup:
            input_backup = create_backup(input_file, "pre_entitize")
            if input_backup:
                logger.info(f"Created backup of input file: {input_backup}")

        # Parse the input XML
        with open(input_file, encoding=encoding) as f:
            xml_content = f.read()

        root = parse_xml(xml_content)
        if root is None:
            logger.error("Failed to parse input XML")
            return {
                "input_file": input_file,
                "output_file": output_file,
                "nei_dict_file": nei_dict_file,
                "success": False,
                "error": "Failed to parse input XML",
            }

        # Extract chunks from the document
        chunks = get_chunks(root)
        logger.info(f"Found {len(chunks)} chunks in the document")

        # Process the chunks
        updated_root, updated_nei_dict = process_chunks(
            chunks,
            model,
            temperature,
            nei_dict,
            output_file if backup else "",  # Only save intermediate if backup is True
            nei_dict_file,
            backup,
        )

        if updated_root is None:
            logger.error("Failed to process chunks")
            return {
                "input_file": input_file,
                "output_file": output_file,
                "nei_dict_file": nei_dict_file,
                "success": False,
                "error": "Failed to process chunks",
            }

        # Serialize the final XML
        final_xml = serialize_xml(updated_root)

        # Save the final output
        with open(output_file, "w", encoding=encoding) as f:
            f.write(final_xml)

        # Save the final NEI dictionary
        if nei_dict_file:
            with open(nei_dict_file, "w", encoding=encoding) as f:
                json.dump(updated_nei_dict, f, ensure_ascii=False, indent=2)

        # Count NEIs
        total_neis = len(updated_nei_dict)
        new_neis = sum(1 for nei in updated_nei_dict.values() if nei.get("new", False))

        logger.info(
            f"Successfully processed document: {total_neis} NEIs ({new_neis} new)"
        )
        logger.info(f"Saved processed document to {output_file}")
        if nei_dict_file:
            logger.info(f"Saved NEI dictionary to {nei_dict_file}")

        return {
            "input_file": input_file,
            "output_file": output_file,
            "nei_dict_file": nei_dict_file,
            "total_neis": total_neis,
            "new_neis": new_neis,
            "success": True,
        }

    except Exception as e:
        logger.error(f"Error processing document: {e}")
        return {
            "input_file": input_file,
            "output_file": output_file,
            "nei_dict_file": nei_dict_file,
            "success": False,
            "error": str(e),
        }
