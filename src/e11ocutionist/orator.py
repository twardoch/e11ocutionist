#!/usr/bin/env python3
# this_file: src/e11ocutionist/orator.py
"""
Orator module for e11ocutionist.

This module enhances text for speech synthesis by restructuring sentences,
normalizing words, enhancing punctuation, and adding emotional emphasis.
"""

import re
from typing import Any
from loguru import logger
from lxml import etree
from dotenv import load_dotenv

from .utils import (
    parse_xml,
    serialize_xml,
)

# Load environment variables
load_dotenv()

# Constants
DEFAULT_MODEL = "gpt-4o"
FALLBACK_MODEL = "gpt-4o-mini"
DEFAULT_TEMPERATURE = 0.2


def extract_chunk_items(xml_content: str) -> list[tuple[str, str, str]]:
    """
    Extract items from an XML document.

    Args:
        xml_content: XML string

    Returns:
        List of tuples containing (item_id, item_content, item_xml)
    """
    items = []

    try:
        # Parse the XML
        root = parse_xml(xml_content)
        if root is None:
            logger.error("Failed to parse XML")
            return items

        # Find all items
        for item in root.xpath("//item"):
            item_id = item.get("id", "")

            # Get the item content (preserving any XML tags inside)
            item_content = etree.tostring(item, encoding="utf-8").decode("utf-8")
            # Extract just the content between the item tags
            item_content = re.sub(
                r"^<item[^>]*>(.*)</item>$", r"\1", item_content, flags=re.DOTALL
            )

            # Get the full item XML
            item_xml = etree.tostring(item, encoding="utf-8").decode("utf-8")

            items.append((item_id, item_content, item_xml))

    except Exception as e:
        logger.error(f"Error extracting items: {e}")

    return items


def restructure_sentences(
    items: list[tuple[str, str, str]], model: str, temperature: float
) -> list[tuple[str, str]]:
    """
    Restructure sentences for better speech synthesis.

    This function divides long sentences into shorter ones and
    ensures paragraphs and headings end with punctuation.

    Args:
        items: List of tuples (item_id, item_content, item_xml)
        model: LLM model to use
        temperature: Temperature setting for the LLM

    Returns:
        List of tuples (item_id, processed_content)
    """
    if not items:
        return []

    # Process items in batches to avoid token limits
    max_items_per_batch = 10
    batches = [
        items[i : i + max_items_per_batch]
        for i in range(0, len(items), max_items_per_batch)
    ]

    processed_items = []

    for batch in batches:
        # Prepare text for the LLM
        items_text = ""
        for i, (item_id, content, _) in enumerate(batch):
            items_text += f"ITEM {i + 1} (ID: {item_id}):\n{content}\n\n"

        # Construct the prompt

        try:
            # Call the LLM API (implementation will depend on your LLM library)
            # This is a placeholder for the actual API call
            logger.info(f"Calling LLM API with model: {model}")

            # Placeholder for LLM call
            response = {
                "choices": [
                    {
                        "message": {
                            "content": "ITEM 1 (ID: 000000-123456): Processed content."
                        }
                    }
                ]
            }

            # Process the response
            restructured_batch = extract_processed_items_from_response(response, batch)
            processed_items.extend(restructured_batch)

        except Exception as e:
            logger.error(f"Error restructuring sentences: {e}")
            # Add the original content as fallback
            for item_id, content, _ in batch:
                processed_items.append((item_id, content))

    return processed_items


def normalize_words(
    items: list[tuple[str, str]], model: str, temperature: float
) -> list[tuple[str, str]]:
    """
    Normalize words for better speech synthesis.

    This function converts digits to numeric words, replaces symbols,
    and spells out rare symbols.

    Args:
        items: List of tuples (item_id, item_content)
        model: LLM model to use
        temperature: Temperature setting for the LLM

    Returns:
        List of tuples (item_id, processed_content)
    """
    if not items:
        return []

    # Process items in batches to avoid token limits
    max_items_per_batch = 10
    batches = [
        items[i : i + max_items_per_batch]
        for i in range(0, len(items), max_items_per_batch)
    ]

    processed_items = []

    for batch in batches:
        # Prepare text for the LLM
        items_text = ""
        for i, (item_id, content) in enumerate(batch):
            items_text += f"ITEM {i + 1} (ID: {item_id}):\n{content}\n\n"

        # Construct the prompt

        try:
            # Call the LLM API
            logger.info(f"Calling LLM API with model: {model}")

            # Placeholder for LLM call
            response = {
                "choices": [
                    {
                        "message": {
                            "content": "ITEM 1 (ID: 000000-123456): Processed content."
                        }
                    }
                ]
            }

            # Process the response
            normalized_batch = extract_processed_items_from_response(response, batch)
            processed_items.extend(normalized_batch)

        except Exception as e:
            logger.error(f"Error normalizing words: {e}")
            # Add the original content as fallback
            for item_id, content in batch:
                processed_items.append((item_id, content))

    return processed_items


def enhance_punctuation(items: list[tuple[str, str]]) -> list[tuple[str, str]]:
    """
    Enhance punctuation for better speech synthesis.

    This function enhances punctuation algorithmically without using an LLM.

    Args:
        items: List of tuples (item_id, item_content)

    Returns:
        List of tuples (item_id, processed_content)
    """
    enhanced_items = []

    for item_id, content in items:
        # Process parenthetical expressions
        # Add commas before and after parentheses if missing
        processed = re.sub(r"(\w)\(", r"\1, (", content)
        processed = re.sub(r"\)(\w)", r"), \1", processed)

        # Replace em/en dashes with ellipses when used as pauses
        # But preserve em dashes at the beginning of dialog
        processed = re.sub(r"(?<!^)[\u2014\u2013](?!\w)", "...", processed)

        # Add a subtle pause after colons
        processed = re.sub(r":(\s*\w)", r": <hr/>\1", processed)

        # Add pauses around quoted speech
        processed = re.sub(r'([.!?])"(\s*[A-Z])', r'\1" <hr/>\2', processed)

        # Add appropriate pacing for lists
        processed = re.sub(r"(\d+\.\s*\w+);", r"\1, ", processed)

        # Clean up excessive commas
        processed = re.sub(r",\s*,", r",", processed)

        enhanced_items.append((item_id, processed))

    return enhanced_items


def add_emotional_emphasis(
    items: list[tuple[str, str]], model: str, temperature: float
) -> list[tuple[str, str]]:
    """
    Add emotional emphasis for more expressive speech.

    This function adds <em> tags for emphasis and <hr/> for dramatic pauses.

    Args:
        items: List of tuples (item_id, item_content)
        model: LLM model to use
        temperature: Temperature setting for the LLM

    Returns:
        List of tuples (item_id, processed_content)
    """
    if not items:
        return []

    # Process items in batches to avoid token limits
    max_items_per_batch = 10
    batches = [
        items[i : i + max_items_per_batch]
        for i in range(0, len(items), max_items_per_batch)
    ]

    processed_items = []

    for batch in batches:
        # Prepare text for the LLM
        items_text = ""
        for i, (item_id, content) in enumerate(batch):
            items_text += f"ITEM {i + 1} (ID: {item_id}):\n{content}\n\n"

        # Construct the prompt

        try:
            # Call the LLM API
            logger.info(f"Calling LLM API with model: {model}")

            # Placeholder for LLM call
            response = {
                "choices": [
                    {
                        "message": {
                            "content": "ITEM 1 (ID: 000000-123456): <em>Processed</em> content."
                        }
                    }
                ]
            }

            # Process the response
            emphasized_batch = extract_processed_items_from_response(response, batch)
            processed_items.extend(emphasized_batch)

        except Exception as e:
            logger.error(f"Error adding emotional emphasis: {e}")
            # Add the original content as fallback
            for item_id, content in batch:
                processed_items.append((item_id, content))

    return processed_items


def extract_processed_items_from_response(
    response, original_items
) -> list[tuple[str, str]]:
    """
    Extract processed items from LLM response.

    Args:
        response: LLM API response
        original_items: Original items that were sent to the LLM

    Returns:
        List of tuples (item_id, processed_content)
    """
    processed_items = []

    # Extract the response content
    if hasattr(response, "choices") and response.choices:
        content = response.choices[0].message.content
    elif isinstance(response, dict) and "choices" in response:
        content = response["choices"][0]["message"]["content"]
    else:
        logger.warning("Unexpected response format from LLM")
        # Use original items as fallback
        if isinstance(original_items[0], tuple) and len(original_items[0]) >= 2:
            return [(item[0], item[1]) for item in original_items]
        return []

    # Create a map of IDs to original content
    id_map = {}
    for item in original_items:
        if isinstance(item, tuple):
            if len(item) >= 3:
                id_map[item[0]] = item[1]  # (item_id, content, item_xml)
            elif len(item) >= 2:
                id_map[item[0]] = item[1]  # (item_id, content)

    # Extract processed items
    pattern = re.compile(
        r"ITEM \d+\s*\(ID:\s*([^)]+)\):\s*(.*?)(?=ITEM \d+|$)", re.DOTALL
    )
    matches = pattern.findall(content)

    for item_id, processed_content in matches:
        item_id = item_id.strip()
        if item_id in id_map:
            processed_items.append((item_id, processed_content.strip()))

    # Check if we got all items
    if len(processed_items) < len(original_items):
        logger.warning(
            f"Some items were not processed: got {len(processed_items)}, expected {len(original_items)}"
        )
        # Add missing items with original content
        found_ids = {item_id for item_id, _ in processed_items}
        for item in original_items:
            if isinstance(item, tuple) and len(item) >= 2:
                item_id = item[0]
                if item_id not in found_ids:
                    processed_items.append((item_id, id_map.get(item_id, "")))

    return processed_items


def merge_processed_items(
    original_xml: str, processed_items: list[tuple[str, str]]
) -> str:
    """
    Merge processed items back into the original XML.

    Args:
        original_xml: Original XML document
        processed_items: List of tuples (item_id, processed_content)

    Returns:
        Updated XML document
    """
    try:
        # Create a mapping of item IDs to processed content
        processed_map = dict(processed_items)

        # Parse the XML
        root = parse_xml(original_xml)
        if root is None:
            logger.error("Failed to parse original XML")
            return original_xml

        # Find all items and update their content
        for item in root.xpath("//item"):
            item_id = item.get("id", "")

            if item_id in processed_map:
                # Get the processed content
                processed_content = processed_map[item_id]

                # Clear the item's existing content
                item.text = None
                for child in item:
                    item.remove(child)

                # Create a temporary root element with the processed content
                # This preserves any XML tags in the processed content
                temp_xml = f"<root>{processed_content}</root>"
                temp_root = parse_xml(temp_xml)

                if temp_root is not None:
                    # Copy content from temp_root to item
                    if temp_root.text:
                        item.text = temp_root.text

                    for child in temp_root:
                        item.append(child)
                else:
                    # Fallback: just set the text (losing any XML tags)
                    item.text = processed_content

        # Serialize the updated XML
        updated_xml = serialize_xml(root)
        return updated_xml

    except Exception as e:
        logger.error(f"Error merging processed items: {e}")
        return original_xml


def process_document(
    input_file: str,
    output_file: str,
    steps: list[str] | None = None,
    model: str = DEFAULT_MODEL,
    temperature: float = DEFAULT_TEMPERATURE,
    encoding: str = "utf-8",
    verbose: bool = False,
    backup: bool = False,
) -> dict[str, Any]:
    """
    Process a document through the speech enhancement pipeline.

    This function enhances text for speech synthesis by:
    1. Restructuring sentences (dividing long sentences)
    2. Normalizing words (converting digits to words, replacing symbols)
    3. Enhancing punctuation (adding pauses, fixing parentheses)
    4. Adding emotional emphasis (with <em> tags and <hr/> tags)

    Args:
        input_file: Path to the input XML file
        output_file: Path to save the processed XML output
        steps: List of steps to perform (sentences, words, punctuation, emphasis)
               If None, all steps are performed
        model: LLM model to use
        temperature: Temperature setting for the LLM
        encoding: Character encoding of the input file
        verbose: Enable verbose logging
        backup: Create backups at intermediate stages

    Returns:
        Dictionary with processing statistics
    """
    logger.info(
        f"Processing document for speech enhancement: {input_file} -> {output_file}"
    )

    # Determine which steps to perform
    all_steps = ["sentences", "words", "punctuation", "emphasis"]
    if steps is None:
        steps = all_steps
    else:
        # Validate steps
        steps = [s for s in steps if s in all_steps]

    logger.info(f"Performing steps: {', '.join(steps)}")

    try:
        # Read the input file
        with open(input_file, encoding=encoding) as f:
            xml_content = f.read()

        # Extract items from the document
        items = extract_chunk_items(xml_content)
        logger.info(f"Extracted {len(items)} items for processing")

        # Create a copy of the original XML
        current_xml = xml_content

        # Process each step sequentially
        processed_items = [(item_id, content) for item_id, content, _ in items]

        # Step 1: Restructure sentences
        if "sentences" in steps:
            logger.info("Step 1: Restructuring sentences")
            processed_items = restructure_sentences(items, model, temperature)

            # Merge processed items back into XML
            current_xml = merge_processed_items(current_xml, processed_items)

            # Save backup if requested
            if backup:
                backup_file = f"{output_file}.sentences"
                with open(backup_file, "w", encoding=encoding) as f:
                    f.write(current_xml)
                logger.info(f"Saved sentences backup to {backup_file}")

        # Step 2: Normalize words
        if "words" in steps:
            logger.info("Step 2: Normalizing words")
            processed_items = normalize_words(processed_items, model, temperature)

            # Merge processed items back into XML
            current_xml = merge_processed_items(current_xml, processed_items)

            # Save backup if requested
            if backup:
                backup_file = f"{output_file}.words"
                with open(backup_file, "w", encoding=encoding) as f:
                    f.write(current_xml)
                logger.info(f"Saved words backup to {backup_file}")

        # Step 3: Enhance punctuation
        if "punctuation" in steps:
            logger.info("Step 3: Enhancing punctuation")
            processed_items = enhance_punctuation(processed_items)

            # Merge processed items back into XML
            current_xml = merge_processed_items(current_xml, processed_items)

            # Save backup if requested
            if backup:
                backup_file = f"{output_file}.punctuation"
                with open(backup_file, "w", encoding=encoding) as f:
                    f.write(current_xml)
                logger.info(f"Saved punctuation backup to {backup_file}")

        # Step 4: Add emotional emphasis
        if "emphasis" in steps:
            logger.info("Step 4: Adding emotional emphasis")
            processed_items = add_emotional_emphasis(
                processed_items, model, temperature
            )

            # Merge processed items back into XML
            current_xml = merge_processed_items(current_xml, processed_items)

            # Save backup if requested
            if backup:
                backup_file = f"{output_file}.emphasis"
                with open(backup_file, "w", encoding=encoding) as f:
                    f.write(current_xml)
                logger.info(f"Saved emphasis backup to {backup_file}")

        # Save the final output
        with open(output_file, "w", encoding=encoding) as f:
            f.write(current_xml)
        logger.info(f"Saved enhanced document to {output_file}")

        return {
            "input_file": input_file,
            "output_file": output_file,
            "steps_performed": steps,
            "items_processed": len(processed_items),
            "success": True,
        }

    except Exception as e:
        logger.error(f"Speech enhancement failed: {e}")
        return {
            "input_file": input_file,
            "output_file": output_file,
            "steps_performed": [],
            "success": False,
            "error": str(e),
        }
