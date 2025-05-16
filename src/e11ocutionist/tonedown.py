#!/usr/bin/env python3
# this_file: src/e11ocutionist/tonedown.py
"""
Tonedown module for e11ocutionist.

This module adjusts the tone of the text by reviewing and refining
pronunciation cues and reducing excessive emphasis.
"""

import json
from typing import Any
from loguru import logger
from lxml import etree
from dotenv import load_dotenv

from .utils import (
    count_tokens,
    parse_xml,
    serialize_xml,
)

# Load environment variables
load_dotenv()

# Constants
DEFAULT_MODEL = "gpt-4o"
FALLBACK_MODEL = "gpt-4o-mini"
DEFAULT_TEMPERATURE = 0.2


def extract_neis_from_document(xml_content: str) -> dict[str, dict[str, str]]:
    """
    Extract Named Entities of Interest (NEIs) and their attributes from a document.

    Args:
        xml_content: XML document content

    Returns:
        Dictionary of NEIs with their attributes
    """
    nei_dict = {}

    try:
        # Parse the XML
        root = parse_xml(xml_content)
        if root is None:
            logger.error("Failed to parse XML")
            return nei_dict

        # Find all NEI tags
        nei_tags = root.xpath("//nei")
        logger.info(f"Found {len(nei_tags)} NEI tags in document")

        # Process each NEI tag
        for nei in nei_tags:
            # Get the content (the entity name)
            nei_text = (nei.text or "").strip()
            if not nei_text:
                continue

            # Only process each unique NEI once (case-insensitive)
            nei_lower = nei_text.lower()

            # Get attributes
            is_new = nei.get("new", "false") == "true"
            orig_value = nei.get("orig", "")

            # Initialize or update the entry
            if nei_lower not in nei_dict:
                nei_dict[nei_lower] = {
                    "text": nei_text,  # Preserve original case
                    "new": is_new,
                    "orig": orig_value,
                    "count": 1,
                }
            else:
                # Increment count
                nei_dict[nei_lower]["count"] += 1

                # If this instance has orig when previous didn't, update it
                if orig_value and not nei_dict[nei_lower]["orig"]:
                    nei_dict[nei_lower]["orig"] = orig_value

                # If any instance is marked as new, the entity is considered new
                if is_new:
                    nei_dict[nei_lower]["new"] = True

        logger.info(f"Extracted {len(nei_dict)} unique NEIs")

    except Exception as e:
        logger.error(f"Error extracting NEIs: {e}")

    return nei_dict


def detect_language(
    xml_content: str, model: str, temperature: float
) -> tuple[str, float]:
    """
    Detect the dominant language of the document using an LLM.

    Args:
        xml_content: XML document content
        model: LLM model to use
        temperature: Temperature setting for the LLM

    Returns:
        Tuple of (language_code, confidence)
    """
    logger.info("Detecting document language")

    try:
        # Parse the XML to extract text content
        root = parse_xml(xml_content)
        if root is None:
            logger.error("Failed to parse XML")
            return ("en", 0.5)  # Default to English with low confidence

        # Extract text from items (up to a certain amount)
        text_samples = []
        sample_count = 0
        max_samples = 10

        for item in root.xpath("//item"):
            # Skip empty items
            if not (item.text and item.text.strip()):
                continue

            # Extract text content (without XML tags)
            item_text = etree.tostring(item, encoding="utf-8", method="text").decode(
                "utf-8"
            )

            # Add to samples if substantial
            if len(item_text.strip()) > 20:
                text_samples.append(item_text.strip())
                sample_count += 1

            # Limit the number of samples
            if sample_count >= max_samples:
                break

        # If we don't have enough samples, try to get more by finding text nodes
        if sample_count < 3:
            for text_node in root.xpath("//text()"):
                text = text_node.strip()
                if len(text) > 20:
                    text_samples.append(text)
                    sample_count += 1

                if sample_count >= max_samples:
                    break

        # If still not enough, default to English
        if sample_count < 2:
            logger.warning("Not enough text samples for language detection")
            return ("en", 0.5)

        # Combine samples
        "\n\n".join(text_samples[:5])  # Use up to 5 samples

        # Create the prompt

        # Call the LLM API (placeholder)
        logger.info(f"Calling LLM API with model: {model}")

        # Placeholder for LLM call
        response = {
            "choices": [
                {"message": {"content": '{"language": "en", "confidence": 0.95}'}}
            ]
        }

        # Extract the response
        if hasattr(response, "choices") and response.choices:
            content = response.choices[0].message.content
        elif isinstance(response, dict) and "choices" in response:
            content = response["choices"][0]["message"]["content"]
        else:
            logger.warning("Unexpected response format from LLM")
            return ("en", 0.5)

        # Parse the JSON response
        try:
            result = json.loads(content)
            language = result.get("language", "en")
            confidence = result.get("confidence", 0.5)

            logger.info(f"Detected language: {language} (confidence: {confidence:.2f})")
            return (language, confidence)

        except json.JSONDecodeError:
            logger.error("Failed to parse LLM response as JSON")
            return ("en", 0.5)

    except Exception as e:
        logger.error(f"Language detection failed: {e}")
        return ("en", 0.5)


def review_pronunciations(
    nei_dict: dict[str, dict[str, str]], language: str, model: str, temperature: float
) -> dict[str, dict[str, str]]:
    """
    Review and refine pronunciation cues for NEIs using an LLM.

    Args:
        nei_dict: Dictionary of NEIs with their attributes
        language: Detected language code
        model: LLM model to use
        temperature: Temperature setting for the LLM

    Returns:
        Updated NEI dictionary with refined pronunciations
    """
    if not nei_dict:
        logger.warning("No NEIs to review")
        return nei_dict

    logger.info(f"Reviewing pronunciations for {len(nei_dict)} NEIs in {language}")

    # Create a copy of the dictionary to update
    updated_dict = {k: v.copy() for k, v in nei_dict.items()}

    # Process NEIs in batches to avoid token limits
    batch_size = 25
    nei_items = list(nei_dict.items())
    batches = [
        nei_items[i : i + batch_size] for i in range(0, len(nei_items), batch_size)
    ]

    for batch_index, batch in enumerate(batches):
        logger.info(
            f"Processing batch {batch_index + 1}/{len(batches)} ({len(batch)} NEIs)"
        )

        # Prepare the batch for the LLM
        nei_json = {}
        for _key, value in batch:
            nei_json[value["text"]] = {
                "orig": value.get("orig", ""),
                "count": value.get("count", 1),
                "new": value.get("new", False),
            }

        # Create the prompt
        f"""
You are reviewing pronunciations for Named Entities of Interest (NEIs) to improve text-to-speech.
The primary language of the document is: {language}

For each entity, provide a pronunciation guide that will help a text-to-speech system
pronounce it correctly. Focus on entities that might be mispronounced
(names, foreign words, technical terms, acronyms).

Guidelines:
1. For common words with standard pronunciation, leave the pronunciation empty
2. For names and unusual words, provide phonetic spelling (e.g., "Kayleigh" → "Kay-lee")
3. For acronyms, specify whether to spell out or pronounce as a word (e.g., "NASA" → "Nasa as one word")
4. For foreign words, provide an English approximation or pronunciation note
5. For initialisms, add periods or spaces between letters (e.g., "IBM" → "I.B.M.")

The NEIs are in this format:
{{
  "Entity Text": {{
    "orig": "original form if different",
    "count": number of occurrences,
    "new": true/false if this is a new entity
  }},
  ...
}}

Return a JSON object where each key is the entity and each value is the pronunciation guide.
Return ONLY the JSON, no explanation.

NEIs to review:
{json.dumps(nei_json, ensure_ascii=False, indent=2)}
"""

        try:
            # Call the LLM API (placeholder)
            logger.info(f"Calling LLM API with model: {model}")

            # Placeholder for LLM call
            response = {
                "choices": [
                    {
                        "message": {
                            "content": '{"NASA": "Nasa as one word", "Kayleigh": "Kay-lee"}'
                        }
                    }
                ]
            }

            # Extract the response
            if hasattr(response, "choices") and response.choices:
                content = response.choices[0].message.content
            elif isinstance(response, dict) and "choices" in response:
                content = response["choices"][0]["message"]["content"]
            else:
                logger.warning("Unexpected response format from LLM")
                continue

            # Parse the JSON response
            try:
                result = json.loads(content)

                # Update pronunciations in the dictionary
                for entity, pronunciation in result.items():
                    entity_lower = entity.lower()
                    if entity_lower in updated_dict:
                        updated_dict[entity_lower]["pronunciation"] = pronunciation
                    elif entity in updated_dict:
                        updated_dict[entity]["pronunciation"] = pronunciation

                logger.info(f"Updated pronunciations for batch {batch_index + 1}")

            except json.JSONDecodeError:
                logger.error("Failed to parse LLM response as JSON")

        except Exception as e:
            logger.error(f"Error reviewing pronunciations: {e}")

    # Count how many pronunciations were added
    pronunciation_count = sum(
        1 for nei in updated_dict.values() if nei.get("pronunciation")
    )
    logger.info(f"Added or updated {pronunciation_count} pronunciations")

    return updated_dict


def update_nei_tags(xml_content: str, nei_dict: dict[str, dict[str, str]]) -> str:
    """
    Update NEI tags in the document with refined pronunciations.

    Args:
        xml_content: XML document content
        nei_dict: Dictionary of NEIs with their attributes and pronunciations

    Returns:
        Updated XML document
    """
    if not nei_dict:
        return xml_content

    logger.info("Updating NEI tags with refined pronunciations")

    try:
        # Parse the XML
        root = parse_xml(xml_content)
        if root is None:
            logger.error("Failed to parse XML")
            return xml_content

        # Create a fast lookup for NEIs (case-insensitive)
        nei_lookup = {}
        for key, value in nei_dict.items():
            nei_lookup[key.lower()] = value
            if "text" in value:
                nei_lookup[value["text"].lower()] = value

        # Find all NEI tags
        update_count = 0
        for nei in root.xpath("//nei"):
            # Get the content (the entity name)
            nei_text = (nei.text or "").strip()
            if not nei_text:
                continue

            # Look up this NEI
            nei_key = nei_text.lower()
            if nei_key in nei_lookup and "pronunciation" in nei_lookup[nei_key]:
                pronunciation = nei_lookup[nei_key]["pronunciation"]

                # Only update if there's a pronunciation
                if pronunciation:
                    # Set the content to the pronunciation
                    nei.text = pronunciation

                    # Set the orig attribute to the original text if not already set
                    if not nei.get("orig"):
                        nei.set("orig", nei_text)

                    update_count += 1

        logger.info(f"Updated {update_count} NEI tags with pronunciations")

        # Serialize the updated XML
        updated_xml = serialize_xml(root)
        return updated_xml

    except Exception as e:
        logger.error(f"Error updating NEI tags: {e}")
        return xml_content


def reduce_emphasis(xml_content: str, min_distance: int = 50) -> str:
    """
    Reduce excessive emphasis tags in the document.

    Args:
        xml_content: XML document content
        min_distance: Minimum token distance between emphasis tags

    Returns:
        Updated XML document with reduced emphasis
    """
    logger.info(f"Reducing excessive emphasis (min distance: {min_distance} tokens)")

    try:
        # Parse the XML
        root = parse_xml(xml_content)
        if root is None:
            logger.error("Failed to parse XML")
            return xml_content

        # Find all emphasis tags
        em_tags = root.xpath("//em")
        logger.info(f"Found {len(em_tags)} emphasis tags")

        if len(em_tags) <= 1:
            logger.info("No excess emphasis to reduce")
            return xml_content

        # Get the positions of emphasis tags
        em_positions = []
        for em in em_tags:
            # Get the position in the document (approximate token position)
            # This is approximate as we're not counting tokens precisely
            ancestor_text = ""
            for ancestor in em.xpath("ancestor::*"):
                if ancestor.text:
                    ancestor_text += ancestor.text

            position = count_tokens(ancestor_text)
            em_positions.append((em, position))

        # Sort by position
        em_positions.sort(key=lambda x: x[1])

        # Identify tags to remove (too close to others)
        to_remove = []
        last_position = -min_distance  # Ensure we keep the first one

        for em, position in em_positions:
            if position - last_position < min_distance:
                # Too close to previous, mark for removal
                to_remove.append(em)
            else:
                # Keep this one
                last_position = position

        # Remove marked tags (replace with their content)
        for em in to_remove:
            parent = em.getparent()
            if parent is None:
                continue

            # Get the index of this element in its parent
            index = parent.index(em)

            # Replace the emphasis tag with its content
            if em.text:
                if index == 0:
                    # First child, add to parent's text
                    parent.text = (parent.text or "") + em.text
                else:
                    # Not first child, add to previous sibling's tail
                    prev = parent[index - 1]
                    prev.tail = (prev.tail or "") + em.text

            # Handle any tail text
            if em.tail:
                if index == len(parent) - 1:
                    # Last child, add to parent's tail
                    parent.tail = (parent.tail or "") + em.tail
                else:
                    # Not last child, add to this element's replacement or next sibling
                    next_elem = parent[index + 1]
                    next_elem.tail = (next_elem.tail or "") + em.tail

            # Remove the emphasis tag
            parent.remove(em)

        logger.info(f"Removed {len(to_remove)} excessive emphasis tags")

        # Serialize the updated XML
        updated_xml = serialize_xml(root)
        return updated_xml

    except Exception as e:
        logger.error(f"Error reducing emphasis: {e}")
        return xml_content


def process_document(
    input_file: str,
    output_file: str,
    nei_dict_file: str | None = None,
    min_emphasis_distance: int = 50,
    model: str = DEFAULT_MODEL,
    temperature: float = DEFAULT_TEMPERATURE,
    encoding: str = "utf-8",
    verbose: bool = False,
    backup: bool = False,
) -> dict[str, Any]:
    """
    Process a document to adjust tone for speech synthesis.

    This function:
    1. Extracts Named Entities of Interest (NEIs) from the document
    2. Detects the document language
    3. Reviews and refines pronunciation cues for NEIs
    4. Updates NEI tags with refined pronunciations
    5. Reduces excessive emphasis tags

    Args:
        input_file: Path to the input XML file
        output_file: Path to save the processed XML output
        nei_dict_file: Path to save/load the NEI dictionary
        min_emphasis_distance: Minimum distance between emphasis tags
        model: LLM model to use
        temperature: Temperature setting for the LLM
        encoding: Character encoding of the input file
        verbose: Enable verbose logging
        backup: Create backups at intermediate stages

    Returns:
        Dictionary with processing statistics
    """
    logger.info(
        f"Processing document for tone adjustment: {input_file} -> {output_file}"
    )

    try:
        # Read the input file
        with open(input_file, encoding=encoding) as f:
            xml_content = f.read()

        # Step 1: Extract NEIs from the document
        logger.info("Step 1: Extracting NEIs from document")
        nei_dict = extract_neis_from_document(xml_content)

        # Save NEI dictionary if requested
        if nei_dict_file and backup:
            backup_file = f"{nei_dict_file}.extracted"
            with open(backup_file, "w", encoding=encoding) as f:
                json.dump(nei_dict, f, ensure_ascii=False, indent=2)
            logger.info(f"Saved extracted NEI dictionary to {backup_file}")

        # Step 2: Detect document language
        logger.info("Step 2: Detecting document language")
        language, confidence = detect_language(xml_content, model, temperature)

        # Step 3: Review pronunciations if we have NEIs
        if nei_dict:
            logger.info("Step 3: Reviewing pronunciations")
            nei_dict = review_pronunciations(nei_dict, language, model, temperature)

            # Save updated NEI dictionary if requested
            if nei_dict_file:
                with open(nei_dict_file, "w", encoding=encoding) as f:
                    json.dump(nei_dict, f, ensure_ascii=False, indent=2)
                logger.info(f"Saved NEI dictionary to {nei_dict_file}")

            # Save backup if requested
            if backup:
                backup_file = f"{output_file}.nei"
                with open(backup_file, "w", encoding=encoding) as f:
                    updated_xml = update_nei_tags(xml_content, nei_dict)
                    f.write(updated_xml)
                logger.info(f"Saved NEI-updated XML to {backup_file}")

        # Step 4: Update NEI tags in the document
        logger.info("Step 4: Updating NEI tags")
        current_xml = update_nei_tags(xml_content, nei_dict)

        # Step 5: Reduce excessive emphasis
        logger.info("Step 5: Reducing excessive emphasis")
        current_xml = reduce_emphasis(current_xml, min_emphasis_distance)

        # Save the final output
        with open(output_file, "w", encoding=encoding) as f:
            f.write(current_xml)
        logger.info(f"Saved tone-adjusted document to {output_file}")

        return {
            "input_file": input_file,
            "output_file": output_file,
            "nei_dict_file": nei_dict_file,
            "language": language,
            "language_confidence": confidence,
            "neis_processed": len(nei_dict),
            "success": True,
        }

    except Exception as e:
        logger.error(f"Tone adjustment failed: {e}")
        return {
            "input_file": input_file,
            "output_file": output_file,
            "nei_dict_file": nei_dict_file,
            "success": False,
            "error": str(e),
        }
