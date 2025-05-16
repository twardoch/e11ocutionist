#!/usr/bin/env -S uv run -s
# /// script
# dependencies = ["fire", "litellm<=1.67.2", "lxml", "backoff", "python-dotenv", "loguru", "types-lxml"]
# ///
# this_file: malmo_orator.py

import fire
import re
import os
import backoff
import datetime
from pathlib import Path
from lxml import etree
from litellm import completion
from litellm._logging import _turn_on_debug
from dotenv import load_dotenv
from loguru import logger
from typing import Any, cast

# Load environment variables from .env file
load_dotenv()

# Global variables
DEFAULT_MODEL = "openrouter/openai/gpt-4.1"
FALLBACK_MODEL = "openrouter/google/gemini-2.5-pro-preview-03-25"
DEFAULT_TEMPERATURE = 1.0

# Dictionary mapping model keys to their actual model identifiers
MODEL_DICT = {
    "geminipro": "openrouter/google/gemini-2.5-pro-preview-03-25",
    "claudeopus": "openrouter/anthropic/claude-3-opus",
    "gpt41": "openrouter/openai/gpt-4.1",
    "gpt41mini": "openrouter/openai/gpt-4.1-mini",
    "r1chimera": "openrouter/tngtech/deepseek-r1t-chimera:free",
    "msr1": "openrouter/microsoft/mai-ds-r1:free",
    "geminiflash": "openrouter/google/gemini-2.5-flash-preview",
    "gpt41nano": "openrouter/openai/gpt-4.1-nano",
    "llama4mav": "openrouter/meta-llama/llama-4-maverick:free",
    "llama4sc": "openrouter/meta-llama/llama-4-scout:free",
    "claudesonnet": "openrouter/anthropic/claude-3.7-sonnet",
    "nemotron": "openrouter/nvidia/llama-3.1-nemotron-ultra-253b-v1:free",
    "qwenturbo": "openrouter/qwen/qwen-turbo",
    "geminiflash2": "openrouter/google/gemini-2.0-flash-001",
    "r1": "openrouter/deepseek/deepseek-r1:free",
    "deepseek3": "openrouter/deepseek/deepseek-chat:free",
    "geminiflash2e": "openrouter/google/gemini-2.0-flash-exp:free",
    "geminiflash1": "openrouter/google/gemini-flash-1.5-8b",
}

# Check if API key is set
if not os.getenv("OPENROUTER_API_KEY"):
    logger.warning("No API key found in environment variables. API calls may fail.")


def parse_xml(input_file: str) -> etree._Element:
    """
    Parse the input XML file.

    Args:
        input_file: Path to the input XML file

    Returns:
        Parsed XML element tree
    """
    try:
        parser = etree.XMLParser(remove_blank_text=False, recover=True)
        with open(input_file, encoding="utf-8") as f:
            xml_content = f.read()
        return etree.fromstring(xml_content.encode("utf-8"), parser)
    except Exception as e:
        logger.error(f"Error parsing XML: {e}")
        raise


def get_chunks(root: etree._Element) -> list[etree._Element]:
    """
    Extract chunks from the XML document.

    Args:
        root: Root XML element

    Returns:
        List of chunk elements
    """
    return root.xpath("//chunk")


def get_items_from_chunk(chunk: etree._Element) -> list[etree._Element]:
    """
    Extract all item elements from a chunk.

    Args:
        chunk: Chunk element

    Returns:
        List of item elements
    """
    return chunk.xpath(".//item")


def extract_item_elements(xml_text: str) -> dict[str, str]:
    """
    Extract item elements from XML text and map them by ID.

    Args:
        xml_text: XML text containing item elements

    Returns:
        Dictionary mapping item IDs to their full XML content
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
    Merge tagged items back into the original chunk structure.

    Args:
        original_chunk: Original chunk element
        tagged_items: Dictionary mapping item IDs to their tagged XML content

    Returns:
        Updated chunk element with tagged items merged
    """
    # Clone the original chunk to avoid modifying it directly
    parser = etree.XMLParser(remove_blank_text=False, recover=True)
    updated_chunk = etree.fromstring(etree.tostring(original_chunk), parser)

    # Find all item elements in the updated chunk
    for item in updated_chunk.xpath(".//item"):
        item_id = item.get("id")
        if item_id and item_id in tagged_items:
            # Parse the tagged item
            try:
                parser = etree.XMLParser(remove_blank_text=False, recover=True)
                tagged_item = etree.fromstring(
                    tagged_items[item_id].encode("utf-8"), parser
                )

                # Replace the original item with the tagged one
                parent = item.getparent()
                if parent is not None:
                    item_idx = parent.index(item)
                    parent[item_idx] = tagged_item
            except etree.XMLSyntaxError as e:
                logger.error(f"Error parsing item {item_id} XML: {e}")
                logger.debug(f"Malformed XML: {tagged_items[item_id]}")
                # Keep the original item if parsing fails
            except Exception as e:
                logger.error(f"Error replacing item {item_id}: {e}")
                # Keep the original item if there's any other error

    return updated_chunk


def save_current_state(
    root: etree._Element,
    output_file: str,
    chunk_id: str,
    step_name: str = "",
    backup: bool = False,
) -> None:
    """
    Save current state to the output file after processing a chunk.

    Args:
        root: Current XML root element
        output_file: File to save output
        chunk_id: ID of the chunk that was processed
        step_name: Name of the processing step (optional)
        backup: Whether to create backup copies with timestamps
    """
    try:
        # Save current XML state
        with open(output_file, "wb") as f:
            xml_bytes = etree.tostring(
                root,
                encoding="utf-8",
                xml_declaration=True,
                pretty_print=False,
            )
            f.write(xml_bytes)

        # Create backup if requested
        if backup:
            output_path = Path(output_file)
            backup_dir = output_path.parent / f"{output_path.stem}_backups"
            backup_dir.mkdir(exist_ok=True)

            # Generate timestamp
            timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            backup_filename = f"{output_path.stem}-{timestamp}{output_path.suffix}"
            backup_path = backup_dir / backup_filename

            # Save backup
            with open(backup_path, "wb") as f:
                f.write(xml_bytes)

            logger.debug(f"Created backup at: {backup_path}")

        if step_name:
            logger.info(
                f"Saved current state after processing chunk {chunk_id} - {step_name} step"
            )
        else:
            logger.info(f"Saved current state after processing chunk {chunk_id}")

    except Exception as e:
        logger.error(f"Error saving current state: {e}")


@backoff.on_exception(backoff.expo, Exception, max_tries=3)
def run_llm_processing(
    chunk_text: str, prompt_template: str, model: str, temperature: float
) -> str:
    """
    Send text to LLM for processing using a specific prompt template.

    Args:
        chunk_text: XML text of the chunk
        prompt_template: Prompt template for the LLM
        model: LLM model identifier
        temperature: LLM temperature setting

    Returns:
        Processed text from the LLM
    """
    # Check if API key is available
    if not os.getenv("OPENROUTER_API_KEY"):
        logger.warning("No API key available for LLM requests")
        return chunk_text

    # Construct the full prompt with the text
    prompt = prompt_template + f"\n```xml\n{chunk_text}\n```"

    logger.info(f"Sending request to LLM model: {model}")
    logger.debug(f"Prompt: {prompt}")

    try:
        # Call the LLM and handle the response in a way that satisfies the type checker
        response = completion(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
        )

        # Use a safer approach with type checking
        try:
            # Cast response to Any to bypass type checking for attribute access
            response_any = cast(Any, response)

            # Check if we can safely access the content
            if (
                hasattr(response_any, "choices")
                and response_any.choices
                and hasattr(response_any.choices[0], "message")
                and hasattr(response_any.choices[0].message, "content")
            ):
                content = response_any.choices[0].message.content
                if content is not None:
                    return str(content)

            logger.warning("Unexpected response structure from LLM")
            return chunk_text
        except (AttributeError, IndexError):
            logger.warning("Unable to access expected attributes in LLM response")
            return chunk_text
    except Exception as e:
        logger.error(f"Error during LLM processing: {e}")

        # Attempt to use fallback model if different from current model
        if model != FALLBACK_MODEL:
            logger.warning(f"Attempting to use fallback model: {FALLBACK_MODEL}")
            try:
                response = completion(
                    model=FALLBACK_MODEL,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=temperature,
                )

                # Try to extract content from fallback response
                try:
                    response_any = cast(Any, response)
                    if (
                        hasattr(response_any, "choices")
                        and response_any.choices
                        and hasattr(response_any.choices[0], "message")
                        and hasattr(response_any.choices[0].message, "content")
                    ):
                        content = response_any.choices[0].message.content
                        if content is not None:
                            logger.info(
                                f"Successfully processed with fallback model: {FALLBACK_MODEL}"
                            )
                            return str(content)
                except (AttributeError, IndexError):
                    logger.warning(
                        "Unable to access expected attributes in fallback LLM response"
                    )
            except Exception as fallback_error:
                logger.error(f"Fallback model also failed: {fallback_error}")

        # If primary model and fallback both fail, or they're the same model, just return the original
        raise


def process_sentences(
    chunk: etree._Element,
    model: str,
    temperature: float,
    root: etree._Element | None = None,
    output_file: str | None = None,
    backup: bool = False,
) -> etree._Element:
    """
    Process a chunk to restructure sentences using LLM.

    Args:
        chunk: XML chunk element
        model: LLM model identifier
        temperature: LLM temperature setting
        root: Root element (optional, for saving intermediate results)
        output_file: Output file path (optional, for saving intermediate results)
        backup: Whether to create backup copies with timestamps

    Returns:
        Updated chunk with restructured sentences
    """
    chunk_id = chunk.get("id", "unknown_chunk")
    logger.info(f"Processing sentences for chunk {chunk_id}")

    # Get the chunk XML as text
    chunk_text = etree.tostring(chunk).decode("utf-8").strip()

    # Define the prompt template
    prompt_template = """
    Take the input text. Divide long sentences into shorter sentences (split at a comma or a semicolon), but DO NOT change any words or word order. Only insert periods (or exclamation marks) at logical places to break up very long sentences.

    Also divide long paragraphs into shorter paragraphs using two newlines.

    Each paragraph and heading must end with a punctuation mark: period or exclamation mark.

    The text must identical to the original in terms of words. Preserve all existing XML tags, punctuation, and structure.

    IMPORTANT: Return the complete new text.

    Input text:
    """

    # Process the chunk with LLM
    try:
        llm_response = run_llm_processing(
            chunk_text, prompt_template, model, temperature
        )

        # Extract item elements from the LLM response
        tagged_items = extract_item_elements(llm_response)

        # Merge the tagged items back into the original chunk
        if tagged_items:
            updated_chunk = merge_tagged_items(chunk, tagged_items)
            logger.info(f"Successfully processed sentences for chunk {chunk_id}")

            # Save intermediate results if root and output_file are provided
            if root is not None and output_file:
                # Replace the chunk in the root
                parent = chunk.getparent()
                if parent is not None:
                    chunk_idx = parent.index(chunk)
                    parent[chunk_idx] = updated_chunk

                # Save the current state
                save_current_state(root, output_file, chunk_id, "sentences", backup)

                # Return the updated chunk
                return updated_chunk
            else:
                return updated_chunk
        else:
            logger.warning(f"No valid items found in LLM response for chunk {chunk_id}")
            return chunk
    except Exception as e:
        logger.error(f"Error processing sentences: {e}")
        return chunk


def process_words(
    chunk: etree._Element,
    model: str,
    temperature: float,
    root: etree._Element | None = None,
    output_file: str | None = None,
    backup: bool = False,
) -> etree._Element:
    """
    Process a chunk to normalize words using LLM.

    Args:
        chunk: XML chunk element
        model: LLM model identifier
        temperature: LLM temperature setting
        root: Root element (optional, for saving intermediate results)
        output_file: Output file path (optional, for saving intermediate results)
        backup: Whether to create backup copies with timestamps

    Returns:
        Updated chunk with normalized words
    """
    chunk_id = chunk.get("id", "unknown_chunk")
    logger.info(f"Processing words for chunk {chunk_id}")

    # Get the chunk XML as text
    chunk_text = etree.tostring(chunk).decode("utf-8").strip()

    # Define the prompt template
    prompt_template = """
    We're converting written text into a script for spoken narrative. Process the text according to these rules:

    1. Convert digits to numeric words. Do it in a context-aware and language-specific way, e.g., '42' → 'forty-two' if the text is in English, '20 maja 2004, 22.11' → 'dwudziestego maja dwa tysiące czwartego roku, dwudziesta druga jedenaście'.

    2. Replace '/' and '&' with appropriate conjunctions, in a language-specific way, e.g., if the text is in English: 'up/down' → 'up or down', 'X & Y' → 'X and Y').

    3. Similarly resolve other rare symbols in the text, spelling them out. But don't change the text if it's already in a readable form.

    Make these changes without altering the content beyond the specified transformations. Preserve all existing XML tags and structure.

    IMPORTANT: Return the complete new text.

    Input text:
    """

    # Process the chunk with LLM
    try:
        llm_response = run_llm_processing(
            chunk_text, prompt_template, model, temperature
        )

        # Extract item elements from the LLM response
        tagged_items = extract_item_elements(llm_response)

        # Merge the tagged items back into the original chunk
        if tagged_items:
            updated_chunk = merge_tagged_items(chunk, tagged_items)
            logger.info(f"Successfully processed words for chunk {chunk_id}")

            # Save intermediate results if root and output_file are provided
            if root is not None and output_file:
                # Replace the chunk in the root
                parent = chunk.getparent()
                if parent is not None:
                    chunk_idx = parent.index(chunk)
                    parent[chunk_idx] = updated_chunk

                # Save the current state
                save_current_state(root, output_file, chunk_id, "words", backup)

                # Return the updated chunk
                return updated_chunk
            else:
                return updated_chunk
        else:
            logger.info(f"No items needed word processing for chunk {chunk_id}")
            return chunk
    except Exception as e:
        logger.error(f"Error processing words: {e}")
        return chunk


def process_punctuation(
    chunk: etree._Element,
    root: etree._Element | None = None,
    output_file: str | None = None,
    backup: bool = False,
) -> etree._Element:
    """
    Process a chunk to enhance punctuation algorithmically (not using LLM).

    Args:
        chunk: XML chunk element
        root: Root element (optional, for saving intermediate results)
        output_file: Output file path (optional, for saving intermediate results)
        backup: Whether to create backup copies with timestamps

    Returns:
        Updated chunk with enhanced punctuation
    """
    chunk_id = chunk.get("id", "unknown_chunk")
    logger.info(f"Processing punctuation for chunk {chunk_id}")

    # Clone the chunk to avoid modifying it directly
    parser = etree.XMLParser(remove_blank_text=False, recover=True)
    updated_chunk = etree.fromstring(etree.tostring(chunk), parser)

    # Find all item elements in the updated chunk
    for item in updated_chunk.xpath(".//item"):
        if item.text:
            # Process parenthetical expressions - surround with spaces and ellipses
            item.text = re.sub(r"\(([^)]+)\)", r" ... (\1) ... ", item.text)

            # Process em dashes, en dashes and hyphens when used as pauses (with spaces around them)
            # Only target dashes that have spaces before and after them, preserving hyphenated words
            item.text = re.sub(r"(?<=\s)[—–-](?=\s)", r"...", item.text)

    logger.info(f"Successfully processed punctuation for chunk {chunk_id}")

    # Save intermediate results if root and output_file are provided
    if root is not None and output_file:
        # Replace the chunk in the root
        parent = chunk.getparent()
        if parent is not None:
            chunk_idx = parent.index(chunk)
            parent[chunk_idx] = updated_chunk

        # Save the current state
        save_current_state(root, output_file, chunk_id, "punctuation", backup)

    return updated_chunk


def process_emotions(
    chunk: etree._Element,
    model: str,
    temperature: float,
    root: etree._Element | None = None,
    output_file: str | None = None,
    backup: bool = False,
) -> etree._Element:
    """
    Process a chunk to add emotional emphasis using LLM.

    Args:
        chunk: XML chunk element
        model: LLM model identifier
        temperature: LLM temperature setting
        root: Root element (optional, for saving intermediate results)
        output_file: Output file path (optional, for saving intermediate results)
        backup: Whether to create backup copies with timestamps

    Returns:
        Updated chunk with emotional emphasis
    """
    chunk_id = chunk.get("id", "unknown_chunk")
    logger.info(f"Processing emotions for chunk {chunk_id}")

    # Get the chunk XML as text
    chunk_text = etree.tostring(chunk).decode("utf-8").strip()

    # Define the prompt template
    prompt_template = """
    Take the text and add emotional cues. In each paragraph, select at least one word or phrase and surround it with <em> tags, like an actor emphasizing that part of speech. Sparingly insert <hr/> tags to add dramatic pauses where appropriate. Add exclamation marks where you need additional energy or drama.

    However: avoid excessive emphasis, quotation marks or ellipses. The text must remain fluid while gaining expressiveness.

    DO NOT REMOVE ANY WORDS! Don't shorten or paraphrase the text. Only enhance its delivery with punctuation and accents, considering its original tone.

    Preserve all existing XML tags and structure.

    IMPORTANT: Return the complete new text.

    Input text:
    """

    # Process the chunk with LLM
    try:
        llm_response = run_llm_processing(
            chunk_text, prompt_template, model, temperature
        )

        # Extract item elements from the LLM response
        tagged_items = extract_item_elements(llm_response)

        # Merge the tagged items back into the original chunk
        if tagged_items:
            updated_chunk = merge_tagged_items(chunk, tagged_items)
            logger.info(f"Successfully processed emotions for chunk {chunk_id}")

            # Save intermediate results if root and output_file are provided
            if root is not None and output_file:
                # Replace the chunk in the root
                parent = chunk.getparent()
                if parent is not None:
                    chunk_idx = parent.index(chunk)
                    parent[chunk_idx] = updated_chunk

                # Save the current state
                save_current_state(root, output_file, chunk_id, "emotions", backup)

                # Return the updated chunk
                return updated_chunk
            else:
                return updated_chunk
        else:
            logger.info(f"No items needed emotional processing for chunk {chunk_id}")
            return chunk
    except Exception as e:
        logger.error(f"Error processing emotions: {e}")
        return chunk


def process_document(
    input_file: str,
    output_file: str,
    model: str = DEFAULT_MODEL,
    temperature: float = DEFAULT_TEMPERATURE,
    sentences: bool = False,
    words: bool = False,
    punctuation: bool = False,
    emotions: bool = False,
    all_steps: bool = False,
    verbose: bool = False,
    backup: bool = False,
) -> None:
    """
    Process the document through the specified enhancement steps.

    Args:
        input_file: Path to the input XML file
        output_file: Path to the output XML file
        model: LLM model identifier
        temperature: LLM temperature setting
        sentences: Whether to apply sentence restructuring
        words: Whether to apply word normalization
        punctuation: Whether to apply punctuation enhancement
        emotions: Whether to apply emotional emphasis
        all_steps: Whether to apply all enhancement steps
        verbose: Whether to enable verbose logging
        backup: Whether to create backup copies with timestamps
    """
    # Set up logging level
    if verbose:
        logger.remove()
        logger.add(lambda msg: print(msg, end=""), level="DEBUG")
    else:
        logger.remove()
        logger.add(lambda msg: print(msg, end=""), level="INFO")

    # Apply all steps if all_steps is True
    if all_steps:
        sentences = words = punctuation = emotions = True

    # Check if at least one step is specified
    if not any([sentences, words, punctuation, emotions]):
        logger.warning(
            "No processing steps specified. Use --sentences, --words, --punctuation, --emotions, or --all."
        )
        return

    try:
        # Parse input XML
        logger.info(f"Parsing input file: {input_file}")
        root = parse_xml(input_file)

        # Get chunks
        chunks = get_chunks(root)
        logger.info(f"Found {len(chunks)} chunks in the document")

        # Process each chunk
        for i, chunk in enumerate(chunks):
            chunk_id = chunk.get("id", f"chunk_{i}")
            logger.info(f"Processing chunk {chunk_id} ({i + 1}/{len(chunks)})")

            # Get the current chunk from the document (it may have been updated)
            current_chunks = get_chunks(root)
            current_chunk = current_chunks[i]

            # Apply each selected enhancement step
            if sentences:
                logger.info("Applying sentence restructuring")
                current_chunk = process_sentences(
                    current_chunk, model, temperature, root, output_file, backup
                )

            if words:
                logger.info("Applying word normalization")
                current_chunk = process_words(
                    current_chunk, model, temperature, root, output_file, backup
                )

            if punctuation:
                logger.info("Applying punctuation enhancement")
                current_chunk = process_punctuation(
                    current_chunk, root, output_file, backup
                )

            if emotions:
                logger.info("Applying emotional emphasis")
                current_chunk = process_emotions(
                    current_chunk, model, temperature, root, output_file, backup
                )

            # Final save for the chunk
            save_current_state(
                root, output_file, chunk_id, "all steps complete", backup
            )

        # Final save
        save_current_state(root, output_file, "final", "processing complete", backup)
        logger.info(f"Document processing completed. Output saved to {output_file}")

    except Exception as e:
        logger.error(f"Error processing document: {e}")
        raise


def main(
    input_file: str,
    output_file: str,
    model: str = DEFAULT_MODEL,
    temperature: float = DEFAULT_TEMPERATURE,
    sentences: bool = False,
    words: bool = False,
    punctuation: bool = False,
    emotions: bool = False,
    all_steps: bool = False,
    verbose: bool = True,
    backup: bool = True,
) -> None:
    """
    Main entry point for the malmo_orator tool.

    Args:
        input_file: Path to the input XML file
        output_file: Path to the output XML file
        model: LLM model identifier or key from MODEL_DICT
        temperature: LLM temperature setting
        sentences: Whether to apply sentence restructuring
        words: Whether to apply word normalization
        punctuation: Whether to apply punctuation enhancement
        emotions: Whether to apply emotional emphasis
        all_steps: Whether to apply all enhancement steps
        verbose: Whether to enable verbose logging
        backup: Whether to create backup copies with timestamps
    """
    # Resolve model name if it's a key in MODEL_DICT
    if model in MODEL_DICT:
        model = MODEL_DICT[model]
    if verbose:
        logger.level("DEBUG")
        _turn_on_debug()

    # Process the document
    try:
        process_document(
            input_file=input_file,
            output_file=output_file,
            model=model,
            temperature=temperature,
            sentences=sentences,
            words=words,
            punctuation=punctuation,
            emotions=emotions,
            all_steps=all_steps,
            verbose=verbose,
            backup=backup,
        )
    except Exception as e:
        logger.error(f"Error in main function: {e}")
        raise


if __name__ == "__main__":
    fire.Fire(main)
