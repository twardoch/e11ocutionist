#!/usr/bin/env -S uv run -s
# /// script
# dependencies = ["fire", "litellm<=1.67.2", "lxml", "backoff", "python-dotenv", "loguru"]
# ///
# this_file: malmo_entitizer.py

import fire
import re
import os
import json
import backoff
import concurrent.futures
from datetime import datetime
from pathlib import Path
from lxml import etree
from litellm import completion
from litellm._logging import _turn_on_debug
from dotenv import load_dotenv
from loguru import logger

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


def get_chunk_text(chunk: etree._Element) -> str:
    """
    Extract the raw text content of a chunk while preserving whitespace.

    Args:
        chunk: XML chunk element

    Returns:
        Raw text content of the chunk with all whitespace preserved
    """
    # Using etree.tostring to preserve all whitespace in the XML structure
    chunk_xml = etree.tostring(chunk, encoding="utf-8", method="xml").decode("utf-8")
    return chunk_xml


def save_current_state(
    root: etree._Element,
    nei_dict: dict[str, str],
    output_file: str,
    chunk_id: str,
    backup: bool = False,
) -> None:
    """
    Save current state to the output file after processing a chunk.

    Args:
        root: Current XML root element
        nei_dict: Current NEI dictionary
        output_file: File to save output
        chunk_id: ID of the chunk that was processed
        backup: Whether to create backup copies with timestamps
    """
    try:
        # Save current XML state
        with open(output_file, "wb") as f:
            xml_bytes = etree.tostring(
                root,
                encoding="utf-8",
                method="xml",
                pretty_print=False,
                xml_declaration=True,
            )
            f.write(xml_bytes)

        # Save current NEI dictionary to a JSON file next to the output file
        nei_path = f"{os.path.splitext(output_file)[0]}_nei_dict.json"
        with open(nei_path, "w", encoding="utf-8") as f:
            json.dump(nei_dict, f, ensure_ascii=False, indent=2)

        # Create backup if requested
        if backup:
            output_path = Path(output_file)
            backup_dir = output_path.parent / f"{output_path.stem}_backups"
            backup_dir.mkdir(exist_ok=True)

            # Generate timestamp
            timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")

            # Backup XML file
            backup_filename = f"{output_path.stem}-{timestamp}{output_path.suffix}"
            backup_path = backup_dir / backup_filename
            with open(backup_path, "wb") as f:
                f.write(xml_bytes)

            # Backup NEI dictionary
            nei_backup_filename = f"{output_path.stem}-{timestamp}_nei_dict.json"
            nei_backup_path = backup_dir / nei_backup_filename
            with open(nei_backup_path, "w", encoding="utf-8") as f:
                json.dump(nei_dict, f, ensure_ascii=False, indent=2)

            logger.debug(f"Created backups at: {backup_path} and {nei_backup_path}")

        logger.info(f"Saved current state after processing chunk {chunk_id}")
        logger.debug(f"Files saved to {output_file} and {nei_path}")

    except Exception as e:
        logger.error(f"Error saving current state: {e}")


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
            except Exception as e:
                logger.error(f"Error replacing item {item_id}: {e}")
                # If parsing fails, keep the original item

    return updated_chunk


@backoff.on_exception(backoff.expo, Exception, max_tries=3)
def identify_entities(
    chunk_text: str, nei_list: dict[str, str], model: str, temperature: float
) -> str:
    """
    Use LLM to identify and tag named entities of interest (NEIs) in the chunk.

    Args:
        chunk_text: Raw text of the chunk
        nei_list: Dictionary mapping NEIs to their pronunciations
        model: LLM model identifier
        temperature: LLM temperature setting

    Returns:
        Chunk text with NEIs tagged
    """
    # Check if API key is available
    if not os.getenv("OPENROUTER_API_KEY"):
        logger.warning("No API key available for LLM requests")
        return chunk_text

    # Construct formatted NEI list for the prompt
    nei_list_str = (
        ", ".join(
            [
                f'"{nei}" (pronunciation: "{pronunciation}")'
                for nei, pronunciation in nei_list.items()
            ]
        )
        if nei_list
        else "none yet"
    )

    # Log the current state of the NEI list before LLM call
    logger.info(f"Current NEI dictionary size before LLM call: {len(nei_list)} entries")
    if nei_list:
        recent_neis = list(nei_list.items())[-min(5, len(nei_list)) :]
        logger.info(
            f"Recent NEIs: {', '.join([f'{nei} ({pron})' for nei, pron in recent_neis])}"
        )

    prompt = f"""
You are a "named entity of interest" recognizer for a document processing system. You need to identify and tag "named entities of interest" (NEIs) in an XML document.

RULES:

1. First, identify the predominant language of the text.

2. Analyze the domain of the text to determine what KINDS of named entities are "of interest" to the domain. Typically your NEIs will be people, locations, organizations, abbreviations, but for example in a biological text, the NEIs will be species, genes, proteins, etc.

3. Go item by item through the text. If you come across a NEI:

a. Mark EACH occurrence of the NEI (not just the first one) with a <nei> XML tag.
b. If the NEI only consists of common words from the predominant language, just surround the NEI with the <nei> tag.
c. Otherwise use the notation `<nei orig="ORIGINAL">PRONUNCIATION</nei>`, where `ORIGINAL` is the actual original spelling of the NEI as it appears in the text, and `PRONUNCIATION` is the pronunciation of the NEI approximated into the predominant language.
d. Build the approximated pronunciation as follows: Use the orthography of the predominant language. If the NEI has letter combinations that are pronounced differently in the predominant language, split syllables with a hyphen.
e. Write out abbreviations. Spell out acronyms.
f. If the predominant language uses "i" to indicate palatalization (like Polish), transcribe words like English "sea" as "sji" and not "si".

> For example, if the NEI is "José Scaglione" in a Polish text, the tag would be <nei orig="José Scaglione">Hose Skaljone</nei>
> If the NEI is "New York Knicks" in a Polish text, the tag would be <nei orig="New York Knicks">Nju Jork Niks</nei>
> If the NEI is "CIA" in a Polish text, the tag would be <nei orig="ONZ">Sji-Aj-Ej</nei>
> If the NEI is "Honolulu" in a Polish text, the tag would be <nei orig="Honolulu">Honolulu</nei> — because that’s an established name in Polish.
> If the NEI is being mentioned for the first time (not in the NEI list), add the attribute new="true": <nei orig="NEI" new="true">PRONUNCIATION</nei>

4. Preserve ALL whitespace, line breaks, pre-existing tags, and formatting EXACTLY as in the original item!

5. Only print the items that contain a NEI that you've marked with an <nei> tag.

Here is the dictionary of previously found NEIs with their pronunciations. If you come across any of them in the text, transcribe their pronunciation in a consistent way. And remember: if a NEI is not in this dictionary, add the `new="true"` attribute to the <nei> tag!

{nei_list_str}

Here is the XML chunk to process:

{chunk_text}
"""

    try:
        logger.info(f"Sending request to LLM model: {model}")
        response = completion(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
        )

        # Extract the content from the response
        result_text = extract_response_text(response)
        logger.debug(f"LLM response length: {len(result_text)}")

        return result_text
    except Exception as e:
        logger.error(f"Error in entity identification: {e}")

        # Try fallback model if different from current model
        if model != FALLBACK_MODEL:
            try:
                logger.warning(f"Attempting to use fallback model: {FALLBACK_MODEL}")
                fallback_response = completion(
                    model=FALLBACK_MODEL,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=temperature,
                )

                # Extract content from fallback response
                fallback_result = extract_response_text(fallback_response)

                if fallback_result:
                    logger.info(
                        f"Successfully processed with fallback model: {FALLBACK_MODEL}"
                    )
                    return fallback_result
            except Exception as fallback_error:
                logger.error(f"Fallback model also failed: {fallback_error}")

        # Return the original text if both primary and fallback fail
        return chunk_text


def extract_response_text(response) -> str:
    """
    Extract text content from an LLM response safely handling different response structures.

    Args:
        response: The LLM response object

    Returns:
        The extracted text content as a string
    """
    result_text = ""
    try:
        if hasattr(response, "choices") and response.choices:
            try:
                result_text = response.choices[0].message.content
            except (AttributeError, IndexError):
                try:
                    choice = response.choices[0]
                    if hasattr(choice, "text"):
                        result_text = choice.text
                    else:
                        result_text = str(choice)
                except (AttributeError, IndexError):
                    result_text = str(response.choices[0])
        elif hasattr(response, "content") and response.content:
            result_text = response.content
        else:
            result_text = str(response)

        # Ensure we have a string, not None
        return result_text or ""
    except Exception:
        # Fallback to string representation if all else fails
        return str(response)


def extract_nei_from_tag(text: str) -> dict[str, str]:
    """
    Extract NEIs and their pronunciations from nei tags in the text.

    Args:
        text: Text with nei tags

    Returns:
        Dictionary mapping NEIs to their pronunciations
    """
    nei_dict = {}
    pattern = r'<nei orig="([^"]+)"(?:\s+new="true")?>(.*?)</nei>'
    matches = re.finditer(pattern, text)

    for match in matches:
        nei = match.group(1).strip()
        pronunciation = match.group(2).strip()
        if nei and pronunciation:
            nei_dict[nei] = pronunciation

    return nei_dict


def process_chunks(
    chunks: list[etree._Element],
    model: str,
    temperature: float,
    output_file: str = "",
    backup: bool = False,
) -> tuple[etree._Element, dict[str, str]]:
    """
    Process chunks to identify and tag NEIs.

    Args:
        chunks: List of XML chunk elements
        model: LLM model identifier
        temperature: LLM temperature setting
        output_file: File to save intermediate outputs
        backup: Whether to create backup copies with timestamps

    Returns:
        Tuple of (root element with tagged NEIs, dictionary of all NEIs found with their pronunciations)
    """
    nei_list: dict[str, str] = {}

    # Log whether we'll be saving intermediate results
    if output_file:
        logger.info(f"Will save progress after each chunk to {output_file}")

    for i, chunk in enumerate(chunks):
        chunk_id = chunk.get("id", f"unknown_{i}")
        logger.info(f"Processing chunk {chunk_id} ({i + 1}/{len(chunks)})")
        logger.info(f"Current NEI dictionary has {len(nei_list)} entries")

        # Get the chunk text
        chunk_text = get_chunk_text(chunk)

        # Identify NEIs in the chunk
        tagged_chunk_text = identify_entities(chunk_text, nei_list, model, temperature)

        # Extract newly found NEIs to add to the nei list
        new_neis = extract_nei_from_tag(tagged_chunk_text)

        # Log the new NEIs found
        if new_neis:
            logger.info(f"Found {len(new_neis)} new NEIs in chunk {chunk_id}")
            for nei, pronunciation in new_neis.items():
                logger.info(f"  - NEI: '{nei}', Pronunciation: '{pronunciation}'")

            # Update the overall NEI list
            original_size = len(nei_list)
            nei_list.update(new_neis)
            logger.info(
                f"NEI dictionary updated: {original_size} → {len(nei_list)} entries"
            )

            # Extract tagged items from the LLM response
            tagged_items = extract_item_elements(tagged_chunk_text)

            if tagged_items:
                logger.debug(f"Found {len(tagged_items)} tagged items")

                # Merge tagged items into the original chunk
                updated_chunk = merge_tagged_items(chunk, tagged_items)

                # Replace the original chunk with the updated one
                parent = chunk.getparent()
                if parent is not None:
                    chunk_idx = parent.index(chunk)
                    parent[chunk_idx] = updated_chunk
        else:
            logger.info(f"No new NEIs found in chunk {chunk_id}")

        # Save current state after processing this chunk
        if output_file:
            save_current_state(
                chunks[0].getroottree().getroot(),
                nei_list,
                output_file,
                chunk_id,
                backup,
            )

    # Log the final NEI dictionary statistics
    logger.info(f"Final NEI dictionary contains {len(nei_list)} entries")

    return chunks[0].getroottree().getroot(), nei_list


def process_document(
    input_file: str,
    output_file: str,
    model: str = DEFAULT_MODEL,
    temperature: float = DEFAULT_TEMPERATURE,
    verbose: bool = False,
    save_incremental: bool = True,
    backup: bool = False,
) -> None:
    """
    Process an XML document to identify and tag named entities of interest.

    Args:
        input_file: Path to the input XML file
        output_file: Path for the XML output
        model: LLM model identifier
        temperature: LLM temperature setting
        verbose: Enable detailed logging
        save_incremental: Save incremental progress after each chunk
        backup: Whether to create backup copies with timestamps
    """
    if verbose:
        logger.level("DEBUG")

    try:
        # Parse input XML
        logger.info(f"Reading input file: {input_file}")
        root = parse_xml(input_file)

        # Get chunks
        chunks = get_chunks(root)
        logger.info(f"Found {len(chunks)} chunks")

        # Process chunks
        logger.info(f"Processing chunks with model: {model}")

        # If save_incremental is true, pass the output file to process_chunks
        chunk_output_file = output_file if save_incremental else ""
        root, nei_dict = process_chunks(
            chunks, model, temperature, chunk_output_file, backup
        )

        # Save final NEI dictionary
        nei_path = f"{os.path.splitext(output_file)[0]}_nei_dict.json"
        with open(nei_path, "w", encoding="utf-8") as f:
            json.dump(nei_dict, f, ensure_ascii=False, indent=2)
        logger.info(f"Final NEI dictionary saved to {nei_path}")

        # Create final backup if requested
        if backup:
            output_path = Path(output_file)
            backup_dir = output_path.parent / f"{output_path.stem}_backups"
            backup_dir.mkdir(exist_ok=True)

            # Generate timestamp
            timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")

            # Backup XML file
            backup_filename = (
                f"{output_path.stem}-{timestamp}-final{output_path.suffix}"
            )
            backup_path = backup_dir / backup_filename

            # Use tostring with xml_declaration to get a proper XML document
            xml_bytes = etree.tostring(
                root,
                encoding="utf-8",
                method="xml",
                pretty_print=False,  # Preserve exact whitespace
                xml_declaration=True,
            )

            with open(backup_path, "wb") as f:
                f.write(xml_bytes)

            # Backup NEI dictionary
            nei_backup_filename = f"{output_path.stem}-{timestamp}-final_nei_dict.json"
            nei_backup_path = backup_dir / nei_backup_filename
            with open(nei_backup_path, "w", encoding="utf-8") as f:
                json.dump(nei_dict, f, ensure_ascii=False, indent=2)

            logger.debug(
                f"Created final backups at: {backup_path} and {nei_backup_path}"
            )

        # Write output file
        logger.info(f"Writing final output to: {output_file}")
        with open(output_file, "wb") as f:
            # Use tostring with xml_declaration to get a proper XML document
            xml_bytes = etree.tostring(
                root,
                encoding="utf-8",
                method="xml",
                pretty_print=False,  # Preserve exact whitespace
                xml_declaration=True,
            )
            f.write(xml_bytes)

        logger.success(f"Processing complete! Tagged {len(nei_dict)} NEIs.")

    except Exception as e:
        logger.error(f"Error processing document: {e}")
        raise


def process_document_with_model(
    input_file: str,
    output_file: str,
    model: str,
    model_key: str = "",
    temperature: float = DEFAULT_TEMPERATURE,
    verbose: bool = False,
    save_incremental: bool = True,
    max_retries: int = 3,
    backup: bool = False,
) -> bool:
    """
    Process an XML document with a specific model, with retry logic for failures.

    Args:
        input_file: Path to the input XML file
        output_file: Path for the XML output
        model: LLM model identifier
        model_key: Key name for the model (used in logging)
        temperature: LLM temperature setting
        verbose: Enable detailed logging
        save_incremental: Save incremental progress after each chunk
        max_retries: Maximum number of retries before giving up
        backup: Whether to create backup copies with timestamps

    Returns:
        bool: True if processing succeeded, False if it failed after retries
    """
    model_display = f"{model_key} ({model})" if model_key else model

    # Track partial output files to clean up on failure
    partial_files = []
    output_path = Path(output_file)
    nei_path = output_path.with_name(f"{output_path.stem}_nei_dict.json")

    for attempt in range(1, max_retries + 1):
        try:
            logger.info(
                f"Processing with model {model_display}, attempt {attempt}/{max_retries}"
            )

            # Process document with the specified model
            process_document(
                input_file,
                output_file,
                model,
                temperature,
                verbose,
                save_incremental,
                backup,
            )

            logger.success(f"Successfully processed with model {model_display}")
            return True

        except Exception as e:
            # Clean up partial output files if they exist
            if os.path.exists(output_file):
                partial_files.append(output_file)

            if os.path.exists(nei_path):
                partial_files.append(str(nei_path))

            # Also check for and add backup directory to cleanup list if it exists
            backup_dir = output_path.parent / f"{output_path.stem}_backups"
            if backup and backup_dir.exists():
                partial_files.append(str(backup_dir))

            if attempt < max_retries:
                logger.warning(
                    f"Attempt {attempt} failed with model {model_display}: {e}. Retrying..."
                )
            else:
                logger.error(
                    f"All {max_retries} attempts failed with model {model_display}. Skipping this model."
                )

                # Clean up any partial output files on final failure
                for file_path in set(partial_files):
                    try:
                        if os.path.exists(file_path):
                            if os.path.isdir(file_path):
                                import shutil

                                shutil.rmtree(file_path)
                                logger.debug(
                                    f"Cleaned up backup directory: {file_path}"
                                )
                            else:
                                os.remove(file_path)
                                logger.debug(
                                    f"Cleaned up partial output file: {file_path}"
                                )
                    except Exception as cleanup_err:
                        logger.warning(
                            f"Failed to clean up file/directory {file_path}: {cleanup_err}"
                        )

                return False


def process_document_with_all_models(
    input_file: str,
    output_folder: str,
    temperature: float = DEFAULT_TEMPERATURE,
    verbose: bool = False,
    save_incremental: bool = True,
    max_workers: int = 4,
    backup: bool = False,
) -> None:
    """
    Process an XML document with all models in parallel.

    Args:
        input_file: Path to the input XML file
        output_folder: Folder to save output files
        temperature: LLM temperature setting
        verbose: Enable detailed logging
        save_incremental: Save incremental progress after each chunk
        max_workers: Maximum number of worker threads
        backup: Whether to create backup copies with timestamps
    """
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    logger.info(f"Created output folder: {output_folder}")

    # Get input file basename without extension
    input_basename = os.path.basename(input_file)
    input_name_without_ext = os.path.splitext(input_basename)[0]
    logger.info(f"Processing input file: {input_file}")

    # Determine optimal number of workers based on available models
    model_count = len(MODEL_DICT)
    effective_workers = min(max_workers, model_count)
    logger.info(f"Using {effective_workers} workers to process {model_count} models")

    # Start time for overall processing
    start_time = datetime.now()

    # Set up threading pool for parallel processing
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=effective_workers
    ) as executor:
        # Start a processing task for each model
        future_to_model = {}

        logger.info(f"Launching processing tasks for {model_count} models")
        for model_key, model_id in MODEL_DICT.items():
            # Generate output filename: basename_modelkey.xml
            output_file = os.path.join(
                output_folder, f"{input_name_without_ext}_{model_key}.xml"
            )
            logger.debug(f"Queuing model {model_key} → {output_file}")

            # Submit the processing task to the executor
            future = executor.submit(
                process_document_with_model,
                input_file=input_file,
                output_file=output_file,
                model=model_id,
                model_key=model_key,
                temperature=temperature,
                verbose=verbose,
                save_incremental=save_incremental,
                backup=backup,
            )

            future_to_model[future] = (model_key, model_id)

        # Process results as they complete
        completed = 0
        successful_models = []
        failed_models = []

        logger.info(f"Waiting for {model_count} processing tasks to complete...")
        for future in concurrent.futures.as_completed(future_to_model):
            model_key, model_id = future_to_model[future]
            completed += 1

            try:
                success = future.result()
                if success:
                    successful_models.append(model_key)
                    output_file = os.path.join(
                        output_folder, f"{input_name_without_ext}_{model_key}.xml"
                    )
                    logger.info(
                        f"[{completed}/{model_count}] ✅ Model {model_key} completed successfully → {output_file}"
                    )
                else:
                    failed_models.append(model_key)
                    logger.warning(
                        f"[{completed}/{model_count}] ❌ Model {model_key} failed after multiple retries"
                    )
            except Exception as e:
                failed_models.append(model_key)
                logger.error(
                    f"[{completed}/{model_count}] ❌ Exception processing with model {model_key}: {e}"
                )

        # Calculate processing time
        elapsed_time = datetime.now() - start_time

        # Log summary
        success_count = len(successful_models)
        failed_count = len(failed_models)

        logger.info(
            f"Processing complete in {elapsed_time.total_seconds():.2f} seconds"
        )
        logger.info(
            f"Successful models ({success_count}/{model_count}): {', '.join(successful_models) if successful_models else 'None'}"
        )

        if failed_count > 0:
            logger.warning(
                f"Failed models ({failed_count}/{model_count}): {', '.join(failed_models)}"
            )

        # Generate a summary file
        summary_file = os.path.join(
            output_folder, f"{input_name_without_ext}_summary.json"
        )
        try:
            summary = {
                "input_file": input_file,
                "timestamp": datetime.now().isoformat(),
                "processing_time_seconds": elapsed_time.total_seconds(),
                "total_models": model_count,
                "successful_models": successful_models,
                "failed_models": failed_models,
                "models": dict(MODEL_DICT.items()),
                "backup_enabled": backup,
            }

            with open(summary_file, "w", encoding="utf-8") as f:
                json.dump(summary, f, ensure_ascii=False, indent=2)

            logger.info(f"Summary saved to {summary_file}")
        except Exception as e:
            logger.error(f"Error saving summary: {e}")


def main(
    input_file: str,
    output_file: str,
    model: str = DEFAULT_MODEL,
    temperature: float = DEFAULT_TEMPERATURE,
    verbose: bool = True,
    save_incremental: bool = True,
    benchmark: bool = False,
    max_workers: int = 4,
    backup: bool = True,
) -> None:
    """
    Main entry point for malmo_entitizer.

    Args:
        input_file: Path to the input XML file
        output_file: Path for the XML output file or folder (if --benchmark is used)
        model: LLM model identifier (ignored if --benchmark is used)
        temperature: LLM temperature setting
        verbose: Enable detailed logging
        save_incremental: Save incremental progress after each chunk
        benchmark: Process with all models in parallel
        max_workers: Maximum number of parallel workers when using --benchmark
        backup: Whether to create backup copies with timestamps
    """
    if verbose:
        logger.level("DEBUG")
        _turn_on_debug()

    if benchmark:
        logger.info(
            f"--benchmark flag specified, will process with all {len(MODEL_DICT)} models in parallel"
        )
        logger.info(f"Treating '{output_file}' as output folder")
        process_document_with_all_models(
            input_file=input_file,
            output_folder=output_file,
            temperature=temperature,
            verbose=verbose,
            save_incremental=save_incremental,
            max_workers=max_workers,
            backup=backup,
        )
    else:
        process_document(
            input_file=input_file,
            output_file=output_file,
            model=model,
            temperature=temperature,
            verbose=verbose,
            save_incremental=save_incremental,
            backup=backup,
        )


if __name__ == "__main__":
    fire.Fire(main)
