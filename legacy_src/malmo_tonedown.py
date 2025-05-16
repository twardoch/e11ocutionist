#!/usr/bin/env -S uv run -s
# /// script
# dependencies = ["fire", "lxml", "loguru", "backoff", "python-dotenv", "litellm<=1.67.2", "rich", "types-lxml"]
# ///
# this_file: malmo_tonedown.py

import fire
import re
import os
import json
import backoff
from lxml import etree
from loguru import logger
from rich.console import Console
from typing import Any, cast
from dotenv import load_dotenv
from litellm import completion
from litellm._logging import _turn_on_debug

# Load environment variables from .env file
load_dotenv()

console = Console()

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
    "claudesonnet": "openrouter/anthropic/claude-3.7-sonnet",
}

# Check if API key is set
if not os.getenv("OPENROUTER_API_KEY"):
    logger.warning("No API key found in environment variables. API calls may fail.")


# --- XML Parsing Utilities ---
def parse_xml(input_file: str) -> etree._Element:
    """
    Parse the input XML file.

    Args:
        input_file: Path to the input XML file

    Returns:
        Parsed XML element tree
    """
    try:
        # Use a more robust parser configuration
        parser = etree.XMLParser(
            remove_blank_text=False,
            recover=True,
            resolve_entities=False,
            huge_tree=True,
            remove_comments=False,  # Preserve comments
            remove_pis=False,  # Preserve processing instructions
        )
        with open(input_file, encoding="utf-8") as f:
            xml_content = f.read()

        # Parse the content
        root = etree.fromstring(xml_content.encode("utf-8"), parser)

        # Log some basic info about the parsed structure
        logger.debug(
            f"XML parsed successfully: {len(root.xpath('//*'))} elements total"
        )
        logger.debug(f"Found {len(root.xpath('//nei'))} nei elements")

        return root
    except Exception as e:
        logger.error(f"Error parsing XML: {e}")
        raise


# --- NEI Dict Extraction and Tag Processing ---
def process_nei_tags(root: etree._Element) -> dict[str, str]:
    """
    Build the NEIs dict and update <nei> tags in-place according to the rules.

    Args:
        root: XML root element

    Returns:
        neis: dict mapping orig attribute to pronunciation cue
    """
    neis = {}
    for nei in root.xpath("//nei"):
        # Skip processing if element is None or doesn't have attributes
        if nei is None or not hasattr(nei, "attrib"):
            continue

        orig = nei.get("orig")
        if not orig:
            continue

        # Get the text content safely by joining all text content, ignoring nested tags
        content_parts = []

        # First add the direct text of the nei element
        if nei.text:
            content_parts.append(nei.text)

        # Then add text from any children, recursively
        for elem in nei.iter():
            if elem != nei and elem.text:
                content_parts.append(elem.text)

        # Join all text parts and normalize whitespace
        content = " ".join(content_parts).strip()
        content = re.sub(r"\s+", " ", content)

        if not content:
            # Fallback to just getting all text
            content = "".join(nei.itertext()).strip()
            content = re.sub(r"\s+", " ", content)

        if orig not in neis:
            neis[orig] = content
            # Set new="true" if not already present with that value
            if nei.get("new") != "true":
                nei.set("new", "true")
        elif "new" in nei.attrib:
            del nei.attrib["new"]

    return neis


# --- Language Detection with LLM ---
def extract_middle_fragment(xml_text: str, length: int = 1024) -> str:
    """
    Extract a fragment of the given length from the middle of the XML text.
    Handles XML cleaning in a safe way.

    Args:
        xml_text: Full XML text
        length: Length of fragment to extract

    Returns:
        Text fragment from the middle of the document
    """
    if not xml_text:
        return ""

    try:
        # Clean XML tags to get plain text for better language detection
        # Uses a more resilient approach to handle potentially malformed XML
        plain_text = re.sub(r"<[^>]*>", " ", xml_text)
        plain_text = re.sub(r"\s+", " ", plain_text).strip()

        n = len(plain_text)
        if n <= length:
            return plain_text
        start = (n - length) // 2
        return plain_text[start : start + length]
    except Exception as e:
        logger.warning(f"Error extracting middle fragment: {e}")
        # Fallback to a simpler approach
        return xml_text[:length] if len(xml_text) > length else xml_text


@backoff.on_exception(backoff.expo, Exception, max_tries=3)
def detect_language_llm(
    fragment: str, model: str = DEFAULT_MODEL, temperature: float = 0.2
) -> str:
    """
    Use LLM to detect the predominant language of the text fragment.

    Args:
        fragment: Text fragment to analyze
        model: LLM model to use
        temperature: Temperature setting for the LLM

    Returns:
        Detected language code or name
    """
    # Check if API key is available
    if not os.getenv("OPENROUTER_API_KEY"):
        logger.warning("No API key available for LLM language detection")
        return "unknown"

    prompt = f"""
    Analyze the following text fragment and determine its predominant language.
    Respond with ONLY the language name in English e.g. "English", "Polish", "German").
    Do not include any explanation or additional text in your response.

    Text fragment:
    ```
    {fragment}
    ```
    """

    logger.info(f"Sending language detection request to LLM model: {model}")

    try:
        response = call_llm(prompt, model, temperature)

        # Extract just the language name (remove any quotation marks, periods, etc.)
        language = response.strip().lower()
        language = re.sub(r'["\'\.,]', "", language)
        logger.info(f"LLM detected language: {language}")
        return language
    except Exception as e:
        logger.error(f"Error during language detection: {e}")
        return "unknown"


# --- Pronunciation Cue Review with LLM ---
@backoff.on_exception(backoff.expo, Exception, max_tries=3)
def revise_neis_llm(
    neis: dict[str, str],
    language: str,
    model: str = DEFAULT_MODEL,
    temperature: float = 1.0,
) -> dict[str, str]:
    """
    Use LLM to review and potentially revise pronunciation cues for named entities.

    Args:
        neis: Dictionary mapping original entity names to pronunciation cues
        language: Detected language of the document
        model: LLM model to use
        temperature: Temperature setting for the LLM

    Returns:
        Revised dictionary of named entities and their pronunciation cues
    """
    if not neis:
        logger.info("No named entities to review")
        return {}

    # Check if API key is available
    if not os.getenv("OPENROUTER_API_KEY"):
        logger.warning("No API key available for LLM pronunciation review")
        return neis

    neis_json = json.dumps(neis, ensure_ascii=False, indent=2)

    prompt = f"""
    You are reviewing pronunciation cues for named entities in a {language} text-to-speech system. Here is a dictionary where:
    - Keys are the original named entities of interest (NEIs)
    - Values are simplified pronunciation cues for speakers of {language}

    For each entry, determine if the pronunciation cue is necessary, and produce a new dictionary with the original entities as keys and the following values:

    Option 1. KEEP the simplified cue as value if the original entity:
        - is a name or phrase that is very unusual, foreign, or ambiguous for speakers of {language}
        - has a completely non-intuitive pronunciation in {language} or actually means something different in {language}

    Option 2. REPLACE cues with the ORIGINAL entity spelling if:
        - the original entity is actually a native phrase in {language}
        - the original entity would be familiar to average {language} readers
        - the original entity is a longer phrase in English
        - the pronunciation cue is misleading or unnecessarily complicated

    Option 3. MAKE a new ADAPTED value (cue) that is not as arcane or complex as the cue we have

    Return ONLY a valid JSON dictionary with the original entities as keys and the new values (which are either the old cues or the original entities or the adapted cues).

    Named entities dictionary:
    {neis_json}
    """

    logger.info(f"Submitting pronunciation review request to LLM model: {model}")
    logger.debug("Waiting for LLM response...")

    try:
        response = call_llm(prompt, model, temperature)
        logger.debug("LLM response received, processing...")

        # Extract JSON dictionary from response
        json_pattern = r"\{[\s\S]*\}"
        json_match = re.search(json_pattern, response)

        if json_match:
            json_str = json_match.group(0)
            revised_neis = json.loads(json_str)
            logger.info(f"Successfully revised {len(revised_neis)} named entities")
            return revised_neis
        else:
            logger.warning("Could not extract valid JSON from LLM response")
            return neis
    except json.JSONDecodeError as e:
        logger.error(f"Error decoding JSON from LLM response: {e}")
        return neis
    except Exception as e:
        logger.error(f"Error during pronunciation cue review: {e}")
        return neis


# --- Common LLM Call Function ---
def call_llm(prompt: str, model: str, temperature: float) -> str:
    """
    Generic function to call the LLM and extract the response text.

    Args:
        prompt: The prompt to send to the LLM
        model: LLM model to use
        temperature: Temperature setting for the LLM

    Returns:
        Text response from the LLM
    """
    try:
        # Call the LLM and handle the response
        logger.debug(f"Calling LLM API with model={model}, temperature={temperature}")

        response = completion(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
        )

        logger.debug("LLM API call completed successfully")

        # Extract content from response
        response_any = cast(Any, response)
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
        msg = "Could not extract content from LLM response"
        raise ValueError(msg)

    except Exception as e:
        logger.error(f"Error calling LLM: {e}")

        # Attempt to use fallback model if different from current model
        if model != FALLBACK_MODEL:
            logger.warning(f"Attempting to use fallback model: {FALLBACK_MODEL}")
            try:
                logger.debug(
                    f"Calling fallback LLM API with model={FALLBACK_MODEL}, temperature={temperature}"
                )

                response = completion(
                    model=FALLBACK_MODEL,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=temperature,
                )

                logger.debug("Fallback LLM API call completed successfully")

                # Extract content from fallback response
                response_any = cast(Any, response)
                if (
                    hasattr(response_any, "choices")
                    and response_any.choices
                    and hasattr(response_any.choices[0], "message")
                    and hasattr(response_any.choices[0].message, "content")
                ):
                    content = response_any.choices[0].message.content
                    if content is not None:
                        logger.info("Successfully processed with fallback model")
                        return str(content)
            except Exception as fallback_error:
                logger.error(f"Fallback model also failed: {fallback_error}")

        raise


# --- Replace Pronunciation Cues in XML ---
def update_nei_cues(root: etree._Element, revised_neis: dict[str, str]) -> None:
    """
    Update <nei> tag contents in-place using revised_neis dict.

    Args:
        root: XML root element
        revised_neis: Dictionary mapping original entities to revised pronunciation cues
    """
    for nei in root.xpath("//nei"):
        orig = nei.get("orig")
        if orig and orig in revised_neis:
            # Save all attributes and the tail
            attrs = dict(nei.attrib)
            tail = nei.tail  # Save the tail text that appears after this element

            if tail:
                logger.debug(f"Preserving tail text for '{orig}': {tail[:30]}...")

            # Clear all children and text
            for child in nei:
                nei.remove(child)

            # Set the new text directly, removing any previous content
            nei.text = revised_neis[orig]

            # Preserve the original tail text
            nei.tail = tail

            # Restore all attributes that might have been affected
            for key, value in attrs.items():
                nei.set(key, value)

            # Verify the update was successful
            verification_text = "content not set"
            if nei.text:
                verification_text = (
                    nei.text[:30] + "..." if len(nei.text) > 30 else nei.text
                )

            verification_tail = "no tail"
            if nei.tail:
                verification_tail = (
                    nei.tail[:30] + "..." if len(nei.tail) > 30 else nei.tail
                )

            logger.debug(
                f"Updated nei tag for '{orig}' - content: '{verification_text}', tail: '{verification_tail}'"
            )


# --- Reduce Excessive Emphasis ---
def reduce_emphasis(xml_text: str, min_distance: int) -> str:
    """
    Remove <em> tags that are too close together (in words).
    Uses a more robust approach to handle XML structure.

    Args:
        xml_text: XML text to process
        min_distance: Minimum word distance required between <em> tags

    Returns:
        Processed XML text with reduced emphasis
    """
    # Safety check for empty input
    if not xml_text:
        logger.warning("Empty XML text provided to reduce_emphasis")
        return xml_text

    # Find all <em>...</em> spans and their positions with a more robust pattern
    # This pattern is designed to better handle nested tags and complex content
    em_pattern = r"<em(?:\s[^>]*)?>(.*?)</em>"
    em_spans = list(re.finditer(em_pattern, xml_text, re.DOTALL))
    if len(em_spans) < 2:
        return xml_text

    logger.info(f"Found {len(em_spans)} emphasized spans in the text")

    # Collect (start, end, text) for each <em>
    em_locs = [(m.start(), m.end(), m.group(1)) for m in em_spans]

    # Compute word distances and mark for removal
    to_unwrap = set()
    for i in range(1, len(em_locs)):
        prev_end = em_locs[i - 1][1]
        curr_start = em_locs[i][0]

        # Ensure indexes are valid
        if (
            prev_end >= len(xml_text)
            or curr_start >= len(xml_text)
            or prev_end > curr_start
        ):
            logger.warning(
                f"Invalid span indexes: prev_end={prev_end}, curr_start={curr_start}"
            )
            continue

        between = xml_text[prev_end:curr_start]
        # More robust word count that handles various unicode characters
        word_count = len(re.findall(r"\w+", between, re.UNICODE))
        if word_count < min_distance:
            to_unwrap.add(i)
            logger.debug(
                f"Marking emphasis span {i} for removal (distance: {word_count} words)"
            )

    # Unwrap by removing <em>...</em> tags for marked indices in reverse order
    # to avoid index shifting problems
    new_text = xml_text
    for idx in sorted(to_unwrap, reverse=True):
        if idx >= len(em_spans):
            logger.warning(
                f"Invalid emphasis span index: {idx}, max: {len(em_spans) - 1}"
            )
            continue

        m = em_spans[idx]
        start, end = m.start(), m.end()
        content = m.group(1)

        # Safety check for valid indices
        if start < 0 or end > len(new_text) or start > end:
            logger.warning(
                f"Invalid text positions: start={start}, end={end}, len={len(new_text)}"
            )
            continue

        # Replace <em>...</em> with just the content
        new_text = new_text[:start] + content + new_text[end:]

    logger.info(f"Removed {len(to_unwrap)} emphasis spans that were too close")
    return new_text


# --- Main Processing Logic ---
def process_document(
    input_file: str,
    output_file: str,
    em: int | None = None,
    model: str = DEFAULT_MODEL,
    temperature: float = DEFAULT_TEMPERATURE,
    verbose: bool = False,
) -> None:
    """
    Main processing function for malmo_tonedown.

    Args:
        input_file: Path to the input XML file
        output_file: Path to the output XML file
        em: Minimum word distance between emphasis tags (optional)
        model: LLM model to use
        temperature: Temperature setting for the LLM
        verbose: Whether to enable verbose logging
    """
    # Configure logging
    log_level = "DEBUG" if verbose else "INFO"
    logger.remove()
    logger.add(lambda msg: console.print(msg, highlight=False), level=log_level)

    # Check if input file exists
    if not os.path.exists(input_file):
        logger.error(f"Input file not found: {input_file}")
        msg = f"Input file not found: {input_file}"
        raise FileNotFoundError(msg)

    logger.info(f"Processing XML document: {input_file}")
    logger.info(f"Output will be saved to: {output_file}")
    logger.info(f"Using LLM model: {model}")

    # Resolve model name if it's a key in MODEL_DICT
    if model in MODEL_DICT:
        model = MODEL_DICT[model]
        logger.info(f"Resolved model name to: {model}")

    # Read XML content - save original for comparison
    with open(input_file, encoding="utf-8") as f:
        original_xml_text = f.read()

    # Parse XML
    root = parse_xml(input_file)

    # Step 1: Build NEIs dict and update <nei> tags
    neis = process_nei_tags(root)
    if neis:
        logger.info(f"Found {len(neis)} named entities of interest")

        # IMPORTANT: Print original NEIs dictionary BEFORE sending to LLM
        if verbose:
            console.print("\n[bold green]Original Named Entities:[/bold green]")
            console.print(
                json.dumps(neis, ensure_ascii=False, indent=2), highlight=False
            )
            logger.debug(f"Initial NEIs dict contains {len(neis)} entries")
    else:
        logger.info("No named entities found in the document")

    # Step 2: Language detection
    fragment = extract_middle_fragment(original_xml_text)
    language = detect_language_llm(fragment, model, temperature=1.0)
    logger.info(f"Detected document language: {language}")

    # Step 3: Pronunciation cue review (only if we found NEIs)
    if neis:
        # Print a clear message indicating the start of LLM processing
        logger.info(f"Starting LLM review of {len(neis)} named entities...")
        revised_neis = revise_neis_llm(neis, language, model, temperature)
        logger.info("LLM review completed")

        # Always log the count of revisions
        unchanged_count = sum(1 for k, v in revised_neis.items() if neis.get(k) == v)
        changed_count = len(revised_neis) - unchanged_count
        logger.info(
            f"Named entities revised: {changed_count} changed, {unchanged_count} unchanged"
        )

        if verbose:
            console.print("\n[bold blue]Revised Named Entities:[/bold blue]")
            console.print(
                json.dumps(revised_neis, ensure_ascii=False, indent=2), highlight=False
            )

            # Log changes for clarity
            if changed_count > 0:
                console.print("\n[bold yellow]Changes Made:[/bold yellow]")
                for key, new_value in revised_neis.items():
                    old_value = neis.get(key)
                    if old_value != new_value:
                        console.print(f"[green]{key}[/green]:")
                        console.print(f"  [red]- {old_value}[/red]")
                        console.print(f"  [green]+ {new_value}[/green]")

        # Step 4: Replace cues in XML
        update_nei_cues(root, revised_neis)
        logger.info("Updated pronunciation cues in the document")

    # Step 5: Get XML with proper encoding, trying to preserve formatting
    try:
        # Use method='xml' to ensure proper XML serialization
        xml_out = etree.tostring(
            root,
            encoding="utf-8",
            xml_declaration=True,
            pretty_print=False,  # Don't use pretty_print to avoid text node changes
            method="xml",
            with_tail=True,  # Ensure tail text is included
        ).decode("utf-8")

        # Fix any self-closing tags that should be properly closed
        # (especially for <nei> tags which must not be self-closing)
        xml_out = re.sub(r"<nei([^>]*)/>(?!</nei>)", r"<nei\1></nei>", xml_out)

        # Validate that all nei tags are properly closed and not self-closing
        nei_count = xml_out.count("<nei")
        nei_close_count = xml_out.count("</nei>")
        if nei_count != nei_close_count:
            logger.warning(
                f"Potential issue with nei tags: {nei_count} opening tags but {nei_close_count} closing tags"
            )

    except Exception as e:
        logger.error(f"Error serializing XML: {e}")
        # Fallback to string representation if tostring fails
        try:
            # Convert to string with safe fallback
            xml_bytes = etree.tostring(
                root,
                pretty_print=False,
                encoding="utf-8",
                with_tail=True,  # Ensure tail text is included
            )
            xml_out = xml_bytes.decode("utf-8") if xml_bytes is not None else ""
            logger.warning("Using fallback XML serialization method")
        except Exception:
            logger.error(
                "Both XML serialization methods failed, using basic string conversion"
            )
            # Last resort fallback - this should always work unless root is None
            xml_out = (
                str(etree.tostring(root, encoding="utf-8"))
                .replace("b'", "")
                .replace("'", "")
            )

    # Ensure xml_out is not None before continuing
    if xml_out is None:
        logger.error("XML serialization returned None, cannot continue")
        msg = "Failed to serialize XML"
        raise ValueError(msg)

    # Step 6: Reduce excessive emphasis if --em is set
    if em is not None:
        logger.info(f"Reducing excessive emphasis with minimum distance of {em} words")
        xml_out = reduce_emphasis(xml_out, em)

    # Step 7: Validate the resulting XML to detect corruption
    try:
        # Try to parse the resulting XML as a validation step
        validation_parser = etree.XMLParser(recover=True)
        validation_root = etree.fromstring(xml_out.encode("utf-8"), validation_parser)

        # Count items in original and processed XML
        original_item_count = len(
            etree.fromstring(
                original_xml_text.encode("utf-8"), etree.XMLParser(recover=True)
            ).xpath("//item")
        )
        processed_item_count = len(validation_root.xpath("//item"))

        if original_item_count != processed_item_count:
            logger.warning(
                f"XML structure changed: original had {original_item_count} items, "
                f"processed has {processed_item_count} items"
            )

        # Check if any <nei> tags are self-closing or improperly formatted
        for nei in validation_root.xpath("//nei"):
            if nei.tag is None or not nei.text:
                # Convert bytes to string before logging
                nei_bytes = etree.tostring(nei, encoding="utf-8")
                nei_str = (
                    nei_bytes.decode("utf-8")
                    if nei_bytes is not None
                    else "Unknown tag"
                )
                logger.warning(f"Potentially malformed <nei> tag detected: {nei_str}")

        logger.info("XML validation passed")

    except Exception as e:
        logger.error(f"XML validation failed: {e}")
        logger.warning("Output XML may be corrupted, attempting to repair")

        # Provide a clean empty dict if we don't have revised_neis
        # Try to repair by reprocessing from scratch with more conservative options
        try:
            logger.info("Attempting XML repair...")

            # Use a different serialization approach that prioritizes content preservation
            if os.path.exists(input_file):
                # Just re-parse the file and serialize it again without modifying nei tags
                logger.info("Repairing by re-parsing original file")
                repaired_xml = repair_xml_by_reparsing(input_file)
            else:
                # Use a different serialization approach with the current root
                logger.info("Repairing by re-serializing current tree")
                repaired_xml = etree.tostring(
                    root,
                    encoding="utf-8",
                    xml_declaration=True,
                    pretty_print=False,  # Avoid pretty_print to prevent text node changes
                    method="xml",
                    with_tail=True,
                ).decode("utf-8")

            # Keep the repaired version
            xml_out = repaired_xml
            logger.info("XML repair successful")
        except Exception as repair_err:
            logger.error(f"XML repair failed: {repair_err}")
            # At this point, we'll use what we have and hope for the best

    # Write output
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(xml_out)
    logger.info(f"Successfully wrote output to {output_file}")


# --- XML Repair Function ---
def repair_xml_by_reparsing(file_path: str) -> str:
    """
    Repair XML by re-parsing the original file.

    Args:
        file_path: Path to the original XML file

    Returns:
        Repaired XML as string
    """
    # Re-parse with even more conservative settings
    repair_parser = etree.XMLParser(
        remove_blank_text=False,
        recover=True,
        resolve_entities=False,
        huge_tree=True,
        remove_comments=False,
        remove_pis=False,
    )

    with open(file_path, encoding="utf-8") as f:
        original_content = f.read()

    repair_root = etree.fromstring(original_content.encode("utf-8"), repair_parser)

    # Just serialize without modifying
    return etree.tostring(
        repair_root,
        encoding="utf-8",
        xml_declaration=True,
        pretty_print=False,
        method="xml",
        with_tail=True,
    ).decode("utf-8")


# --- CLI Entrypoint ---
def main(
    input_file: str,
    output_file: str,
    em: int | None = None,
    model: str = DEFAULT_MODEL,
    temperature: float = DEFAULT_TEMPERATURE,
    verbose: bool = True,
) -> None:
    """
    CLI entrypoint for malmo_tonedown.

    Args:
        input_file: Path to the input XML file
        output_file: Path to the output XML file
        em: Minimum word distance between emphasis tags (optional)
        model: LLM model identifier or key from MODEL_DICT
        temperature: LLM temperature setting (0.0-1.0)
        verbose: Whether to enable verbose logging
    """
    if verbose:
        logger.level("DEBUG")
        _turn_on_debug()

    try:
        process_document(
            input_file=input_file,
            output_file=output_file,
            em=em,
            model=model,
            temperature=temperature,
            verbose=verbose,
        )
    except Exception as e:
        logger.error(f"Error in main function: {e}")
        if verbose:
            import traceback

            logger.error(traceback.format_exc())
        raise


if __name__ == "__main__":
    fire.Fire(main)
