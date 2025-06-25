#!/usr/bin/env python3
# this_file: src/e11ocutionist/elevenlabs_converter.py
"""
ElevenLabs converter for e11ocutionist.

This module converts processed XML files to a text format compatible with ElevenLabs text-to-speech,
with options for processing dialog and handling plaintext input.
"""

import re
from pathlib import Path
from typing import Any
from lxml import etree
from loguru import logger

from .utils import clean_consecutive_newlines, parse_xml, unescape_xml_chars


def extract_text_from_xml(xml_root: etree._Element) -> str:
    """
    Extract text from XML document, processing each item element.

    Args:
        xml_root: XML root element

    Returns:
        Extracted and processed text

    Used in:
    - e11ocutionist/elevenlabs_converter.py
    """
    result = []

    for chunk in xml_root.xpath("//chunk"):
        for item in chunk.xpath(".//item"):
            # Get the inner XML of the item
            inner_xml = etree.tostring(item, encoding="utf-8", method="xml").decode(
                "utf-8"
            )
            # Extract just the content between the opening and closing tags
            item_text = re.sub(
                r"^<item[^>]*>(.*)</item>$", r"\1", inner_xml, flags=re.DOTALL
            )

            # Process item text
            # Convert <em>...</em> to "..."
            item_text = re.sub(r"<em>(.*?)</em>", r'"\1"', item_text, flags=re.DOTALL)
            # Convert <nei new="true">...</nei> to "..."
            # Using a temporarily simpler regex for diagnosis
            item_text = re.sub(
                r'<nei[^>]*new="true"[^>]*>(.*?)</nei>', r'"\1"', item_text, flags=re.DOTALL
            )
            # Convert remaining <nei>...</nei> (that don't have new="true") to its content
            item_text = re.sub(r"<nei(?![^>]*new=\"true\")[^>]*>(.*?)</nei>", r"\1", item_text, flags=re.DOTALL)
            # Replace <hr/> with <break time="0.5s" />
            item_text = re.sub(r"<hr\s*/?>", r'<break time="0.5s" />', item_text) # This typically doesn't span lines
            # Strip any remaining HTML/XML tags
            item_text = re.sub(r"<[^>]+>", "", item_text)

            # Unescape HTML entities
            item_text = unescape_xml_chars(item_text)

            result.append(item_text.strip()) # Strip whitespace from each item's text

    # Join all items with double newlines
    return "\n\n".join(result)


def process_dialog(text: str) -> str:
    """
    Process dialog in text, converting em dashes to appropriate quotation marks.

    Args:
        text: Text with dialog lines

    Returns:
        Text with processed dialog

    Used in:
    - e11ocutionist/elevenlabs_converter.py
    """
    # Adapted from legacy malmo_11labs.py
    lines = text.split("\n")
    processed_lines = []
    q_open = False

    for line_str in lines:
        current_line_parts = []
        # Split by " — " to handle toggles. Process parts.
        parts = line_str.split(" — ")

        for i, part_content in enumerate(parts):
            is_dialog_start_of_part = part_content.strip().startswith("— ") if i == 0 else False # Only first part can start with "— "

            if is_dialog_start_of_part:
                if q_open: # If a q was open from previous line or part
                    current_line_parts.append("</q>")
                current_line_parts.append("<q>" + part_content.strip()[2:])
                q_open = True
            else: # Not a dialog start for this part
                if i == 0 and not q_open : # Non-dialog line start
                    current_line_parts.append(part_content)
                elif i==0 and q_open: # Continues dialog from previous line
                    current_line_parts.append(part_content)
                elif i > 0 : # This is a toggle part
                    if q_open:
                        current_line_parts.append("</q><q>" + part_content)
                    else: # Should not happen if previous part closed q correctly
                        current_line_parts.append("<q>" + part_content) # Start q if it was closed by toggle
                    # q_open state is effectively toggled by </q><q>

            # After processing a part, if it's a dialog part and it's the last part of the line,
            # q_open should reflect its state.
            # If it's not the last part of the line, the toggle </q><q> handles the state.

        # Logic to close <q> if the next line is not a dialog or it's the end of text
        # This is tricky because we only have current line here.
        # The legacy code closed the <q> on the *previous* line when it saw a non-dialog line.
        # Or at the very end. This needs to be handled after joining.

        processed_lines.append("".join(current_line_parts))

    # Post-process to ensure all <q> are closed correctly based on line context
    # This is difficult to do perfectly without a more complex state machine or lookahead
    # The legacy method of closing on the previous line when a non-dialog line is encountered
    # is simpler.

    # Let's try the simpler legacy line-by-line state machine:
    final_processed_lines = []
    q_actually_open = False
    for line_to_finalize in processed_lines: # These lines now have <q>...</q><q>...
        # This simple approach might not be enough.
        # The core issue is that a single line can be "— dialog — narration — dialog."
        # which requires <q>dialog</q> narration <q>dialog</q>.
        # The split by " — " and then rejoining parts with toggles handles this.
        # The main thing is to ensure the overall line, if it started dialog, ends dialog.
        # And if a non-dialog line follows a dialog line, the dialog line is closed.

        # The current `processed_lines` from above loop might be too complex.
        # Let's use a clearer implementation based on legacy.
        pass # Placeholder for now. I need to rethink this.

    # Re-implementing with clearer state based on legacy
    processed_lines = []
    q_is_open_for_current_line = False
    for line_str in lines:
        line_parts = line_str.split(" — ")
        output_line = ""

        # Determine if the line as a whole starts dialog
        first_part_stripped = line_parts[0].strip()
        if first_part_stripped.startswith("— "):
            if q_is_open_for_current_line and output_line: # Should not happen if output_line is fresh
                 output_line += "</q>" # Should be on previous line actually
            output_line += "<q>" + first_part_stripped[2:]
            q_is_open_for_current_line = True
            line_parts[0] = first_part_stripped[2:] # Content after "— "
        elif q_is_open_for_current_line : # Previous line ended open, this one is continuation
             output_line += line_parts[0]
        else: # Normal line
            output_line += line_parts[0]

        # Handle toggles for parts > 0
        for k in range(1, len(line_parts)):
            if q_is_open_for_current_line:
                output_line += "</q>"
            else:
                output_line += "<q>"
            q_is_open_for_current_line = not q_is_open_for_current_line
            output_line += line_parts[k]

        # If the line ended with an open q, and next line is not dialog, or it's last line
        # This part is tricky. The legacy code did this by looking at next line or at end.
        # For now, if a line started with <q> or had <q> in it, assume it should close if not explicitly toggled off.
        if ("<q>" in output_line or output_line.startswith("<q>")) and q_is_open_for_current_line:
            # Check if this line is the last dialog line in a sequence
            # This requires looking ahead, which is complex here.
            # A simpler approach: if q_is_open_for_current_line is true at end of processing this line,
            # it means it expects to be continued or it's the last part of a toggle.
            # The overall open/close is better handled by post-processing the joined string.
            pass


        processed_lines.append(output_line)

    # At this point, `processed_lines` contains lines with internal `<q>...</q>` toggles.
    # We need a pass to ensure that if a line started a dialog, and the next line doesn't, the first line's dialog is closed.
    # And if the last line is dialog, it's closed.

    # Simplified approach: Tag dialog lines, then post-process for quotes.
    # This is what the original current code was trying. The issue was the lookahead.
    # Let's revert to the version just before this complex change and fix its lookahead.
    # The version was:
    # lines = text.split("\n")
    # processed_lines = []
    # for i, current_line_original in enumerate(lines):
    #     line_to_process = current_line_original ...
    # This logic was almost there.

    # Reverting to slightly fixed version of what was present before this big change:
    lines = text.split("\n")
    processed_lines = []
    in_dialog_context = False # True if we are generally in a dialog across multiple lines

    for idx, line_str_original in enumerate(lines):
        line_to_process = line_str_original # work on a copy

        is_current_line_dialog_start = line_to_process.strip().startswith("— ")

        if is_current_line_dialog_start:
            if in_dialog_context and processed_lines: # Dialog was open from previous line. Close it.
                 processed_lines[-1] += "</q>"
            line_to_process = "<q>" + line_to_process.strip()[2:] # Start new dialog
            in_dialog_context = True
        elif in_dialog_context : # This line is not a dialog start, but previous was.
            if processed_lines : processed_lines[-1] += "</q>" # Close previous line's dialog
            in_dialog_context = False

        # Handle internal toggles " — " for lines that are now considered dialog
        if "<q>" in line_to_process or is_current_line_dialog_start: # if it's a dialog line
            parts = line_to_process.split(" — ")
            if len(parts) > 1: # Contains " — " toggles
                # First part might already start with <q>
                current_part_is_open = parts[0].startswith("<q>")
                temp_line = parts[0]
                for k in range(1, len(parts)):
                    if current_part_is_open: temp_line += "</q>"
                    else: temp_line += "<q>"
                    current_part_is_open = not current_part_is_open
                    temp_line += parts[k]
                line_to_process = temp_line
                # Ensure the line ends in a consistent state regarding q_open for the context
                in_dialog_context = current_part_is_open


        processed_lines.append(line_to_process)

    if in_dialog_context and processed_lines: # If overall dialog context is still open at the end
        processed_lines[-1] += "</q>"

    result = "\n".join(processed_lines)

    # Final post-processing for dialog
    # Replace <q> at start of line with opening smart quote
    result = re.sub(r"^\s*<q>", '"', result, flags=re.MULTILINE)
    # Replace </q> at end of line with closing smart quote
    result = re.sub(r"</q>\s*$", '"', result, flags=re.MULTILINE)
    # Replace other <q> and </q> tags with special markers for ElevenLabs
    result = re.sub(r"<q>", "; OPEN_Q", result)
    result = re.sub(r"</q>", "CLOSE_Q;", result)

    # Replace break tags with ellipsis
    result = re.sub(r'<break time="[^"]*"\s*/>', "...", result)

    # Normalize consecutive newlines
    result = clean_consecutive_newlines(result)

    # Add some spacing for readability
    result = re.sub(r"\n", "\n\n", result)

    return result


def process_document(
    input_file: str,
    output_file: str,
    dialog: bool = True,
    plaintext: bool = False,
    verbose: bool = False,
) -> dict[str, Any]:
    """
    Convert an XML document to a text format compatible with ElevenLabs.

    Args:
        input_file: Path to the input XML file
        output_file: Path to save the converted text output
        dialog: Whether to process dialog (default: True)
        plaintext: Whether to treat the input as plaintext (default: False)
        verbose: Enable verbose logging

    Returns:
        Dictionary with processing statistics
    """
    if verbose:
        logger.info(f"Processing {input_file} to {output_file}")
        logger.info(f"Dialog processing: {dialog}, Plaintext mode: {plaintext}")

    # Read the input file
    input_content = Path(input_file).read_text(encoding="utf-8")

    # Process the content
    output_text = ""

    if plaintext:
        # Plaintext mode - just process the content directly
        output_text = input_content
        if dialog:
            output_text = process_dialog(output_text)
    else:
        # XML mode - parse the XML and extract the text
        xml_root = parse_xml(input_content)
        if xml_root is None:
            logger.error(f"Failed to parse XML from {input_file}")
            return {"success": False, "error": "XML parsing failed"}

        # Check for actual content if in XML mode
        if not plaintext and not xml_root.xpath("//chunk//item"):
            logger.error(f"No valid content (chunk/item) found in XML {input_file}")
            # Write empty output file to avoid leftover from previous runs
            Path(output_file).write_text("", encoding="utf-8")
            return {"success": False, "error": "No valid content in XML"}

        output_text = extract_text_from_xml(xml_root)
        if dialog:
            output_text = process_dialog(output_text)

    # Write the output file
    Path(output_file).write_text(output_text, encoding="utf-8")

    if verbose:
        logger.info(f"Conversion completed: {output_file}")

    return {
        "input_file": input_file,
        "output_file": output_file,
        "dialog_processed": dialog,
        "plaintext_mode": plaintext,
        "success": True,
    }
