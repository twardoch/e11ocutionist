#!/usr/bin/env -S uv run -s
# /// script
# dependencies = ["fire", "lxml", "loguru", "rich", "types-lxml", "litellm<=1.67.2"]
# ///
# this_file: malmo_11labs.py

import fire
import re
import os
import html
from lxml import etree
from litellm._logging import _turn_on_debug
from loguru import logger
from rich.console import Console
from rich.progress import (
    Progress,
    TextColumn,
    BarColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

# Set up Rich console for pretty output
console = Console()

# Unicode smart quotes
OPEN_Q = "\u201c"  # Left double quotation mark
CLOSE_Q = "\u201d"  # Right double quotation mark

# Quote character sets for normalization
OPEN_QQ = """❛❝🙶🙸»›'‚‛"„‟⹂⌜「『〝｢﹁﹃⠴"""
CLOSE_QQ = """❜❞🙷«‹'"⌝」』〞〟｣﹂﹀⠦"""


def parse_xml(input_file: str) -> etree._Element:
    """
    Parse the input XML file.

    Args:
        input_file: Path to the input XML file

    Returns:
        Parsed XML element tree

    Raises:
        ValueError: If the XML cannot be parsed
    """
    try:
        parser = etree.XMLParser(remove_blank_text=False, recover=True)
        with open(input_file, encoding="utf-8") as f:
            xml_content = f.read()

        if not xml_content.strip():
            msg = f"File is empty: {input_file}"
            raise ValueError(msg)

        tree = etree.fromstring(xml_content.encode("utf-8"), parser)
        if tree is None:
            msg = f"Failed to parse XML: {input_file}"
            raise ValueError(msg)
        return tree
    except Exception as e:
        logger.error(f"Error parsing XML: {e}")
        # Check if file exists and has readable content
        if not os.path.exists(input_file):
            msg = f"Input file not found: {input_file}"
            raise ValueError(msg)
        elif os.path.getsize(input_file) == 0:
            msg = f"Input file is empty: {input_file}"
            raise ValueError(msg)
        else:
            # Include a small sample of the file in the error message
            with open(input_file, encoding="utf-8", errors="replace") as f:
                sample = f.read(200)
            msg = (
                f"Failed to parse XML file: {input_file}\n"
                f"Error: {e!s}\n"
                f"File sample: {sample}..."
            )
            raise ValueError(msg)


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


def normalize_consecutive_quotes(text: str) -> str:
    """
    Replace multiple consecutive opening or closing quotes with a single
    smart quote.

    Args:
        text: Input text with potentially consecutive quotes

    Returns:
        Text with normalized quotation marks
    """
    # Create regex patterns for consecutive opening and closing quotes
    opening_pattern = f"[{re.escape(OPEN_QQ)}]+"
    closing_pattern = f"[{re.escape(CLOSE_QQ)}]+"

    # Replace consecutive opening quotes with a single smart opening quote
    text = re.sub(opening_pattern, OPEN_Q, text)

    # Replace consecutive closing quotes with a single smart closing quote
    text = re.sub(closing_pattern, CLOSE_Q, text)

    return text


def postprocess(text: str) -> str:
    """
    Final clean-ups:
    1) turn &lt; &gt; into real brackets
    2) collapse 3+ consecutive newlines -> exactly 2
    3) normalize consecutive quotation marks
    """
    text = html.unescape(text)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    text = normalize_consecutive_quotes(text)
    return text


def final_postprocess(text: str) -> str:
    """
    Apply final text formatting rules:
    1) Replace start of line + <q> by the opening quote
    2) Replace </q> + the end of line by the closing quote
    3) Replace <q> by space + the opening quote
    4) Replace </q> by the closing quote + space
    5) Replace the portion from <break time= to the nearest /> by ...
    """
    # Replace start of line + <q> by the opening quote
    text = re.sub(r"^<q>", OPEN_Q, text, flags=re.MULTILINE)

    # Replace </q> + the end of line by the closing quote
    text = re.sub(r"</q>$", CLOSE_Q, text, flags=re.MULTILINE)

    # Replace <q> by space + the opening quote
    text = re.sub(r"<q>", f"; {OPEN_Q}", text)

    # Replace </q> by the closing quote + space
    text = re.sub(r"</q>", f"{CLOSE_Q}; ", text)

    # Replace the portion from <break time= to the nearest /> by ...
    text = re.sub(r"<break time=[^/>]+/>", "...", text)

    # Replace one newline with two
    text = re.sub(r"\n", "\n\n", text)

    return text


def process_dialog_lines(text: str) -> str:
    """
    Process dialog lines to add proper <q></q> tags.

    Dialog lines start with "— " (em dash + space).
    Dialog toggles are " — " (space + em dash + space).

    Args:
        text: Text to process

    Returns:
        Processed text with <q></q> tags for dialogs
    """
    lines = text.split("\n")
    processed_lines = []

    for line in lines:
        # Check if this is a dialog line (starts with "— ")
        if line.startswith("— "):
            # Replace the dialog open with <q>
            processed_line = "<q>" + line[2:]

            # Track the current tag state (True = <q> was last, False = </q> was last)
            tag_open = True

            # Find all dialog toggles
            toggle_positions = [m.start() for m in re.finditer(" — ", processed_line)]

            # Process each toggle
            offset = 0  # Adjustment for the changing string length as we replace
            for pos in toggle_positions:
                adjusted_pos = pos + offset
                if tag_open:
                    # Replace with closing tag
                    processed_line = (
                        processed_line[:adjusted_pos]
                        + "</q>"
                        + processed_line[adjusted_pos + 3 :]
                    )
                    offset += len("</q>") - 3  # -3 for the replaced " — "
                else:
                    # Replace with opening tag
                    processed_line = (
                        processed_line[:adjusted_pos]
                        + "<q>"
                        + processed_line[adjusted_pos + 3 :]
                    )
                    offset += len("<q>") - 3  # -3 for the replaced " — "

                tag_open = not tag_open  # Toggle the state

            # Close any open dialog tag at the end of the line
            if tag_open:
                processed_line += "</q>"

            processed_lines.append(processed_line)
        else:
            processed_lines.append(line)

    return "\n".join(processed_lines)


def process_item_content(item_element: etree._Element) -> str:
    """
    Process the content of an item element to extract and format text.

    Args:
        item_element: Item element from XML

    Returns:
        Formatted text from the item
    """
    # -------------------------------------------------
    # 0.  get the raw innerXML of <item> … </item>
    raw = etree.tostring(item_element, encoding="unicode", method="xml")
    inner = re.search(r"<item[^>]*>(.*?)</item>", raw, re.DOTALL)
    if not inner:
        return ""
    txt = inner.group(1)

    # -------------------------------------------------
    # 1.  <em> … </em>  →  "…"        (smart quotes)
    txt = re.sub(r"<em>(.*?)</em>", rf"{OPEN_Q}\1{CLOSE_Q}", txt, flags=re.DOTALL)

    # -------------------------------------------------
    # 2a. <nei … new="true" …> … </nei> →  "…"
    txt = re.sub(
        r'<nei[^>]*\bnew="true"[^>]*>(.*?)</nei>',
        rf"{OPEN_Q}\1{CLOSE_Q}",
        txt,
        flags=re.DOTALL,
    )

    # 2b. remaining <nei …> … </nei>      →  plain content
    txt = re.sub(r"<nei[^>]*>(.*?)</nei>", r"\1", txt, flags=re.DOTALL)

    # -------------------------------------------------
    # 3.  <hr/> (any spelling) → escaped break tag
    txt = re.sub(r"<hr\s*/?>", r'&lt;break time="0.5s" /&gt;', txt)

    # -------------------------------------------------
    # 4.  strip **all** remaining tags
    txt = re.sub(r"<[^>]+>", "", txt)

    # -------------------------------------------------
    # 5.  unescape &lt; &gt;  +  squeeze blank lines
    return postprocess(txt)


def process_chunk(chunk: etree._Element) -> list[str]:
    """
    Process all items in a chunk.

    Args:
        chunk: Chunk element from XML

    Returns:
        List of processed text lines from the chunk
    """
    processed_lines = []
    items = get_items_from_chunk(chunk)

    for item in items:
        processed_text = process_item_content(item)
        if processed_text:
            processed_lines.append(processed_text)

    return processed_lines


def process_document(
    input_file: str,
    output_file: str,
    verbose: bool = False,
    dialog: bool = False,
) -> None:
    """
    Process the entire XML document and generate output file.

    Args:
        input_file: Path to the input XML file
        output_file: Path to the output file
        verbose: Whether to enable verbose logging
        dialog: Whether to process dialog lines with <q></q> tags
    """
    # Configure logging
    log_level = "DEBUG" if verbose else "INFO"
    logger.remove()
    logger.add(lambda msg: console.print(msg, highlight=False), level=log_level)

    logger.info(f"Processing XML document: {input_file}")
    logger.info(f"Output will be saved to: {output_file}")
    if dialog:
        logger.info("Dialog processing is enabled")

    # Parse the XML
    root = parse_xml(input_file)
    chunks = get_chunks(root)
    logger.info(f"Found {len(chunks)} chunks in the document")

    all_lines = []

    # Process each chunk
    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console,
    ) as progress:
        process_task = progress.add_task("Processing chunks", total=len(chunks))

        for i, chunk in enumerate(chunks):
            logger.debug(
                f"Processing chunk {i + 1}/{len(chunks)} "
                f"(id: {chunk.get('id', 'unknown')})"
            )

            # Process the chunk
            processed_lines = process_chunk(chunk)
            all_lines.extend(processed_lines)

            progress.update(process_task, advance=1)

    # Join all lines
    output_text = "\n\n".join(all_lines)

    # Process dialog lines if enabled
    if dialog:
        logger.debug("Processing dialog lines")
        output_text = process_dialog_lines(output_text)

    # Apply final postprocessing
    logger.debug("Applying final postprocessing")
    output_text = final_postprocess(output_text)

    # Write to output file
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(output_text)

    logger.info(f"Processing complete. Output saved to {output_file}")


def main(
    input_file: str,
    output_file: str,
    verbose: bool = True,
    dialog: bool = False,
    plaintext: bool = False,
) -> None:
    """
    Convert XML to 11Labs compatible text format.

    Args:
        input_file: Path to the input XML file
        output_file: Path to the output file
        verbose: Whether to enable verbose logging
        dialog: Whether to process dialog lines with <q></q> tags
        plaintext: Process as plain text instead of XML
    """
    if verbose:
        logger.level("DEBUG")
        _turn_on_debug()

    try:
        # Check if input file exists
        if not os.path.exists(input_file):
            console.print(
                f"[bold red]Error:[/bold red] Input file not found: {input_file}"
            )
            return

        # If plaintext option is set, process as plain text
        if plaintext:
            logger.info(f"Processing as plain text: {input_file}")
            with open(input_file, encoding="utf-8", errors="replace") as f:
                text = f.read()

            # Process dialog lines if enabled
            if dialog:
                logger.debug("Processing dialog lines")
                text = process_dialog_lines(text)

            # Apply final postprocessing
            logger.debug("Applying final postprocessing")
            text = final_postprocess(text)

            # Write to output file
            with open(output_file, "w", encoding="utf-8") as f:
                f.write(text)

            console.print(
                f"[bold green]Success:[/bold green] Processed {input_file} "
                f"as plain text"
            )
            return

        # Process the document as XML
        process_document(input_file, output_file, verbose, dialog)

        console.print(
            f"[bold green]Success:[/bold green] Converted {input_file} to {output_file}"
        )

    except ValueError as e:
        error_msg = str(e)
        console.print(f"[bold red]Error:[/bold red] {error_msg}")

        # Check if this might be a plain text file
        try:
            with open(input_file, encoding="utf-8", errors="replace") as f:
                sample = f.read(100)

            # Simple heuristic: if it doesn't start with < or has lines without tags
            if not sample.strip().startswith("<") or any(
                line.strip() and not ("<" in line and ">" in line)
                for line in sample.split("\n")
                if line.strip()
            ):
                console.print(
                    "[bold yellow]Suggestion:[/bold yellow] This appears to be a "
                    "plain text file, not XML. Try again with the --plaintext flag:"
                )
                console.print(
                    f"[bold cyan]python -m malmo_11labs {input_file} {output_file} "
                    f"--plaintext{'--dialog' if dialog else ''}[/bold cyan]"
                )
        except Exception:
            # Ignore any errors in suggestion logic
            pass

        logger.exception("An error occurred during processing")
        raise

    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {e!s}")
        logger.exception("An error occurred during processing")
        raise


if __name__ == "__main__":
    fire.Fire(main)
