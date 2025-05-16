#!/usr/bin/env -S uv run -s
# /// script
# dependencies = ["fire", "rich"]
# ///
# this_file: neifix.py
"""
Script to parse XML files and transform content inside <nei> tags.
Transforms text by keeping initial uppercase letters for words,
converting other uppercase letters to lowercase, and removing hyphens.
"""

import re
import sys
from pathlib import Path

import fire
from rich.console import Console


console = Console()
error_console = Console(stderr=True)


def transform_nei_content(match, verbose=False):
    """
    Transform the content inside <nei> tags according to rules:
    - Keep initial uppercase letter for every word
    - Convert every uppercase inside a word to lowercase
    - Eliminate hyphens

    Special handling for:
    - Preserve acronyms (words consisting of single letters separated by spaces)
    """
    # Extract the full tag, the opening tag with attributes, and the content
    full_tag = match.group(0)
    opening_tag = match.group(1)
    content = match.group(2)

    if verbose:
        console.print(f"Processing tag: [bold cyan]{full_tag}[/]")
        console.print(f"Opening tag: [bold blue]{opening_tag}[/]")
        console.print(f"Content: [bold green]{content}[/]")

    # Process each word in the content
    words = content.split()
    transformed_words = []

    for word in words:
        # Check if it's likely an acronym (single letters separated by spaces)
        if len(word) == 1 or (
            word.replace("-", "").isalpha() and len(word.replace("-", "")) == 1
        ):
            # Preserve single letters (likely parts of acronyms)
            transformed_words.append(word)
            continue

        # Remove hyphens and process capitalization
        word_without_hyphens = word.replace("-", "")

        if len(word_without_hyphens) > 0:
            # Keep first letter as is, convert rest to lowercase
            transformed_word = (
                word_without_hyphens[0] + word_without_hyphens[1:].lower()
            )
            transformed_words.append(transformed_word)

    # Join the transformed words and rebuild the tag
    transformed_content = " ".join(transformed_words)
    transformed_tag = f"{opening_tag}{transformed_content}</nei>"

    if verbose:
        console.print(f"Transformed content: [bold green]{transformed_content}[/]")
        console.print(f"Transformed tag: [bold cyan]{transformed_tag}[/]")

    return transformed_tag


def process_file(input_path, output_path=None, verbose=False):
    """Process the input file and write the transformed content to the output file."""
    try:
        input_path = Path(input_path)
        if not input_path.exists():
            error_console.print(f"[bold red]Input file not found: {input_path}[/]")
            return False

        with open(input_path, encoding="utf-8") as f:
            content = f.read()

        if verbose:
            console.print(f"[bold]Processing file:[/] {input_path}")
            console.print(f"[bold]File size:[/] {len(content)} bytes")

        # Pattern to match <nei> tags with or without attributes
        # This handles both <nei>content</nei> and <nei attr="value">content</nei>
        pattern = r"(<nei(?:\s+[^>]*)?>)(.*?)</nei>"

        # Create a function that includes the verbose parameter
        def transform_with_verbose(match):
            return transform_nei_content(match, verbose)

        transformed_content = re.sub(
            pattern, transform_with_verbose, content, flags=re.DOTALL
        )

        # Count the number of replacements
        original_matches = re.findall(pattern, content, flags=re.DOTALL)
        transformed_matches = re.findall(pattern, transformed_content, flags=re.DOTALL)

        if verbose:
            console.print(
                f"Number of <nei> tags found: [bold cyan]{len(original_matches)}[/]"
            )
            console.print(
                f"Number of <nei> tags transformed: [bold green]{len(transformed_matches)}[/]"
            )

        if output_path:
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(transformed_content)
            console.print(
                f"Transformed content written to [bold green]{output_path}[/]"
            )
            console.print(
                f"Number of <nei> tags processed: [bold cyan]{len(original_matches)}[/]"
            )
        else:
            pass

        return True
    except Exception as e:
        # Use error_console for error messages
        error_console.print(f"[bold red]Error processing file: {e}[/]")
        return False


def transform(input_file: str, output_file: str | None = None, verbose: bool = False):
    """
    Transform content inside <nei> tags according to rules.

    Args:
        input_file: Path to the input XML file
        output_file: Path to the output file. If not provided, output to stdout
        verbose: Enable verbose output

    Returns:
        0 for success, 1 for failure
    """
    success = process_file(input_file, output_file, verbose)
    return 0 if success else 1


def main():
    """Entry point for the CLI."""
    return fire.Fire(transform)


if __name__ == "__main__":
    sys.exit(main())
