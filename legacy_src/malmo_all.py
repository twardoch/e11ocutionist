#!/usr/bin/env -S uv run -s
# /// script
# dependencies = ["fire", "loguru", "python-dotenv", "tenacity", "tqdm", "rich", "litellm<=1.67.2", "elevenlabs"]
# ///
# this_file: malmo_all.py

import fire
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any
from loguru import logger
import datetime
from dotenv import load_dotenv
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)
from rich.progress import (
    Progress,
    TextColumn,
    BarColumn,
    TimeElapsedColumn,
    SpinnerColumn,
)
from rich.console import Console
from rich.panel import Panel

# Load environment variables from .env file
load_dotenv()

# Constants
MAX_RETRIES = 3
RETRY_WAIT_MIN = 1  # seconds
RETRY_WAIT_MAX = 30  # seconds
PROGRESS_FILE_NAME = "progress.json"

# Default values from each tool
CHUNKER_DEFAULT_MODEL = "openrouter/openai/gpt-4.1"
CHUNKER_DEFAULT_CHUNK_SIZE = 12288
CHUNKER_DEFAULT_TEMPERATURE = 1.0

ENTITIZER_DEFAULT_MODEL = "openrouter/google/gemini-2.5-pro-preview-03-25"
ENTITIZER_DEFAULT_TEMPERATURE = 1.0

ORATOR_DEFAULT_MODEL = "openrouter/google/gemini-2.5-pro-preview-03-25"
ORATOR_DEFAULT_TEMPERATURE = 1.0

# Add default values for tonedown and 11labs steps
TONEDOWN_DEFAULT_MODEL = "openrouter/google/gemini-2.5-pro-preview-03-25"
TONEDOWN_DEFAULT_TEMPERATURE = 1.0

# Rich console for pretty output
console = Console()

# Required dependencies for each script
CHUNKER_DEPENDENCIES = [
    "backoff",
    "fire",
    "loguru",
    "python-dotenv",
    "openai",
    "tiktoken",
    "litellm",
]
ENTITIZER_DEPENDENCIES = [
    "fire",
    "loguru",
    "python-dotenv",
    "openai",
    "lxml",
    "litellm",
]
ORATOR_DEPENDENCIES = ["fire", "loguru", "python-dotenv", "openai", "lxml", "litellm"]

# Required dependencies for new steps
TONEDOWN_DEPENDENCIES = ["fire", "loguru", "python-dotenv", "openai", "lxml", "litellm"]
ELEVENLABS_DEPENDENCIES = ["fire", "loguru", "python-dotenv", "lxml"]


def ensure_dependencies_installed(dependencies: list[str]) -> bool:
    """
    Ensure that all required dependencies are installed.

    Args:
        dependencies: List of dependency package names

    Returns:
        True if all dependencies are installed or successfully installed
    """
    missing_deps = []

    # Check which dependencies are missing
    for dep in dependencies:
        try:
            __import__(
                dep.replace("-", "_")
            )  # Replace hyphen with underscore for import
        except ImportError:
            missing_deps.append(dep)

    if not missing_deps:
        return True

    # Install missing dependencies
    if missing_deps:
        deps_str = " ".join(missing_deps)
        console.print(f"[yellow]Installing missing dependencies: {deps_str}[/yellow]")

        try:
            # Try using uv
            uv_cmd = ["uv", "pip", "install", *missing_deps]
            subprocess.run(uv_cmd, check=True, capture_output=True, text=True)
            console.print("[green]Successfully installed dependencies using uv[/green]")
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            try:
                # Fall back to pip if uv is not available
                pip_cmd = [sys.executable, "-m", "pip", "install", *missing_deps]
                subprocess.run(pip_cmd, check=True, capture_output=True, text=True)
                console.print(
                    "[green]Successfully installed dependencies using pip[/green]"
                )
                return True
            except subprocess.CalledProcessError as e:
                console.print(
                    "[bold red]Failed to install dependencies. Please install manually:[/bold red]"
                )
                console.print(f"[bold yellow]pip install {deps_str}[/bold yellow]")
                logger.error(f"Failed to install dependencies: {e}")
                return False

    return True


def create_output_directory(input_file: str) -> Path:
    """
    Create an output directory based on the input file name.

    Args:
        input_file: Path to the input file

    Returns:
        Path object for the output directory
    """
    input_path = Path(input_file)
    # Create directory with input filename (without extension) + timestamp
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    output_dir = input_path.parent / f"{input_path.stem}_malmo_{timestamp}"
    output_dir.mkdir(exist_ok=True)
    return output_dir


@retry(
    stop=stop_after_attempt(MAX_RETRIES),
    wait=wait_exponential(multiplier=1, min=RETRY_WAIT_MIN, max=RETRY_WAIT_MAX),
    retry=retry_if_exception_type(subprocess.CalledProcessError),
    before_sleep=lambda retry_state: logger.warning(
        f"Retrying command (attempt {retry_state.attempt_number}/{MAX_RETRIES}) after {retry_state.outcome.exception() if retry_state.outcome else 'error'}"
    ),
)
def run_command(cmd: list[str], verbose: bool = False) -> dict:
    """
    Run a command as a subprocess with retries on failure.

    Args:
        cmd: Command to run, as a list of strings
        verbose: Whether to print verbose output

    Returns:
        Dict with stdout, stderr, and return code
    """
    # Ensure all parts of the command are properly encoded strings
    sanitized_cmd = []
    for part in cmd:
        if part is not None:
            sanitized_cmd.append(str(part))

    cmd_str = " ".join(
        [
            f'"{c}"'
            if " " in c or any(special in c for special in "()[]{}*?!$&|<>;'\"")
            else c
            for c in sanitized_cmd
        ]
    )
    logger.debug(f"Running command: {cmd_str}")

    try:
        # Use shell=False for better security and handling of arguments with special characters
        result = subprocess.run(
            sanitized_cmd, check=True, capture_output=True, text=True, encoding="utf-8"
        )

        if verbose and result.stdout:
            logger.debug(result.stdout)

        if result.stderr:
            logger.warning(result.stderr)

        return {
            "stdout": result.stdout,
            "stderr": result.stderr,
            "returncode": result.returncode,
        }
    except UnicodeEncodeError as ue:
        # Handle Unicode encoding errors explicitly
        logger.error(f"Unicode encoding error: {ue}")
        logger.debug(
            "Command contained non-ASCII characters that couldn't be encoded properly."
        )
        raise


def save_progress(output_dir: Path, progress_state: dict[str, Any]) -> None:
    """
    Save the current progress state to a file.

    Args:
        output_dir: Directory to save the progress file in
        progress_state: Current progress state
    """
    progress_file = output_dir / PROGRESS_FILE_NAME
    with open(progress_file, "w") as f:
        json.dump(progress_state, f, indent=2, default=str)
    logger.debug(f"Progress saved to {progress_file}")


def load_progress(output_dir: Path) -> dict[str, Any]:
    """
    Load progress state from a file.

    Args:
        output_dir: Directory containing the progress file

    Returns:
        Progress state as a dict, or empty dict if file not found
    """
    progress_file = output_dir / PROGRESS_FILE_NAME
    if not progress_file.exists():
        return {}

    try:
        with open(progress_file) as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        logger.warning(f"Could not load progress file: {e}")
        return {}


def step_completed(progress_state: dict[str, Any], step: int) -> bool:
    """
    Check if a step has been completed successfully.

    Args:
        progress_state: Current progress state
        step: Step number to check

    Returns:
        True if the step is completed, False otherwise
    """
    return progress_state.get(f"step{step}_completed", False)


def initialize_progress(
    input_file: str,
    output_dir: Path,
    step1_output: Path,
    step2_output: Path,
    step3_output: Path,
    step4_output: Path,
    step5_output: Path,
    chunker_model: str,
    chunker_temperature: float,
    chunker_chunk_size: int,
    entitizer_model: str,
    entitizer_temperature: float,
    orator_model: str,
    orator_temperature: float,
    tonedown_model: str,
    tonedown_temperature: float,
) -> dict[str, Any]:
    """
    Initialize or load the progress state.

    Args:
        input_file: Path to the input file
        output_dir: Output directory
        step1_output: Path to step 1 output file
        step2_output: Path to step 2 output file
        step3_output: Path to step 3 output file
        step4_output: Path to step 4 output file (tonedown)
        step5_output: Path to step 5 output file (11labs)
        chunker_model: Model to use for chunker
        chunker_temperature: Temperature for chunker
        chunker_chunk_size: Chunk size for chunker
        entitizer_model: Model to use for entitizer
        entitizer_temperature: Temperature for entitizer
        orator_model: Model to use for orator
        orator_temperature: Temperature for orator
        tonedown_model: Model to use for tonedown
        tonedown_temperature: Temperature for tonedown

    Returns:
        Progress state as a dict
    """
    # Try to load existing progress
    progress_state = load_progress(output_dir)

    # If no existing progress or configuration has changed, init new progress
    if not progress_state or progress_state.get("input_file") != str(input_file):
        progress_state = {
            "input_file": str(input_file),
            "start_time": datetime.datetime.now().isoformat(),
            "output_dir": str(output_dir),
            "step1_output": str(step1_output),
            "step2_output": str(step2_output),
            "step3_output": str(step3_output),
            "step4_output": str(step4_output),
            "step5_output": str(step5_output),
            "chunker_model": chunker_model,
            "chunker_temperature": chunker_temperature,
            "chunker_chunk_size": chunker_chunk_size,
            "entitizer_model": entitizer_model,
            "entitizer_temperature": entitizer_temperature,
            "orator_model": orator_model,
            "orator_temperature": orator_temperature,
            "tonedown_model": tonedown_model,
            "tonedown_temperature": tonedown_temperature,
            "step1_attempts": 0,
            "step2_attempts": 0,
            "step3_attempts": 0,
            "step4_attempts": 0,
            "step5_attempts": 0,
            "step1_completed": False,
            "step2_completed": False,
            "step3_completed": False,
            "step4_completed": False,
            "step5_completed": False,
            "step1_error": None,
            "step2_error": None,
            "step3_error": None,
            "step4_error": None,
            "step5_error": None,
        }
        save_progress(output_dir, progress_state)

    return progress_state


def run_step(
    step: int,
    cmd: list[str],
    output_file: Path,
    progress_state: dict[str, Any],
    output_dir: Path,
    verbose: bool = False,
) -> tuple[bool, dict[str, Any]]:
    """
    Run a processing step with proper error handling and progress tracking.

    Args:
        step: Step number (1, 2, or 3)
        cmd: Command to run
        output_file: Expected output file
        progress_state: Current progress state
        output_dir: Directory to save progress file
        verbose: Whether to print verbose output

    Returns:
        Tuple of (success, updated_progress_state)
    """
    step_key = f"step{step}"
    attempts_key = f"{step_key}_attempts"
    completed_key = f"{step_key}_completed"
    error_key = f"{step_key}_error"

    # If step is already completed and output file exists, skip
    if progress_state.get(completed_key, False) and output_file.exists():
        logger.info(f"Step {step} already completed, skipping")
        return True, progress_state

    # Increment attempt counter
    progress_state[attempts_key] = progress_state.get(attempts_key, 0) + 1
    save_progress(output_dir, progress_state)

    # Run the command with retries
    try:
        run_command(cmd, verbose)

        # Verify output file exists
        if not output_file.exists():
            msg = f"Output file {output_file} was not created"
            raise FileNotFoundError(msg)

        # Update progress
        progress_state[completed_key] = True
        progress_state[error_key] = None
        progress_state[f"{step_key}_completed_time"] = (
            datetime.datetime.now().isoformat()
        )
        save_progress(output_dir, progress_state)

        logger.success(f"Step {step} completed. Output saved to {output_file}")
        return True, progress_state

    except Exception as e:
        # Capture detailed error information
        error_detail = str(e)

        # For subprocess errors, try to get more details
        if isinstance(e, subprocess.CalledProcessError):
            error_detail = f"Command failed with exit code {e.returncode}:\n"
            if e.stderr:
                error_detail += f"STDERR: {e.stderr}\n"
            if e.stdout:
                error_detail += f"STDOUT: {e.stdout}\n"

        # For tenacity RetryError, unwrap to get the real exception
        if "RetryError" in str(e):
            try:
                # Look for CalledProcessError in the exception message
                (str(e).replace("RetryError[<Future at ", "").split(" state=")[0])
                logger.error(
                    f"Command failed after {MAX_RETRIES} retries. Run with --verbose for more details."
                )

                # Just extract any useful information from the error string
                if "CalledProcessError" in str(e):
                    # Try to get the exit code
                    if "exit code" in str(e):
                        exit_code = str(e).split("exit code")[1].split()[0].strip()
                        error_detail += f"Command failed with exit code {exit_code}\n"

                # Additional debugging info
                if verbose:
                    logger.debug(f"Full error: {e!s}")
                    logger.debug(f"Command was: {' '.join(cmd)}")
            except (AttributeError, IndexError) as parse_error:
                # Couldn't parse the error message
                logger.debug(f"Failed to parse error details: {parse_error}")

        # Update progress with error
        progress_state[completed_key] = False
        progress_state[error_key] = error_detail
        save_progress(output_dir, progress_state)

        logger.error(f"Error in Step {step}: {e}")
        if verbose:
            logger.error(f"Detailed error information:\n{error_detail}")
        return False, progress_state


def process_file(
    input_file: str,
    output_dir: Path | None = None,
    chunker_model: str = CHUNKER_DEFAULT_MODEL,
    chunker_temperature: float = CHUNKER_DEFAULT_TEMPERATURE,
    chunker_chunk_size: int = CHUNKER_DEFAULT_CHUNK_SIZE,
    entitizer_model: str = ENTITIZER_DEFAULT_MODEL,
    entitizer_temperature: float = ENTITIZER_DEFAULT_TEMPERATURE,
    orator_model: str = ORATOR_DEFAULT_MODEL,
    orator_temperature: float = ORATOR_DEFAULT_TEMPERATURE,
    tonedown_model: str = TONEDOWN_DEFAULT_MODEL,
    tonedown_temperature: float = TONEDOWN_DEFAULT_TEMPERATURE,
    verbose: bool = True,
    backup: bool = True,
    start_from_step: int = 1,
    force_restart: bool = False,
) -> None:
    """
    Process a file through all malmo tools with robust error handling and progress tracking.

    Args:
        input_file: Path to the input file
        output_dir: Output directory (created automatically if not provided)
        chunker_model: Model to use for chunker
        chunker_temperature: Temperature for chunker
        chunker_chunk_size: Chunk size for chunker
        entitizer_model: Model to use for entitizer
        entitizer_temperature: Temperature for entitizer
        orator_model: Model to use for orator
        orator_temperature: Temperature for orator
        tonedown_model: Model to use for tonedown
        tonedown_temperature: Temperature for tonedown
        verbose: Enable verbose logging
        backup: Create backups at each step
        start_from_step: Start processing from this step (1-5)
        force_restart: Ignore previous progress and start from scratch
    """
    # Setup output directory if not provided
    if output_dir is None:
        output_dir = create_output_directory(input_file)
    else:
        output_dir.mkdir(exist_ok=True)

    console.print(
        Panel.fit(
            f"[bold green]Processing:[/bold green] [yellow]{input_file}[/yellow]\n"
            f"[bold green]Output directory:[/bold green] [yellow]{output_dir}[/yellow]",
            title="Malmo All-in-One Processor",
            border_style="blue",
        )
    )

    # Check dependencies based on what steps will be run
    all_dependencies = set()
    if start_from_step <= 1:
        all_dependencies.update(CHUNKER_DEPENDENCIES)
    if start_from_step <= 2:
        all_dependencies.update(ENTITIZER_DEPENDENCIES)
    if start_from_step <= 3:
        all_dependencies.update(ORATOR_DEPENDENCIES)
    if start_from_step <= 4:
        all_dependencies.update(TONEDOWN_DEPENDENCIES)
    if start_from_step <= 5:
        all_dependencies.update(ELEVENLABS_DEPENDENCIES)

    if not ensure_dependencies_installed(list(all_dependencies)):
        console.print(
            "[bold red]Critical dependencies are missing. Cannot continue.[/bold red]"
        )
        return

    # Normalize input file path to handle special characters
    input_path = Path(input_file).resolve()

    # Directly check file existence and readability
    if not input_path.exists():
        error_msg = f"Input file does not exist: {input_path}"
        logger.error(error_msg)
        console.print(f"[bold red]{error_msg}[/bold red]")
        if output_dir:
            progress_state = {"step1_error": error_msg}
            save_progress(output_dir, progress_state)
        return

    if not input_path.is_file():
        error_msg = f"Input path is not a file: {input_path}"
        logger.error(error_msg)
        console.print(f"[bold red]{error_msg}[/bold red]")
        if output_dir:
            progress_state = {"step1_error": error_msg}
            save_progress(output_dir, progress_state)
        return

    # Try to read the file to verify it's accessible
    try:
        with open(input_path, encoding="utf-8") as f:
            # Just read a bit to make sure it's accessible
            sample = f.read(1024)
            logger.debug(
                f"Successfully read sample from {input_path} ({len(sample)} bytes)"
            )
    except (OSError, UnicodeDecodeError) as e:
        error_msg = f"Error reading input file: {e}"
        logger.error(error_msg)
        console.print(f"[bold red]{error_msg}[/bold red]")
        if output_dir:
            progress_state = {"step1_error": error_msg}
            save_progress(output_dir, progress_state)
        return

    # Define file paths for each step
    step1_output = output_dir / f"{input_path.stem}_step1_chunked.xml"
    step2_output = output_dir / f"{input_path.stem}_step2_entited.xml"
    step3_output = output_dir / f"{input_path.stem}_step3_orated.xml"
    step4_output = output_dir / f"{input_path.stem}_step4_toneddown.xml"
    step5_output = output_dir / f"{input_path.stem}_step5_11labs.txt"

    # Get current directory for relative paths to python scripts
    current_dir = Path(__file__).parent.absolute()

    # Initialize/load progress tracking
    if force_restart:
        # Remove existing progress file if forcing restart
        progress_file = output_dir / PROGRESS_FILE_NAME
        if progress_file.exists():
            progress_file.unlink()

    progress_state = initialize_progress(
        input_file=str(input_path),
        output_dir=output_dir,
        step1_output=step1_output,
        step2_output=step2_output,
        step3_output=step3_output,
        step4_output=step4_output,
        step5_output=step5_output,
        chunker_model=chunker_model,
        chunker_temperature=chunker_temperature,
        chunker_chunk_size=chunker_chunk_size,
        entitizer_model=entitizer_model,
        entitizer_temperature=entitizer_temperature,
        orator_model=orator_model,
        orator_temperature=orator_temperature,
        tonedown_model=tonedown_model,
        tonedown_temperature=tonedown_temperature,
    )

    # Create the steps with their respective commands
    steps = []

    # Step 1: Run malmo_chunker
    chunker_script = current_dir / "malmo_chunker.py"

    # Check if the chunker script exists
    if not chunker_script.exists():
        error_msg = f"Chunker script not found: {chunker_script}"
        logger.error(error_msg)
        console.print(f"[bold red]{error_msg}[/bold red]")
        progress_state["step1_error"] = error_msg
        save_progress(output_dir, progress_state)
        return

    chunker_cmd = [
        sys.executable,
        str(chunker_script),
        input_file,
        str(step1_output),
        "--chunk",
        str(chunker_chunk_size),
        "--model",
        chunker_model,
        "--temperature",
        str(chunker_temperature),
    ]

    # Try to run malmo_chunker with --help to check if it's working correctly
    if start_from_step == 1:
        try:
            logger.info("Testing if the chunker script can be executed correctly...")
            help_cmd = [sys.executable, str(chunker_script), "--help"]
            result = subprocess.run(
                help_cmd, capture_output=True, text=True, check=False
            )
            if result.returncode != 0:
                logger.warning(f"Chunker help command failed: {result.stderr}")
                # If the help command fails, check if the script is executable
                if not os.access(chunker_script, os.X_OK):
                    logger.warning(
                        "The chunker script is not executable. Attempting to fix permissions."
                    )
                    os.chmod(chunker_script, 0o755)  # Make executable
        except Exception as e:
            logger.warning(f"Error testing chunker script: {e}")

    if verbose:
        chunker_cmd.append("--verbose")
    if backup:
        chunker_cmd.append("--backup")
    steps.append((1, chunker_cmd, step1_output, "Running chunker"))

    # Step 2: Run malmo_entitizer
    entitizer_script = current_dir / "malmo_entitizer.py"
    entitizer_cmd = [
        sys.executable,
        str(entitizer_script),
        str(step1_output),
        str(step2_output),
        "--model",
        entitizer_model,
        "--temperature",
        str(entitizer_temperature),
    ]
    if verbose:
        entitizer_cmd.append("--verbose")
    if backup:
        entitizer_cmd.append("--backup")
    steps.append((2, entitizer_cmd, step2_output, "Running entitizer"))

    # Step 3: Run malmo_orator
    orator_script = current_dir / "malmo_orator.py"
    orator_cmd = [
        sys.executable,
        str(orator_script),
        str(step2_output),
        str(step3_output),
        "--model",
        orator_model,
        "--temperature",
        str(orator_temperature),
        "--all_steps",  # Apply all enhancement steps
    ]
    if verbose:
        orator_cmd.append("--verbose")
    if backup:
        orator_cmd.append("--backup")
    steps.append((3, orator_cmd, step3_output, "Running orator"))

    # Step 4: Run malmo_tonedown
    tonedown_script = current_dir / "malmo_tonedown.py"
    tonedown_cmd = [
        sys.executable,
        str(tonedown_script),
        str(step3_output),
        str(step4_output),
        "--model",
        tonedown_model,
        "--temperature",
        str(tonedown_temperature),
    ]
    if verbose:
        tonedown_cmd.append("--verbose")
    if backup:
        tonedown_cmd.append("--backup")
    steps.append((4, tonedown_cmd, step4_output, "Running tonedown"))

    # Step 5: Run malmo_11labs
    elevenlabs_script = current_dir / "malmo_11labs.py"
    elevenlabs_cmd = [
        sys.executable,
        str(elevenlabs_script),
        str(step4_output),
        str(step5_output),
    ]
    if verbose:
        elevenlabs_cmd.append("--verbose")
    steps.append((5, elevenlabs_cmd, step5_output, "Running 11labs"))

    # Process all steps in sequence with progress tracking
    with Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        TextColumn("[bold green]{task.completed}/{task.total}"),
        TimeElapsedColumn(),
    ) as progress:
        overall_task = progress.add_task("[yellow]Overall progress", total=len(steps))

        for step_num, cmd, output_file, description in steps:
            # Skip steps before start_from_step
            if step_num < start_from_step:
                if step_completed(progress_state, step_num):
                    progress.advance(overall_task)
                continue

            # Check for dependency (previous step must be completed)
            if step_num > 1 and not step_completed(progress_state, step_num - 1):
                error_msg = f"Cannot run step {step_num} because step {step_num - 1} has not been completed"
                progress_state[f"step{step_num}_error"] = error_msg
                logger.error(error_msg)
                save_progress(output_dir, progress_state)
                break

            step_task = progress.add_task(
                f"[cyan]Step {step_num}: {description}", total=1
            )
            logger.info(f"Step {step_num}: {description}")

            success, progress_state = run_step(
                step=step_num,
                cmd=cmd,
                output_file=output_file,
                progress_state=progress_state,
                output_dir=output_dir,
                verbose=verbose,
            )

            progress.update(step_task, completed=1)
            progress.advance(overall_task)

            if not success:
                break

    # Check if all steps completed successfully
    all_completed = all(
        step_completed(progress_state, i + 1) for i in range(len(steps))
    )

    if all_completed:
        logger.success(f"Processing complete! Final output: {step5_output}")
        console.print(
            Panel.fit(
                f"[bold green]Final output:[/bold green] [yellow]{step5_output}[/yellow]",
                title="Processing Complete",
                border_style="green",
            )
        )
    else:
        logger.warning("Processing did not complete successfully.")
        failed_step = next(
            (
                i + 1
                for i in range(len(steps))
                if not step_completed(progress_state, i + 1)
            ),
            None,
        )
        error_msg = progress_state.get(f"step{failed_step}_error", "Unknown error")

        console.print(
            Panel.fit(
                f"[bold red]Processing stopped at step {failed_step}[/bold red]\n"
                f"[bold yellow]Error:[/bold yellow] {error_msg}\n\n"
                f"To resume from this step, run again with:\n"
                f"--start_from_step={failed_step}",
                title="Processing Incomplete",
                border_style="red",
            )
        )

    # Create a summary file
    summary_file = output_dir / f"{input_path.stem}_summary.txt"
    with open(summary_file, "w") as f:
        f.write("Malmo processing summary\n")
        f.write("======================\n\n")
        f.write(f"Input file: {input_file}\n")

        # Add start time
        start_time = datetime.datetime.fromisoformat(
            progress_state.get("start_time", datetime.datetime.now().isoformat())
        )
        f.write(f"Start time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}\n")

        # Add end time
        end_time = datetime.datetime.now()
        f.write(f"End time: {end_time.strftime('%Y-%m-%d %H:%M:%S')}\n")

        # Calculate total duration
        duration = end_time - start_time
        hours, remainder = divmod(duration.total_seconds(), 3600)
        minutes, seconds = divmod(remainder, 60)
        f.write(f"Total duration: {int(hours)}h {int(minutes)}m {int(seconds)}s\n\n")

        # Step details
        f.write("Step 1 (Chunker):\n")
        f.write(f"  - Output: {step1_output}\n")
        f.write(f"  - Model: {chunker_model}\n")
        f.write(f"  - Temperature: {chunker_temperature}\n")
        f.write(f"  - Chunk size: {chunker_chunk_size}\n")
        f.write(f"  - Completed: {progress_state.get('step1_completed', False)}\n")
        f.write(f"  - Attempts: {progress_state.get('step1_attempts', 0)}\n")
        if progress_state.get("step1_error"):
            f.write(f"  - Error: {progress_state.get('step1_error')}\n")
        f.write("\n")

        f.write("Step 2 (Entitizer):\n")
        f.write(f"  - Output: {step2_output}\n")
        f.write(f"  - Model: {entitizer_model}\n")
        f.write(f"  - Temperature: {entitizer_temperature}\n")
        f.write(f"  - Completed: {progress_state.get('step2_completed', False)}\n")
        f.write(f"  - Attempts: {progress_state.get('step2_attempts', 0)}\n")
        if progress_state.get("step2_error"):
            f.write(f"  - Error: {progress_state.get('step2_error')}\n")
        f.write("\n")

        f.write("Step 3 (Orator):\n")
        f.write(f"  - Output: {step3_output}\n")
        f.write(f"  - Model: {orator_model}\n")
        f.write(f"  - Temperature: {orator_temperature}\n")
        f.write(f"  - Completed: {progress_state.get('step3_completed', False)}\n")
        f.write(f"  - Attempts: {progress_state.get('step3_attempts', 0)}\n")
        if progress_state.get("step3_error"):
            f.write(f"  - Error: {progress_state.get('step3_error')}\n")
        f.write("\n")

        f.write("Step 4 (Tonedown):\n")
        f.write(f"  - Output: {step4_output}\n")
        f.write(f"  - Model: {tonedown_model}\n")
        f.write(f"  - Temperature: {tonedown_temperature}\n")
        f.write(f"  - Completed: {progress_state.get('step4_completed', False)}\n")
        f.write(f"  - Attempts: {progress_state.get('step4_attempts', 0)}\n")
        if progress_state.get("step4_error"):
            f.write(f"  - Error: {progress_state.get('step4_error')}\n")
        f.write("\n")

        f.write("Step 5 (11labs):\n")
        f.write(f"  - Output: {step5_output}\n")
        f.write(f"  - Completed: {progress_state.get('step5_completed', False)}\n")
        f.write(f"  - Attempts: {progress_state.get('step5_attempts', 0)}\n")
        if progress_state.get("step5_error"):
            f.write(f"  - Error: {progress_state.get('step5_error')}\n")


def main(
    input_file: str,
    output_dir: str | None = None,
    chunker_model: str = CHUNKER_DEFAULT_MODEL,
    chunker_temperature: float = CHUNKER_DEFAULT_TEMPERATURE,
    chunker_chunk_size: int = CHUNKER_DEFAULT_CHUNK_SIZE,
    entitizer_model: str = ENTITIZER_DEFAULT_MODEL,
    entitizer_temperature: float = ENTITIZER_DEFAULT_TEMPERATURE,
    orator_model: str = ORATOR_DEFAULT_MODEL,
    orator_temperature: float = ORATOR_DEFAULT_TEMPERATURE,
    tonedown_model: str = TONEDOWN_DEFAULT_MODEL,
    tonedown_temperature: float = TONEDOWN_DEFAULT_TEMPERATURE,
    verbose: bool = True,
    backup: bool = True,
    start_from_step: int = 1,
    force_restart: bool = False,
) -> None:
    """
    Run all five malmo tools (chunker, entitizer, orator, tonedown, 11labs) in sequence on a file.

    Args:
        input_file: Path to the input file
        output_dir: Output directory (created automatically if not provided)
        chunker_model: Model to use for chunker
        chunker_temperature: Temperature for chunker
        chunker_chunk_size: Chunk size for chunker
        entitizer_model: Model to use for entitizer
        entitizer_temperature: Temperature for entitizer
        orator_model: Model to use for orator
        orator_temperature: Temperature for orator
        tonedown_model: Model to use for tonedown
        tonedown_temperature: Temperature for tonedown
        verbose: Enable verbose logging
        backup: Create backups at each step
        start_from_step: Start processing from this step (1-5)
        force_restart: Ignore previous progress and start from scratch
    """
    if verbose:
        logger.level("DEBUG")

    # Validate inputs
    if start_from_step < 1 or start_from_step > 5:
        logger.error("start_from_step must be between 1 and 5")
        sys.exit(1)

    output_dir_path = Path(output_dir) if output_dir else None

    try:
        process_file(
            input_file=input_file,
            output_dir=output_dir_path,
            chunker_model=chunker_model,
            chunker_temperature=chunker_temperature,
            chunker_chunk_size=chunker_chunk_size,
            entitizer_model=entitizer_model,
            entitizer_temperature=entitizer_temperature,
            orator_model=orator_model,
            orator_temperature=orator_temperature,
            tonedown_model=tonedown_model,
            tonedown_temperature=tonedown_temperature,
            verbose=verbose,
            backup=backup,
            start_from_step=start_from_step,
            force_restart=force_restart,
        )
    except KeyboardInterrupt:
        logger.warning("Process interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unhandled error: {e}")
        if verbose:
            logger.exception("Detailed error information:")
        sys.exit(1)


if __name__ == "__main__":
    fire.Fire(main)
