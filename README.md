# e11ocutionist

**Transform your written content into captivating audio with `e11ocutionist`, a powerful multi-stage document processing pipeline designed to prepare literary text for high-quality speech synthesis, especially optimized for services like ElevenLabs.**

`e11ocutionist` intelligently refines your documents through a series of steps, ensuring that the final audio output is natural, engaging, and accurately reflects the nuances of your text.

## Who is this for?

*   **Authors & Publishers:** Bring your books, articles, and stories to life as audiobooks or narrated content with enhanced clarity and consistent character voices.
*   **Content Creators:** Convert blog posts, scripts, or educational materials into polished audio for podcasts, videos, or accessibility purposes.
*   **Developers:** Integrate sophisticated text pre-processing into your applications that leverage text-to-speech technology.

## Why is `e11ocutionist` useful?

*   **Superior Narration Quality:** Goes beyond simple text-to-speech by semantically understanding and restructuring content for a more human-like narration.
*   **Consistent Pronunciation:** Identifies and allows for consistent pronunciation of Named Entities of Interest (NEIs) – such as character names, locations, or specific terms – throughout your document.
*   **Optimized for Speech Engines:** The pipeline specifically prepares text to get the best results from advanced speech synthesis services like ElevenLabs.
*   **Modular & Flexible:** Run the entire pipeline with a single command or execute individual processing steps for fine-grained control.
*   **Progress Tracking & Resumption:** Long documents? No problem. `e11ocutionist` can save its progress and resume from where it left off.
*   **LLM-Powered Intelligence:** Leverages Large Language Models (LLMs) for sophisticated tasks like semantic chunking, entity recognition, and narrative enhancement.

## Installation

Get started with `e11ocutionist` by installing it via pip:

```bash
pip install e11ocutionist
```

Ensure you have Python 3.10 or newer. You will also need API access to an LLM provider configured for `litellm` (e.g., by setting environment variables like `OPENAI_API_KEY`). For direct speech synthesis, an ElevenLabs API key (`ELEVENLABS_API_KEY`) is required.

## How to Use `e11ocutionist`

### Command-Line Interface (CLI)

The easiest way to process your document is using the `process` command:

```bash
e11ocutionist process "path/to/your_document.txt" --output-dir "path/to/output_folder" --verbose
```

This command will run your input document through the entire pipeline, placing intermediate and final files in the specified output directory. The `--verbose` flag provides detailed logging of the process.

**Key CLI Operations:**

*   **Full Pipeline:** `e11ocutionist process <input_file> [options]`
*   **Individual Steps:** You can run specific steps like `chunk`, `entitize`, `orate`, `tonedown`, and `convert-11labs`. For example:
    ```bash
    e11ocutionist chunk "input.txt" "output_chunked.xml"
    ```
    Use `e11ocutionist <command> --help` for detailed options for each step.
*   **Direct Speech Synthesis:** Once your text is processed (or if you have a ready-to-synthesize text file), use the `say` command:
    ```bash
    e11ocutionist say --input-file "path/to/output_folder/your_document_step5_11labs.txt" --output-dir "path/to/audio_files"
    ```
    This requires your `ELEVENLABS_API_KEY` environment variable to be set.

### Programmatic Usage (Python)

Integrate `e11ocutionist` into your Python projects for more customized workflows:

```python
from pathlib import Path
from e11ocutionist import E11ocutionistPipeline, PipelineConfig, ProcessingStep

# 1. Configure the pipeline
config = PipelineConfig(
    input_file=Path("path/to/your_document.txt"),
    output_dir=Path("path/to/output_folder"),
    verbose=True,
    # You can customize models, temperatures, and other parameters here
    # For example, to start from the 'orating' step:
    # start_step=ProcessingStep.ORATING,
    # chunker_model="gpt-4o-mini", # Example: use a different model
)

# 2. Create and run the pipeline
pipeline = E11ocutionistPipeline(config)
try:
    final_output_path = pipeline.run()
    print(f"Pipeline completed. Final output: {final_output_path}")
except Exception as e:
    print(f"An error occurred: {e}")

```

This script initializes the pipeline with your input file and desired output directory, then runs all processing stages. The `final_output_path` will point to the text file ready for speech synthesis.

## Technical Deep Dive

This section provides a more detailed look into the inner workings of `e11ocutionist` and guidelines for contributors.

### How `e11ocutionist` Works

The `e11ocutionist` tool processes documents through a sequential pipeline, where each stage refines the text for optimal speech synthesis. The core orchestrator is the `E11ocutionistPipeline` class, defined in `src/e11ocutionist/e11ocutionist.py`. This class manages the execution of processing steps, handles progress tracking (via a `progress.json` file in the output directory), and allows for resumption of interrupted pipelines.

The pipeline consists of the following `ProcessingStep`s (enum values):

1.  **`CHUNKING` (`chunker.py`)**
    *   **Purpose:** Splits the input document (plain text or Markdown) into smaller, semantically coherent chunks. This is crucial for maintaining context during subsequent LLM processing and for managing API limits.
    *   **Input:** Raw text file (e.g., `.txt`, `.md`).
    *   **Output:** An XML file where the document is structured into `<chunk>` elements.
    *   **Mechanism:** Uses an LLM (configurable, e.g., GPT-4) to identify natural breakpoints in the text, aiming for chunks that represent complete thoughts or scenes.

2.  **`ENTITIZING` (`entitizer.py`)**
    *   **Purpose:** Identifies Named Entities of Interest (NEIs) within the text. These are typically character names, locations, or specific terms that require consistent pronunciation or emphasis.
    *   **Input:** The XML file produced by the `CHUNKING` step.
    *   **Output:** An XML file where NEIs within chunks are tagged (e.g., `<NEI type="PERSON">Entity Name</NEI>`).
    *   **Mechanism:** Employs an LLM to perform entity recognition based on context. It also generates pronunciation guidance or alternative phrasing for these entities, stored within the NEI tags.

3.  **`ORATING` (`orator.py`)**
    *   **Purpose:** Enhances the text for a more natural and engaging spoken narrative. This involves sentence restructuring, word normalization (e.g., converting numbers to words), and adding stylistic elements like emphasis or emotional cues.
    *   **Input:** The XML file from the `ENTITIZING` step.
    *   **Output:** An XML file with further refinements to the text content within chunks, potentially including SSML-like tags for emphasis (e.g., `<emphasis level="strong">word</emphasis>`) or emotion.
    *   **Mechanism:** Uses an LLM to "rewrite" text for speech. It can perform several sub-steps:
        *   `--sentences`: Restructures sentences for better flow.
        *   `--words`: Normalizes words (e.g., abbreviations, numbers).
        *   `--punctuation`: Adjusts punctuation for speech pauses.
        *   `--emotions`: Infers and suggests emotional delivery (if supported by the target TTS).
        *   `--all_steps` (default): Performs all available orating transformations.

4.  **`TONING_DOWN` (`tonedown.py`)**
    *   **Purpose:** Reviews and refines the NEI pronunciation cues and oratorical enhancements. This step aims to reduce excessive or unnatural-sounding emphasis and ensure that NEI treatments are consistent and contextually appropriate.
    *   **Input:** The XML file from the `ORATING` step.
    *   **Output:** A refined XML file, with moderated emphasis and NEI tags.
    *   **Mechanism:** Utilizes an LLM to analyze the density and appropriateness of previously added markup, adjusting it to improve overall naturalness. The `min_em_distance` parameter helps control the proximity of emphasized elements.

5.  **`ELEVENLABS_CONVERSION` (`elevenlabs_converter.py`)**
    *   **Purpose:** Converts the processed XML document into a plain text format specifically tailored for the ElevenLabs TTS API. It handles the extraction of dialogue, narration, and applies any special formatting suitable for ElevenLabs.
    *   **Input:** The XML file from the `TONING_DOWN` step.
    *   **Output:** A plain text file (`.txt`) ready for synthesis.
    *   **Mechanism:** Parses the XML, extracts relevant text content, and formats it according to ElevenLabs best practices. It can operate in:
        *   `--dialog_mode` (default): Optimizes for text containing dialogue.
        *   `--plaintext_mode`: Produces a simpler text output.

**Command-Line Interface (`cli.py`)**

The CLI is built using the `python-fire` library, which automatically generates command-line interfaces from Python functions. Each processing step and the main pipeline `process` function in `cli.py` are exposed as subcommands. Helper functions for input validation, logging configuration (`loguru`), and file system operations support the CLI.

**Key Dependencies:**

*   `litellm`: For interacting with various LLM APIs (OpenAI, Anthropic, etc.) in a standardized way.
*   `elevenlabs`: The official Python client for the ElevenLabs API, used by the `say` command.
*   `lxml`: For robust and efficient XML parsing and manipulation.
*   `loguru`: For flexible and powerful logging.
*   `python-fire`: For generating the CLI.
*   `hatch`: For project management, dependency control, and running development tasks (see below).

### Coding and Contribution Guidelines

We welcome contributions to `e11ocutionist`! Please follow these guidelines:

**Project Management with Hatch**

This project uses [Hatch](https://hatch.pypa.io/latest/) for managing environments, dependencies, and running common development tasks. Refer to `pyproject.toml` for the full project configuration.

1.  **Install Hatch:**
    ```bash
    pip install hatch
    ```

2.  **Activate Development Environment:**
    Navigate to the project root and run:
    ```bash
    hatch shell
    ```
    This will create or activate a virtual environment with all necessary dependencies installed.

**Development Tasks (run via Hatch):**

*   **Run Tests:**
    ```bash
    hatch run test
    ```
    For test coverage reports:
    ```bash
    hatch run test-cov
    ```
*   **Linting (Ruff):**
    Check for code style issues:
    ```bash
    hatch run lint
    ```
*   **Formatting (Ruff):**
    Automatically format your code:
    ```bash
    hatch run format
    ```
*   **Auto-fixes (Ruff):**
    Apply available automatic fixes for linting errors:
    ```bash
    hatch run fix
    ```
*   **Type Checking (Mypy):**
    Perform static type analysis:
    ```bash
    hatch run typecheck
    ```

**Coding Standards:**

*   **Style:** Code is formatted using Ruff. Please run `hatch run format` before committing.
*   **Linting:** Ruff is also used for linting. Ensure `hatch run lint` passes.
*   **Type Hints:** All new code should include Python type hints, and `hatch run typecheck` must pass.
*   **Tests:** Contributions should include unit tests for new functionality or bug fixes. Place tests in the `tests/` directory.
*   **Commit Messages:** Follow conventional commit message formats if possible (e.g., `feat: Add new feature`, `fix: Correct a bug`).

**Requirements:**

*   Python 3.10+
*   Access to an LLM API (e.g., OpenAI, Anthropic) configured for `litellm`. This usually involves setting environment variables like `OPENAI_API_KEY`.
*   (Optional) An ElevenLabs API key (set as `ELEVENLABS_API_KEY` environment variable) for using the `e11ocutionist say` command for direct speech synthesis.

**License:**

`e11ocutionist` is licensed under the MIT License. See the `LICENSE` file for details.

**Submitting Changes:**

1.  Fork the repository on GitHub.
2.  Create a new branch for your feature or bug fix.
3.  Make your changes, adhering to the coding standards and adding tests.
4.  Ensure all checks (linting, type checking, tests) pass.
5.  Push your branch to your fork and open a pull request against the main `e11ocutionist` repository.

We look forward to your contributions!
