
# TODO.md

## 1. Description of Tools in `legacy_src`

The `legacy_src` directory contains several Python scripts that form a pipeline for processing text documents for text-to-speech (TTS) applications.

### 1.1. Core Tools

#### 1.1.1. Pipeline Orchestration
- **malmo_all.py**: Main pipeline orchestrator that sequentially runs the entire workflow with robust error handling, progress tracking, and resumption capability.

#### 1.1.2. Processing Steps
1. **malmo_chunker.py**: Splits input document into semantic chunks based on content boundaries.
2. **malmo_entitizer.py**: Identifies and tags Named Entities of Interest (NEIs) for consistent pronunciation.
3. **malmo_orator.py**: Enhances text for spoken narrative through sentence restructuring, word normalization, punctuation enhancement, and emotional emphasis.
4. **malmo_tonedown.py**: Reviews and refines NEI pronunciation cues and reduces excessive emphasis.
5. **malmo_11labs.py**: Converts processed XML to ElevenLabs-compatible text format.

#### 1.1.3. Other Tools
- **say_11labs.py**: Synthesizes text using personal ElevenLabs voices and saves as MP3 files.
- **malmo_neifix.py**: Utility script for transforming NEI tag content with specific rules.
- **sapko.sh**: A shell script demonstrating the pipeline with a specific input file.

### 1.2. Main Workflow: `malmo_all.py` and `sapko.sh`

The primary workflow involves a sequence of Python scripts orchestrated by `malmo_all.py`, followed by steps executed by `sapko.sh`.

#### 1.2.1. `malmo_all.py`
* **Purpose**: This script acts as the main orchestrator for a multi-step document processing pipeline. [cite: 78, 118, 159] It sequentially runs several other Python scripts (`malmo_chunker.py`, `malmo_entitizer.py`, `malmo_orator.py`, `malmo_tonedown.py`, and `malmo_11labs.py`) to process an input file. [cite: 130, 135, 137, 138, 139]
* **Functionality**:
    * **Dependency Management**: Ensures necessary Python package dependencies for each sub-script are installed, attempting installation via `uv` or `pip` if missing. [cite: 80, 82, 83]
    * **Output Directory Management**: Creates a timestamped output directory for each input file to store intermediate and final results, as well as a progress file. [cite: 86, 101]
    * **Progress Tracking & Resumption**: Saves and loads processing progress in a JSON file (`progress.json`), allowing the workflow to be resumed from a specific step. [cite: 93, 94, 120, 149] It supports starting from a specified step and force restarting. [cite: 118, 120]
    * **Step Execution**: Runs each script in the pipeline as a subprocess, with retries on failure. [cite: 87, 106, 108]
    * **Configuration**: Accepts various command-line arguments to configure models, temperatures, and other parameters for each step. [cite: 78, 117, 158]
    * **Backup**: Optionally creates backup copies of intermediate files at each step. [cite: 118, 135]
    * **Summary Generation**: Creates a summary file (`_summary.txt`) at the end of processing, detailing the configuration and outcome of each step. [cite: 150]
* **Pipeline Steps & Associated Scripts**:
    1.  **Chunking (`malmo_chunker.py`)**: Splits the input document into semantic chunks. [cite: 130]
    2.  **Entitizing (`malmo_entitizer.py`)**: Identifies and tags named entities of interest (NEIs). [cite: 135]
    3.  **Orating (`malmo_orator.py`)**: Enhances the text for spoken narrative (sentence restructuring, word normalization, punctuation, emotional emphasis). [cite: 137]
    4.  **Toning Down (`malmo_tonedown.py`)**: Reviews and revises pronunciation cues for NEIs and reduces excessive emphasis. [cite: 138]
    5.  **11Labs Preparation (`malmo_11labs.py`)**: Converts the processed XML to a text format suitable for ElevenLabs. [cite: 139, 66]
* **Tech Stack**: Python, `fire` (for CLI), `loguru` (logging), `python-dotenv` (environment variables), `tenacity` (retries), `tqdm`/`rich` (progress bars), `litellm` (LLM interaction), `elevenlabs` (text-to-speech). [cite: 78]

#### 1.2.2. `malmo_chunker.py`
* **Purpose**: Splits a markdown document into semantically chunked XML. [cite: 165, 298] It aims to divide the text into meaningful units like chapters, scenes, or thematic sections before further processing.
* **Functionality**:
    * **Itemization**: Splits the input text into paragraphs and classifies them (heading, blockquote, list, code block, etc.) to create `<item>` elements with unique IDs. [cite: 174, 179, 182]
    * **Token Counting**: Uses `tiktoken` to count tokens for items, units, and chunks, storing this in `tok` attributes. [cite: 169, 193]
    * **Semantic Analysis (LLM-based)**: Optionally uses an LLM (e.g., GPT-4, Gemini) to identify semantic boundaries (chapters, scenes, units) based on the itemized text. [cite: 199, 200, 205] It requires a minimum number of semantic units based on document size. [cite: 202, 208]
    * **Unit Tagging**: Adds `<unit>` tags around sequences of items based on the semantic analysis or heuristics. [cite: 234, 236]
    * **Chunk Creation**: Groups `<unit>` elements into `<chunk>` elements, ensuring each chunk does not exceed a specified maximum token size. [cite: 253, 262] Chapters usually start new chunks. [cite: 261] Handles units larger than the maximum chunk size by splitting them. [cite: 272, 283]
    * **XML Output**: Produces an XML document with a hierarchical structure: `<doc>` -> `<chunk>` -> `<unit>` -> `<item>`.
    * **Backup**: Optionally creates timestamped backups at various processing stages (itemized, semantic_boundaries, units_added, chunks_created, final). [cite: 298, 308]
* **Tech Stack**: Python, `fire`, `litellm`, `tiktoken`, `lxml`, `backoff`, `python-dotenv`, `loguru`. [cite: 165]

#### 1.2.3. `malmo_entitizer.py`
* **Purpose**: Identifies and tags Named Entities of Interest (NEIs) in an XML document, preparing them for consistent pronunciation in text-to-speech. [cite: 314, 336, 376]
* **Functionality**:
    * **XML Parsing**: Parses the input XML (typically the output of `malmo_chunker.py`). [cite: 317]
    * **Chunk Processing**: Processes the document chunk by chunk. [cite: 319, 368]
    * **NEI Identification (LLM-based)**: For each chunk, it sends the text to an LLM with a prompt to identify NEIs. [cite: 336] The prompt instructs the LLM to:
        * Determine the predominant language. [cite: 341]
        * Identify NEIs relevant to the text's domain (people, locations, organizations, abbreviations, etc.). [cite: 342]
        * Tag each occurrence of an NEI with `<nei>`. [cite: 343]
        * For NEIs not consisting of common words in the predominant language, use `<nei orig="ORIGINAL">PRONUNCIATION</nei>`, where `PRONUNCIATION` is an approximation in the predominant language. [cite: 345, 346]
        * Mark newly found NEIs (not in the provided dictionary) with `new="true"`. [cite: 350]
    * **NEI Dictionary**: Maintains a dictionary of identified NEIs and their pronunciations, updating it as new NEIs are found in subsequent chunks. [cite: 336, 365, 370]
    * **XML Merging**: Merges the LLM's output (tagged items) back into the original XML structure of the chunk. [cite: 331, 371]
    * **Incremental Saves & Backups**: Optionally saves the state of the XML document and the NEI dictionary after processing each chunk and creates timestamped backups. [cite: 321, 375]
    * **Benchmark Mode**: Can process the input file with multiple LLM models in parallel for comparison, saving individual outputs and a summary. [cite: 401, 418, 419]
* **Tech Stack**: Python, `fire`, `litellm`, `lxml`, `backoff`, `python-dotenv`, `loguru`. [cite: 314]

#### 1.2.4. `malmo_orator.py`
* **Purpose**: Enhances XML text for spoken narrative by applying several LLM-based and algorithmic transformations. [cite: 437, 511]
* **Functionality**:
    * **Modular Processing**: Applies transformations chunk by chunk. [cite: 515]
    * **Sentence Restructuring (LLM-based)**: Divides long sentences into shorter ones and ensures paragraphs/headings end with punctuation, without changing words or order. [cite: 471, 473, 474, 475]
    * **Word Normalization (LLM-based)**: Converts digits to numeric words, replaces symbols like '/' and '&' with appropriate conjunctions, and spells out other rare symbols. [cite: 482, 484, 485, 486]
    * **Punctuation Enhancement (Algorithmic)**: Surrounds parenthetical expressions with spaces and ellipses, and replaces em/en dashes (used as pauses) with ellipses. [cite: 494, 496]
    * **Emotional Emphasis (LLM-based)**: Adds `<em>` tags to emphasize words/phrases, inserts `<hr/>` for dramatic pauses, and adds exclamation marks for energy, while trying to avoid excessive changes. [cite: 499, 501, 502]
    * **Selective Application**: Allows specific steps (`--sentences`, `--words`, `--punctuation`, `--emotions`) or all steps (`--all_steps`) to be run. [cite: 510, 513]
    * **XML Merging**: Similar to `malmo_entitizer.py`, it extracts item elements from LLM responses and merges them back. [cite: 444, 446]
    * **Incremental Saves & Backups**: Optionally saves the document state after each processing step for each chunk and creates timestamped backups. [cite: 451, 511]
* **Tech Stack**: Python, `fire`, `litellm`, `lxml`, `backoff`, `python-dotenv`, `loguru`. [cite: 437]

#### 1.2.5. `malmo_tonedown.py`
* **Purpose**: Reviews and revises pronunciation cues for Named Entities of Interest (NEIs) in an XML document and reduces excessive text emphasis. [cite: 527, 581]
* **Functionality**:
    * **NEI Dictionary Building**: Parses the input XML and builds a dictionary of NEIs from `<nei>` tags, noting their original spelling (`orig` attribute) and existing pronunciation cues (tag content). [cite: 534, 535] Sets `new="true"` if an `orig` value is encountered for the first time. [cite: 538]
    * **Language Detection (LLM-based)**: Extracts a middle fragment of the text and uses an LLM to detect the predominant language. [cite: 539, 543, 544]
    * **Pronunciation Cue Review (LLM-based)**: Sends the NEI dictionary and detected language to an LLM. The LLM is prompted to review each pronunciation cue and decide whether to:
        * Keep the cue (if the NEI is unusual/foreign or has non-intuitive pronunciation). [cite: 551]
        * Replace the cue with the original entity spelling (if it's native, familiar, a long English phrase, or the cue is misleading). [cite: 552]
        * Create a new, less complex adapted cue. [cite: 553]
        The LLM returns a revised JSON dictionary of NEIs. [cite: 554]
    * **XML Update**: Updates the content of `<nei>` tags in the XML document based on the revised NEI dictionary. [cite: 566, 568]
    * **Emphasis Reduction (Algorithmic)**: If the `--em <min_distance>` argument is provided, it removes `<em>` tags that are too close together (closer than `min_distance` words). [cite: 573, 577, 581]
    * **XML Serialization & Validation**: Carefully serializes the modified XML, trying to preserve formatting and validate the output. Includes repair attempts for corrupted XML. [cite: 591, 598, 603]
* **Tech Stack**: Python, `fire`, `lxml`, `loguru`, `backoff`, `python-dotenv`, `litellm`, `rich`. [cite: 527]

#### 1.2.6. `malmo_11labs.py`
* **Purpose**: Converts an XML file (typically the output of the Malmo pipeline) into a text format compatible with ElevenLabs text-to-speech, with options for processing dialog. [cite: 31, 66]
* **Functionality**:
    * **XML Parsing**: Parses the input XML file. [cite: 32]
    * **Chunk and Item Processing**: Iterates through `<chunk>` and `<item>` elements. [cite: 37, 38, 59, 63]
    * **Text Extraction and Transformation**:
        * Extracts inner XML of `<item>` elements. [cite: 55]
        * Converts `<em>...</em>` to smart double quotes (`“...”`). [cite: 56]
        * Converts `<nei new="true">...</nei>` to smart double quotes. [cite: 56]
        * Converts remaining `<nei>...</nei>` to their plain content. [cite: 57]
        * Replaces `<hr/>` with `<break time="0.5s" />` (escaped). [cite: 57]
        * Strips all other remaining HTML/XML tags. [cite: 57]
        * Unescapes HTML entities (e.g., `&lt;` to `<`). [cite: 41, 58]
        * Normalizes consecutive newlines (3+ to 2). [cite: 41]
        * Normalizes consecutive opening/closing quotation marks to single smart quotes. [cite: 39, 40, 41]
    * **Dialog Processing (Optional, via `--dialog` flag)**:
        * Identifies dialog lines starting with "— " (em dash + space). [cite: 45, 48]
        * Wraps these lines in `<q>...</q>` tags. [cite: 48]
        * Handles dialog toggles (" — " within a line) by alternating `</q><q>`. [cite: 46, 49, 50, 52]
    * **Final Postprocessing**:
        * Replaces `<q>` at the start of a line with an opening smart quote. [cite: 42]
        * Replaces `</q>` at the end of a line with a closing smart quote. [cite: 42]
        * Replaces other `<q>` instances with "; OPEN_Q" and `</q>` with "CLOSE_Q;". [cite: 43, 44]
        * Replaces `<break time=.../>` with "...". [cite: 44]
        * Replaces single newlines with double newlines. [cite: 44]
    * **Plaintext Mode (Optional, via `--plaintext` flag)**: Processes the input as plain text instead of XML, applying dialog processing and final postprocessing if enabled. [cite: 66, 68]
* **Tech Stack**: Python, `fire`, `lxml`, `loguru`, `rich`, `litellm`. [cite: 31]

#### 1.2.7. `sapko.sh`
* **Purpose**: This is a shell script designed to run a specific sequence of the Malmo tools (`malmo_orator.py`, `malmo_tonedown.py`, `malmo_11labs.py`) on a predefined set of input/output filenames related to "Andrzej Sapkowski - Wiedźmin Geralt z Rivii 0.1 - Rozdroże kruków". [cite: 614]
* **Functionality**:
    1.  Runs `malmo_orator.py` on `..._step2_entited.xml` to produce `..._step3_orated.xml`, with all steps, verbose logging, and backups enabled. [cite: 614]
    2.  Runs `malmo_tonedown.py` on `..._step3_orated.xml` to produce `..._step4_toneddown.xml`, with verbose logging. [cite: 614]
    3.  Runs `malmo_11labs.py` on `..._step4_toneddown.xml` to produce `..._step5_11.txt`, with verbose logging. [cite: 614]
* **Note**: The script uses relative paths (`../`) suggesting it's intended to be run from within the `legacy_src` directory itself, referencing scripts in a parent directory (which seems incorrect given the provided file structure where scripts are within `legacy_src`). This might be a path issue in the script. The filenames are hardcoded.
* **Tech Stack**: Bash.

### 1.3. Separate Tool: `say_11labs.py`

#### 1.3.1. `say_11labs.py`
* **Purpose**: A command-line tool to synthesize text using all personal (cloned, generated, professional) ElevenLabs voices and save the output as MP3 files. [cite: 615, 616]
* **Functionality**:
    * **ElevenLabs Client Initialization**: Initializes the ElevenLabs client using an API key (from argument or `ELEVENLABS_API_KEY` environment variable). [cite: 617, 618, 620]
    * **Voice Retrieval**: Fetches all voices belonging to the user, specifically filtering for "cloned", "generated", and "professional" categories. [cite: 622, 624]
    * **Text Synthesis**: For each retrieved voice, it synthesizes the provided input text. [cite: 628, 631]
        * Uses a specified `model_id` (default: `eleven_multilingual_v2`) and `output_format` (default: `mp3_44100_128`). [cite: 631, 634]
        * Includes retry logic for API errors during synthesis and voice retrieval. [cite: 621, 628]
    * **Output**: Saves each synthesized audio as an MP3 file in a specified output folder (default: `output_audio`). [cite: 631, 633, 636]
        * Filenames are sanitized and structured as `{voice_id}--{sanitized_voice_name}.mp3`. [cite: 626, 638]
    * **Progress & Logging**: Uses `rich` for progress bars during synthesis and `loguru` for logging. [cite: 615, 619, 637]
* **Tech Stack**: Python, `elevenlabs` SDK, `fire` (CLI), `python-dotenv`, `loguru`, `tenacity`, `rich`. [cite: 615]

### 1.4. Utility Script: `malmo_neifix.py`

#### 1.4.1. `malmo_neifix.py`
* **Purpose**: A script to parse XML files and transform the text content inside `<nei>` tags according to specific capitalization and hyphenation rules. [cite: 423]
* **Functionality**:
    * **XML Parsing**: Reads an input XML file. [cite: 429]
    * **NEI Tag Targeting**: Uses regex to find all `<nei ...>` tags and their content. [cite: 430]
    * **Content Transformation**: For the text content within each `<nei>` tag:
        * Splits the content into words.
        * For each word:
            * Preserves single-letter words (likely parts of acronyms). [cite: 426]
            * Removes hyphens. [cite: 427]
            * Keeps the first letter of the (hyphen-less) word as is.
            * Converts all subsequent letters in the word to lowercase. [cite: 427]
        * Joins the transformed words back with spaces. [cite: 428]
    * **Output**: Writes the modified XML content to an output file or to stdout if no output file is specified. [cite: 433, 434]
* **Tech Stack**: Python, `fire`, `rich`. [cite: 423]

