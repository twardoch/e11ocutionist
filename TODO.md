# TODO List for e11ocutionist

The old code that we are porting (for your reference) is in the `legacy_src/` folder. 

The new code is in the `src/` folder.

## 1. Critical: Fix Test Failures [PRIORITY: CRITICAL]

- [ ] **Fix `ImportError` in `tests/test_cli.py`**
    - **Reasoning:** Tests are failing to collect due to `ImportError: cannot import name 'process' from 'e11ocutionist.cli'`. This blocks all other test execution and reporting.
    - **Plan:**
        - Investigate `src/e11ocutionist/cli.py` to understand why the `process` function (or object) is not found or not exported correctly.
        - Check for typos in `tests/test_cli.py` or `src/e11ocutionist/cli.py`.
        - Ensure `src/e11ocutionist/cli.py` defines and correctly exports the `process` symbol.
        - Modify `__init__.py` files if necessary to ensure proper module exports.
        - Run `pytest tests/test_cli.py` locally to confirm the fix.

## 2. Testing and Documentation [PRIORITY: HIGH]

### 2.1. Implement Unit Tests

- [ ] **Improve test coverage (currently at 11% overall)**
    - **Reasoning:** Low test coverage makes the codebase fragile and hard to refactor safely.
    - **Plan:** Incrementally increase coverage for all modules, focusing on critical paths first.

  - [ ] **Implement unit tests for `cli.py` (currently at 9%)**
    - **Plan:**
        - After fixing the import error, identify main CLI commands and their functionalities.
        - Write tests for command parsing (using `click.testing.CliRunner`).
        - Test different command arguments and options (valid and invalid).
        - Mock underlying service calls (e.g., `E11ocutionistPipeline`) to isolate CLI logic.
        - Verify correct output and exit codes.

  - [ ] **Implement unit tests for `e11ocutionist.py` (pipeline orchestration, currently at 18%)**
    - **Plan:**
        - Test `PipelineConfig` creation with default and custom values.
        - Test pipeline step execution order.
        - Test handling of `start_step` and `force_restart`.
        - Test progress tracking and backup functionality.
        - Mock individual processing steps (`chunker`, `entitizer`, etc.) to test the pipeline's control flow.
        - Test error handling if a step fails.

  - [ ] **Implement unit tests for `entitizer.py` (currently at 9%)**
    - **Plan:**
        - Test NEI identification and tagging logic with various input texts.
        - Test dictionary building, updates, and merging processes.
        - Test XML input parsing and output generation.
        - Cover edge cases like empty inputs or malformed XML.
        - Mock LLM calls for entity recognition if applicable.

  - [ ] **Implement unit tests for `utils.py` (currently at 20%)**
    - **Plan:**
        - For each utility function:
            - Identify its purpose and expected inputs/outputs.
            - Write tests for typical use cases.
            - Write tests for edge cases (e.g., empty inputs, invalid types).
            - Ensure pure functions are tested for a variety of inputs.

  - [ ] **Implement unit tests for `chunker.py` (currently at 7%)**
    - **Plan:**
        - Test paragraph splitting and classification logic.
        - Test text processing utilities (token counting, hash generation).
        - Test ID generation and character escaping.
        - Test semantic analysis (mocking LLM calls).
        - Test chunk creation with various inputs (short text, long text, text with/without metadata).

  - [ ] **Implement unit tests for `orator.py` (currently at 6%)**
    - **Plan:**
        - Test item extraction from XML.
        - Test merging processed items back into XML.
        - Test punctuation, sentence, word, and emotion enhancement logic (mocking LLM calls).
        - Test the full `orator` processing pipeline with various inputs.

  - [ ] **Implement unit tests for `tonedown.py` (currently at 5%)**
    - **Plan:**
        - Test NEI extraction from documents.
        - Test language detection logic.
        - Test pronunciation cue review process (mocking LLM calls).
        - Test emphasis reduction logic.
        - Test NEI tag updates in the XML.
  
  - [ ] **Implement unit tests for `elevenlabs_converter.py` (currently at 9%)**
    - **Plan:**
      - Test text extraction from various XML structures.
      - Test dialog processing rules and transformations.
      - Test handling of different quote styles and character assignments.
      - Test generation of ElevenLabs-specific markup.
      - Cover complex dialog scenarios (nested quotes, multiple speakers).

  - [ ] **Implement unit tests for `elevenlabs_synthesizer.py` (currently at 29%)**
    - **Plan:**
      - Test filename sanitization with various inputs.
      - Test voice filtering and selection logic.
      - Mock ElevenLabs API calls to test text-to-speech synthesis logic.
      - Test handling of API errors and retries.
      - Test batch processing and file output.

  - [ ] **Implement unit tests for `neifix.py` (currently at 10%)**
    - **Plan:**
      - Test NEI content transformation rules.
      - Test input file reading and output file writing.
      - Test handling of edge cases (empty files, missing files, invalid XML).
      - Verify the integrity of the output XML structure.

- [ ] **Create test fixtures**
    - **Reasoning:** Reusable test data and mocks simplify test writing and maintenance.
    - **Plan:**
        - Create a `fixtures/` directory within `tests/`.
        - Add sample input XML files representing different scenarios (e.g., simple text, dialogs, various entities).
        - Develop mock LLM response generators or static mock response files.
        - Define standard configurations for test environment variables (e.g., API keys for mock servers if used).

- [ ] **Improve `orator.py` tests: Test error handling and recovery**
    - **Reasoning:** Robust error handling is crucial for pipeline stability.
    - **Plan:**
        - Identify potential failure points in `orator.py` (e.g., LLM errors, malformed XML).
        - Write tests that simulate these failures.
        - Verify that `orator.py` handles these errors gracefully (e.g., logs errors, retries, or skips problematic items).

- [ ] **Improve `elevenlabs_converter.py` tests: Test with more complex dialog scenarios**
    - **Reasoning:** Dialog processing is a key feature and needs thorough testing.
    - **Plan:**
        - Create input XML files with complex dialogs (e.g., nested quotes, rapid speaker changes, attributed vs. non-attributed speech).
        - Write tests to verify correct conversion of these scenarios.

- [ ] **Improve integration tests: Test error handling between modules**
    - **Reasoning:** Ensure that errors in one stage of the pipeline are handled correctly by subsequent stages or the main pipeline orchestrator.
    - **Plan:**
        - Design integration test scenarios where one module (e.g., `chunker`) is forced to produce an error or invalid output.
        - Verify that the pipeline (`e11ocutionist.py`) catches this error or that the next module in sequence handles the problematic input gracefully.

### 2.2. Improve Documentation

- [ ] **Update README.md**
    - **Reasoning:** The README is the entry point for new users and contributors.
    - **Plan:**
        - Review current README for accuracy and completeness.
        - Add a section on detailed configuration options for the pipeline and individual modules.
        - Create a troubleshooting section covering common errors (like the CLI import error) and their solutions.
        - Add a "Project Status" section including current test coverage and known major issues.

- [ ] **Add docstrings to all public functions and classes in `src/`**
    - **Reasoning:** Good docstrings improve code understanding and maintainability.
    - **Plan:**
        - Iterate through all `.py` files in `src/e11ocutionist/`.
        - For each public function, class, and method:
            - Ensure a clear, imperative docstring exists.
            - Describe the purpose, arguments (with types), and return values.
            - Add simple usage examples where helpful (e.g., for `utils.py` functions).
            - Follow PEP 257 conventions.

- [ ] **Document CLI commands comprehensively**
    - **Reasoning:** Users need clear guidance on how to use the CLI.
    - **Plan:**
        - Review `src/e11ocutionist/cli.py`.
        - Ensure all `fire` commands and their options have comprehensive help text.
        - Add examples for each command and common combinations of options.
        - Generate a `CLI_USAGE.md` file or section in README with this information.

### 2.3. Final Integration Tasks

- [ ] **Add comprehensive error handling in `src/`**
    - **Reasoning:** The application should be resilient to unexpected issues.
    - **Plan:**
        - Define custom exception classes (e.g., `ChunkerError`, `EntitizerError`, `LLMError`) in `e11ocutionist.exceptions`.
        - Review `src/` modules and replace generic `Exception` handling with specific custom exceptions.
        - Implement `try-except` blocks around external calls (LLMs, file I/O).
        - Add meaningful error messages that guide the user.
        - Consider recovery strategies for common failures (e.g., retries for LLM calls).

- [ ] **Add logging enhancements in `src/`**
    - **Reasoning:** Effective logging is essential for debugging and monitoring.
    - **Plan:**
        - Standardize on `loguru` for logging throughout `src/`.
        - Improve log messages for clarity, including context-specific information.
        - Implement structured logging where appropriate.
        - Add configurable log levels (DEBUG, INFO, WARNING, ERROR) via CLI options or config file.
        - Add progress indicators for long-running operations (e.g., processing a large document).

- [ ] **Create example workflows**
    - **Reasoning:** Examples help users understand how to use the system for common tasks.
    - **Plan:**
        - Identify 2-3 common use cases for `e11ocutionist`.
        - Create example scripts (e.g., shell scripts or Python scripts) that demonstrate these workflows.
        - Document these workflows in the README or a separate `EXAMPLES.md` file.
        - Include sample input files and describe the expected outputs.

## 3. Code Quality and Maintenance [PRIORITY: HIGH]

### 3.1. Fix Linting Issues in `src/` (New Code)

- [ ] **Update `pyproject.toml` ruff configuration**
    - **Reasoning:** Ensure linting rules are appropriate for the project.
    - **Plan:**
        - Review existing `pyproject.toml` for `ruff` settings.
        - Add specific rules to `lint.select` or `lint.ignore` to address common issues or noisy warnings not relevant to `src/` code.
        - Ensure `per-file-ignores` is used sparingly, primarily for specific legacy or generated code if any.

- [ ] **Address FBT001/FBT002 warnings (boolean arguments) in `src/`**
    - **Reasoning:** Boolean positional arguments can be confusing.
    - **Plan:**
        - Identify any new functions in `src/` that use boolean positional arguments.
        - Refactor them to use keyword-only arguments for boolean flags (`def func(*, flag: bool = False)`).
        - Update function calls accordingly.

- [ ] **Fix PLR0913 warnings (too many arguments) in `src/`**
    - **Reasoning:** Functions with too many arguments are hard to use and test.
    - **Plan:**
        - Identify any new functions in `src/` with more than 5-6 arguments.
        - Refactor by grouping related parameters into dataclasses or small objects.
        - Update function definitions and call sites.

- [ ] **Fix datetime issues (DTZ005 - naive datetimes) in `src/`**
    - **Reasoning:** Naive datetimes can lead to timezone-related bugs.
    - **Plan:**
        - Search for `datetime.now()` or `datetime.utcnow()` in `src/`.
        - Replace with timezone-aware datetimes, e.g., `datetime.now(timezone.utc)`.
        - Ensure all datetime objects are handled with explicit timezone information.

- [ ] **Refactor complex functions (C901, PLR0912, PLR0915) in `src/`**
    - **Reasoning:** Complex functions are difficult to understand, test, and maintain.
    - **Plan:**
        - Run `ruff` to identify functions in `src/` flagged for complexity, too many branches, or too many statements.
        - For each identified function:
            - Analyze its logic and responsibilities.
            - Break it down into smaller, more focused helper functions.
            - Aim for functions that do one thing well.

- [ ] **Address security issues (S320, S603, S324, B904 etc.) in `src/`**
    - **Reasoning:** Ensure new code is secure.
    - **Plan:**
        - **S320 (lxml):** If parsing external XML in `src/`, ensure `defusedxml.lxml` is used or parsing is done with appropriate security measures.
        - **S603 (subprocess):** If using `subprocess` in `src/`, ensure commands are not constructed from untrusted input, or use `shlex.quote`.
        - **S324 (insecure hash):** Avoid insecure hash functions like MD5/SHA1 in `src/` if cryptographic security is needed; prefer SHA256+.
        - **B904 (raise from):** In `src/`, when re-raising exceptions, use `raise NewException from original_exception` to preserve context.
        - Regularly review `ruff` security warnings for new code in `src/`.

### 3.2. Code Structure Improvements for `src/`

- [ ] **Implement consistent error handling (using custom exceptions)**
    - **Previously mentioned in 2.3, also a code quality item.**
    - **Plan:**
        - Finalize the design of custom exception classes in `e11ocutionist.exceptions`.
        - Systematically refactor `src/` modules to use these custom exceptions.
        - Ensure a clear hierarchy (e.g., `E11ocutionistError` as a base).

- [ ] **Enhance module structure and interfaces**
    - **Reasoning:** A well-defined structure improves clarity and reusability.
    - **Plan:**
        - Review the responsibilities of each module in `src/e11ocutionist/`.
        - Identify opportunities to organize related functions into logical classes.
        - Define clear public interfaces for each module, minimizing direct access to internal implementation details.
        - Look for code duplication and refactor into shared utility functions or classes.

- [ ] **Implement robust configuration management**
    - **Reasoning:** Centralized and validated configuration is easier to manage.
    - **Plan:**
        - Design a central configuration object/dataclass (e.g., `PipelineConfig` extension or a new one) to hold all pipeline and module settings.
        - Implement loading of configuration from a file (e.g., TOML or YAML) and/or environment variables, in addition to CLI arguments.
        - Add validation for configuration parameters (e.g., using Pydantic).
        - Provide sensible defaults and clear documentation for all configuration options.

### 3.3. Testing and CI/CD

- [ ] **Set up GitHub Actions workflow**
    - **Reasoning:** Automated testing and linting improve code quality and catch regressions early.
    - **Plan:**
        - Create a workflow file (e.g., `.github/workflows/ci.yml`).
        - **Linting & Formatting:** Add a job to run `ruff check` and `ruff format --check`.
        - **Testing:** Add a job to run `pytest` with coverage reporting.
        - **Coverage Upload:** Configure the workflow to upload coverage reports to a service like Codecov or Coveralls.
        - **Publishing (Optional, later):** Add a job to build and publish the package to PyPI on tagged releases.
        - Trigger the workflow on pushes to `main` and on pull requests.

## 4. Feature Enhancements [PRIORITY: LOW]

### 4.1. User Experience Improvements

- [ ] **Add detailed progress reporting for CLI**
    - **Reasoning:** Users need feedback on long-running operations.
    - **Plan:**
        - Integrate a library like `rich.progress` or `tqdm` into the CLI.
        - Display progress bars for document processing stages.
        - Show estimated time remaining if feasible.
        - Provide a summary of processing statistics upon completion.

- [ ] **Create more user-friendly error messages**
    - **Reasoning:** Clear error messages help users diagnose and fix problems.
    - **Plan:**
        - Review existing error messages in `src/`.
        - Enhance them to include more context and actionable advice.
        - For common configuration errors, suggest specific solutions or point to relevant documentation.


