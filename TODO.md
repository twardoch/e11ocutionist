# TODO List for e11ocutionist

The old code that we are porting (for your reference) is in the `legacy_src/` folder. 

The new code is in the `src/` folder.

## 1. Core Processing Modules [COMPLETED]

- [x] Implement `chunker.py` Module (from `legacy_src/malmo_chunker.py`)
- [x] Implement `entitizer.py` Module (from `legacy_src/malmo_entitizer.py`)
- [x] Implement `orator.py` Module (from `legacy_src/malmo_orator.py`)
- [x] Implement `tonedown.py` Module (from `legacy_src/malmo_tonedown.py`)
- [x] Implement utilities and helper modules
- [x] Implement CLI interface

## 2. Testing and Documentation [PRIORITY: HIGH]

### 2.1. Implement Unit Tests

- [ ] Improve test coverage (currently at 55%, up from 38%)
  - [x] Implement unit tests for `tonedown.py` (coverage increased to 38%)
  - [x] Implement unit tests for `orator.py` (coverage increased to 70%)
  - [x] Implement unit tests for `chunker.py` (coverage increased to 52%)
  - [x] Implement unit tests for `elevenlabs_converter.py` (coverage increased to 69%)
  - [x] Implement unit tests for `elevenlabs_synthesizer.py` (coverage increased to 88%)
  - [x] Implement unit tests for `neifix.py` (coverage increased to 95%)
  - [ ] Implement unit tests for remaining modules:
    - [ ] `cli.py` (currently at 9%)
    - [ ] `e11ocutionist.py` (currently at 18%)
    - [ ] `entitizer.py` (currently at 29%)
    - [ ] `utils.py` (currently at 42%)

- [ ] Create test fixtures
  - [ ] Add sample XML files for testing
  - [ ] Create mock LLM responses
  - [ ] Set up test environment variables

- [x] Add tests for `entitizer.py`
  - [x] Test NEI identification and tagging
  - [x] Test dictionary building and updates
  - [x] Test XML merging

- [x] Add tests for `chunker.py`
  - [x] Test paragraph splitting and classification
  - [x] Test text processing utilities (token counting, hash generation)
  - [x] Test ID generation and character escaping
  - [x] Test semantic analysis
  - [x] Test chunk creation with various inputs

- [x] Add tests for `orator.py`
  - [x] Test item extraction from XML
  - [x] Test merging processed items back into XML
  - [x] Test punctuation enhancement
  - [x] Test full pipeline with various inputs
  - [ ] Test error handling and recovery

- [x] Add tests for `tonedown.py`
  - [x] Test NEI extraction from documents
  - [x] Test language detection (partially implemented)
  - [x] Test pronunciation cue review (with mock data)
  - [x] Test emphasis reduction
  - [x] Test NEI tag updates

- [x] Add tests for `elevenlabs_converter.py`
  - [x] Test text extraction from XML
  - [x] Test dialog processing
  - [x] Test document processing with sample files
  - [ ] Test with more complex dialog scenarios

- [x] Add tests for `elevenlabs_synthesizer.py`
  - [x] Test filename sanitization
  - [x] Test voice filtering and selection
  - [x] Test text-to-speech synthesis (with mocks)
  - [x] Test API integration (with mocks)
  - [x] Test error handling for missing API keys

- [x] Add tests for `neifix.py`
  - [x] Test NEI content transformation
  - [x] Test input/output file handling
  - [x] Test edge cases (empty files, missing output)

- [x] Add tests for integration between modules
  - [x] Test passing results between pipeline stages
  - [x] Test full pipeline execution
  - [ ] Test error handling between modules

### 2.2. Improve Documentation

- [ ] Update README.md
  - [x] Add basic installation instructions
  - [x] Add usage examples for CLI
  - [x] Add basic programmatic usage examples
  - [ ] Add more detailed configuration options
  - [ ] Add troubleshooting section

- [ ] Add docstrings to all public functions
  - [ ] Check and improve existing docstrings
  - [ ] Include parameter descriptions
  - [ ] Document return values
  - [ ] Add examples where appropriate

- [ ] Document CLI commands
  - [ ] Create comprehensive help text
  - [ ] Add examples for each command
  - [ ] Document all options and parameters

### 2.3. Final Integration Tasks

- [ ] Add comprehensive error handling
  - [ ] Implement proper exception handling throughout
  - [ ] Add meaningful error messages
  - [ ] Add recovery strategies for common failures

- [ ] Add logging enhancements
  - [ ] Improve log messages for clarity
  - [ ] Add log levels for different verbosity needs
  - [ ] Add progress indicators for long-running operations

- [ ] Create example workflows
  - [ ] Add example scripts for common tasks
  - [ ] Document typical processing workflows
  - [ ] Include sample inputs and expected outputs

## 3. Code Quality and Maintenance [PRIORITY: HIGH]

### 3.1. Fix Linting Issues

- [x] Update pyproject.toml configuration
  - [x] Move 'per-file-ignores' to 'lint.per-file-ignores' section
  - [x] Update target Python version to 3.12
  - [ ] Add specific rules to address common issues

- [ ] Address FBT001/FBT002 warnings (many throughout the codebase)
  - [ ] Replace boolean positional arguments with keyword-only arguments in `entitizer.py`
  - [ ] Replace boolean positional arguments in `malmo_chunker.py`
  - [ ] Replace boolean positional arguments in `malmo_orator.py`
  - [ ] Add type hints for boolean parameters
  - [ ] Update function calls to use keyword arguments

- [ ] Fix PLR0913 warnings (too many arguments)
  - [ ] Refactor `malmo_all.py` functions with too many parameters
  - [ ] Refactor `malmo_chunker.py` functions with too many parameters
  - [ ] Refactor `malmo_orator.py` functions with too many parameters
  - [ ] Create parameter classes/dataclasses where appropriate
  - [ ] Use keyword arguments consistently

- [ ] Fix datetime issues (DTZ005)
  - [ ] Replace datetime.now() with datetime.now(timezone.utc) in `malmo_all.py`
  - [ ] Replace datetime.now() in `malmo_chunker.py`
  - [ ] Replace datetime.now() in `malmo_orator.py`
  - [ ] Add proper timezone awareness throughout

- [ ] Refactor complex functions (C901, PLR0912, PLR0915)
  - [ ] Break down complex functions in `malmo_chunker.py`:
    - [ ] `semantic_analysis` (complexity: 27)
    - [ ] `extract_and_parse_llm_response` (complexity: 13)
    - [ ] `add_unit_tags` (complexity: 14)
    - [ ] `create_chunks` (complexity: 11)
  - [ ] Break down complex functions in `malmo_all.py`:
    - [ ] `run_step` (complexity: 13)
    - [ ] `process_file` (complexity: 41)
  - [ ] Break down complex functions in `malmo_orator.py`:
    - [ ] `run_llm_processing` (complexity: 11)

- [ ] Address security issues
  - [ ] Fix lxml security warnings (S320) in:
    - [ ] `malmo_11labs.py`
    - [ ] `malmo_orator.py`
    - [ ] `test_entitizer.py`
    - [ ] `test_elevenlabs_converter.py`
  - [ ] Implement proper input sanitization
  - [ ] Address subprocess security concerns (S603) in `malmo_all.py`
  - [ ] Fix SHA1 usage (S324) in `malmo_chunker.py`

### 3.2. Code Structure Improvements

- [ ] Implement consistent error handling
  - [ ] Create custom exception classes
  - [ ] Add exception hierarchy for different error types
  - [ ] Fix B904 issues in `malmo_11labs.py` (raise with from)

- [ ] Enhance module structure
  - [ ] Organize related functions into logical classes
  - [ ] Improve code reuse across modules
  - [ ] Create clear interfaces between components

- [ ] Implement configuration management
  - [ ] Create central configuration object
  - [ ] Add validation for configuration parameters
  - [ ] Provide sensible defaults and documentation

### 3.3. Testing and CI/CD

- [ ] Increase test coverage
  - [x] Add tests for tonedown.py module (now at 38%)
  - [x] Add tests for orator.py module (now at 70%)
  - [x] Add tests for chunker.py module (now at 52%)
  - [x] Add tests for elevenlabs_converter.py module (now at 69%)
  - [x] Add tests for elevenlabs_synthesizer.py module (now at 88%)
  - [x] Add tests for neifix.py module (now at 95%)
  - [ ] Add tests for cli.py module (currently at 9%)
  - [ ] Add tests for e11ocutionist.py module (currently at 18%)
  - [ ] Add tests for entitizer.py module (currently at 29%)
  - [ ] Add tests for utils.py module (currently at 42%)

- [ ] Set up GitHub Actions workflow
  - [ ] Add workflow for running tests
  - [ ] Add workflow for linting and type checking
  - [ ] Add workflow for publishing package

## 4. Feature Enhancements [PRIORITY: LOW]

### 4.1. Performance Optimizations

- [ ] Improve LLM usage efficiency
  - [ ] Optimize prompts for token efficiency
  - [ ] Implement caching for LLM responses
  - [ ] Add batch processing where possible

- [ ] Implement parallel processing
  - [ ] Process chunks in parallel
  - [ ] Add concurrency controls
  - [ ] Ensure thread safety

### 4.2. User Experience Improvements

- [ ] Add progress reporting
  - [ ] Implement CLI progress bars
  - [ ] Add estimated time remaining
  - [ ] Show processing statistics

- [ ] Create user-friendly error messages
  - [ ] Add detailed diagnostics
  - [ ] Suggest solutions for common errors
  - [ ] Provide clear guidance for configuration issues

### 4.3. Additional Features

- [ ] Add voice customization options
  - [ ] Support for voice cloning
  - [ ] Implement voice style controls
  - [ ] Add support for additional TTS providers

- [ ] Implement document format conversions
  - [ ] Add support for Markdown
  - [ ] Add support for HTML
  - [ ] Add support for ePub

- [ ] Create visualization tools
  - [ ] Add speech synthesis preview
  - [ ] Implement processing visualization
  - [ ] Create interactive pipeline control
