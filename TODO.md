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

- [ ] Improve test coverage (currently at 38%, up from 15%)
  - [x] Implement unit tests for `tonedown.py` (coverage increased from 5% to 38%)
  - [x] Implement unit tests for `orator.py` (coverage increased from 6% to 24%)
  - [x] Implement unit tests for `chunker.py` (coverage increased from 7% to 16%)
  - [x] Implement unit tests for `elevenlabs_converter.py` (coverage increased to 48%)
  - [x] Implement unit tests for `elevenlabs_synthesizer.py` (coverage increased from 31% to 89%)
  - [x] Implement unit tests for `neifix.py` (coverage increased from 10% to 95%)
  - [ ] Implement unit tests for remaining modules

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
  - [ ] Test semantic analysis
  - [ ] Test chunk creation with various inputs

- [x] Add tests for `orator.py`
  - [x] Test item extraction from XML
  - [x] Test merging processed items back into XML
  - [x] Test punctuation enhancement
  - [ ] Test full pipeline with various inputs
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

- [ ] Add tests for integration between modules
  - [ ] Test passing results between pipeline stages
  - [ ] Test full pipeline execution
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

## 3. Code Quality and Maintenance [PRIORITY: MEDIUM]

### 3.1. Fix Linting Issues

- [x] Update pyproject.toml configuration
  - [x] Move 'per-file-ignores' to 'lint.per-file-ignores' section
  - [x] Update target Python version to 3.12
  - [ ] Add specific rules to address common issues

- [ ] Address FBT001/FBT002 warnings (many throughout the codebase)
  - [ ] Replace boolean positional arguments with keyword-only arguments
  - [ ] Update function signatures and calls throughout the codebase
  - [ ] Add type hints for boolean parameters

- [ ] Fix PLR0913 warnings (too many arguments)
  - [ ] Refactor functions with too many parameters
  - [ ] Create parameter classes/dataclasses where appropriate
  - [ ] Use keyword arguments consistently

- [ ] Fix datetime issues (DTZ005)
  - [ ] Replace datetime.now() with datetime.now(timezone.utc)
  - [ ] Update date/time handling for consistency
  - [ ] Add proper timezone awareness

- [ ] Refactor complex functions (C901, PLR0912, PLR0915)
  - [ ] Break down functions with high complexity
  - [ ] Reduce branch complexity
  - [ ] Reduce statement count

- [ ] Address security issues
  - [ ] Fix lxml security warnings (S320)
  - [ ] Implement proper input sanitization
  - [ ] Address subprocess security concerns (S603)
  - [ ] Fix SHA1 usage (S324)

### 3.2. Code Structure Improvements

- [ ] Implement consistent error handling
  - [ ] Create custom exception classes
  - [ ] Add exception hierarchy for different error types
  - [ ] Use try/except with specific exceptions (B904)

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
  - [x] Add tests for tonedown.py module (now at 38% coverage)
  - [x] Add tests for orator.py module (now at 24% coverage)
  - [x] Add tests for chunker.py module (now at 16% coverage, up from 7%)
  - [x] Add tests for elevenlabs_converter.py module (now at 48% coverage)
  - [x] Add tests for elevenlabs_synthesizer.py module (now at 89% coverage, up from 31%)
  - [x] Add tests for neifix.py module (now at 95% coverage, up from 10%)
  - [ ] Add tests for edge cases and error handling

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
