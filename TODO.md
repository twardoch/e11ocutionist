# TODO List for e11ocutionist

The old code that we are porting (for your reference) is in the `legacy_src/` folder. 

The new code is in the `src/` folder.

## 1. Implement Core Processing Modules

### 1.1. [x] 1.1. Implement `chunker.py` Module

Your reference is: `legacy_src/malmo_chunker.py`

### 1.2. [x] Implement `entitizer.py` Module

Your reference is: `legacy_src/malmo_entitizer.py`


### 1.3. [x] Implement `orator.py` Module

Your reference is: `legacy_src/malmo_orator.py`

### 1.4. [x] Implement `tonedown.py` Module

Your reference is: `legacy_src/malmo_tonedown.py`

## 2. Add Testing and Documentation

### 2.1. Implement Unit Tests

- [ ] Create test fixtures
  - Add sample XML files for testing
  - Create mock LLM responses
  - Set up test environment variables

- [x] Add tests for `entitizer.py`
  - [x] Test NEI identification and tagging (test_entitizer.py has functions)
  - [x] Test dictionary building and updates
  - [x] Test XML merging

- [ ] Add tests for `chunker.py`
  - Test paragraph splitting and classification
  - Test semantic analysis
  - Test chunk creation with various inputs

- [ ] Add tests for `orator.py`
  - Test each processing step individually
  - Test full pipeline with various inputs
  - Test error handling and recovery

- [ ] Add tests for `tonedown.py`
  - Test NEI dictionary building
  - Test language detection
  - Test pronunciation cue review
  - Test emphasis reduction

- [ ] Add tests for integration between modules
  - Test passing results between pipeline stages
  - Test full pipeline execution
  - Test error handling between modules

### 2.2. Improve Documentation

- [ ] Update README.md
  - Add comprehensive installation instructions
  - Add detailed usage examples for CLI
  - Add detailed examples for programmatic usage
  - Add development setup instructions

- [ ] Add docstrings to all public functions
  - Use Google-style docstrings
  - Include parameter descriptions
  - Document return values
  - Add examples where appropriate

- [ ] Document CLI commands
  - Create comprehensive help text
  - Add examples for each command
  - Document all options and parameters

### 2.3. Final Integration Tasks

- [ ] Add comprehensive error handling
  - Implement proper exception handling throughout
  - Add meaningful error messages
  - Add recovery strategies for common failures

- [ ] Add logging enhancements
  - Improve log messages for clarity
  - Add log levels for different verbosity needs
  - Add progress indicators for long-running operations

- [ ] Create example workflows
  - Add example scripts for common tasks
  - Document typical processing workflows
  - Include sample inputs and expected outputs

## 3. Code Quality and Maintenance 

### 3.1. Fix Linting Issues

- [ ] Update pyproject.toml configuration
  - Move 'per-file-ignores' to 'lint.per-file-ignores' section
  - Update target Python version to 3.12
  - Add specific rules to address common issues

- [ ] Address FBT001/FBT002 warnings
  - Replace boolean positional arguments with keyword-only arguments
  - Update function signatures and calls throughout the codebase
  - Add type hints for boolean parameters

- [ ] Fix PLR0913 warnings (too many arguments)
  - Refactor functions with too many parameters
  - Create parameter classes/dataclasses where appropriate
  - Use keyword arguments consistently

- [ ] Add timezone to datetime calls
  - Replace datetime.now() with datetime.now(timezone.utc)
  - Update date/time handling for consistency
  - Add proper timezone awareness

- [ ] Refactor complex functions
  - Break down functions with high complexity (C901)
  - Reduce branch complexity (PLR0912)
  - Reduce statement count (PLR0915)

- [ ] Address security issues
  - Fix lxml security warnings (S320)
  - Implement proper input sanitization
  - Address subprocess security concerns (S603)

### 3.2. Code Structure Improvements

- [ ] Implement consistent error handling
  - Create custom exception classes
  - Add exception hierarchy for different error types
  - Use try/except with specific exceptions

- [ ] Enhance module structure
  - Organize related functions into logical classes
  - Improve code reuse across modules
  - Create clear interfaces between components

- [ ] Implement configuration management
  - Create central configuration object
  - Add validation for configuration parameters
  - Provide sensible defaults and documentation

### 3.3. Testing and CI/CD

- [ ] Increase test coverage
  - Add tests for all core functionality
  - Add integration tests for full pipeline
  - Add tests for edge cases and error handling

- [ ] Set up GitHub Actions workflow
  - Add workflow for running tests
  - Add workflow for linting and type checking
  - Add workflow for publishing package

## 4. Feature Enhancements

### 4.1. Performance Optimizations

- [ ] Improve LLM usage efficiency
  - Optimize prompts for token efficiency
  - Implement caching for LLM responses
  - Add batch processing where possible

- [ ] Implement parallel processing
  - Process chunks in parallel
  - Add concurrency controls
  - Ensure thread safety

### 4.2. User Experience Improvements

- [ ] Add progress reporting
  - Implement CLI progress bars
  - Add estimated time remaining
  - Show processing statistics

- [ ] Create user-friendly error messages
  - Add detailed diagnostics
  - Suggest solutions for common errors
  - Provide clear guidance for configuration issues

### 4.3. Additional Features

- [ ] Add voice customization options
  - Support for voice cloning
  - Implement voice style controls
  - Add support for additional TTS providers

- [ ] Implement document format conversions
  - Add support for Markdown
  - Add support for HTML
  - Add support for ePub

- [ ] Create visualization tools
  - Add speech synthesis preview
  - Implement processing visualization
  - Create interactive pipeline control
