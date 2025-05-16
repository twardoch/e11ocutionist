# TODO List for e11ocutionist

The old code that we are porting (for your reference) is in the `legacy_src/` directory.

## 1. Implement Core Processing Modules

### 1.1. Complete `chunker.py` Implementation

Your reference is: `legacy_src/malmo_chunker.py`

- [x] Implement `itemize_document` function
  - Add a function that takes a text document and splits it into semantic paragraph-level units
  - Use regex to identify paragraph boundaries (blank lines)
  - Classify paragraphs as headings, blockquotes, lists, code blocks, etc.
  - Return a list of tuples with `(item_text, attachment_type)` where attachment_type is "following", "preceding", or "normal"

- [x] Implement `create_item_elements` function
  - Create a function that processes the output of `itemize_document`
  - Generate XML items with appropriate attributes (id, tok)
  - Preserve whitespace by setting `xml:space="preserve"`
  - Generate unique IDs for each item using the `generate_id` function in `utils.py`
  - Return a list of tuples containing `(item_xml, item_id)`

- [x] Implement `semantic_analysis` function
  - Create a function that uses an LLM to identify semantic boundaries (chapters, scenes, etc.)
  - Send the document with its items to the LLM with a prompt to identify semantic units
  - Process the LLM's response to extract identified boundaries
  - Add fallback logic for cases where the LLM fails or identifies too few boundaries
  - Return a list of tuples with `(item_id, boundary_type)` where boundary_type is "chapter", "scene", or "unit"

- [x] Implement `add_unit_tags` function
  - Create a function that adds `<unit>` tags around sequences of items based on semantic analysis
  - Group items by their boundary types
  - Set appropriate attributes on unit tags (type, tok)
  - Handle cases where semantic boundaries weren't identified
  - Add fallback logic using regex for parsing failures

- [x] Implement `create_chunks` function
  - Create a function that groups units into chunks of manageable size
  - Ensure each chunk doesn't exceed the maximum token size
  - Allow chapters to always start new chunks
  - Handle oversized units by splitting them across chunks
  - Add tok attributes with accurate token counts

- [x] Handle oversized units
  - Add function to split oversized units across multiple chunks
  - Preserve chunk and unit attributes
  - Maintain document hierarchy

### 1.2. Implement `entitizer.py` Module

Your reference is: `legacy_src/malmo_entitizer.py`

- [x] Create the basic module structure
  - Copy the file structure from `chunker.py` as a template
  - Add imports for lxml, backoff, loguru, etc.
  - Define constants for default models and temperature

- [x] Implement XML parsing and serialization
  - Add functions to parse input XML
  - Extract chunks and items from XML
  - Add functions to serialize XML with proper encoding and formatting

- [x] Implement NEI identification with LLM
  - Create a function that uses an LLM to identify named entities
  - Craft a prompt that explains NEI tagging rules
  - Process the LLM's response to extract tagged text
  - Add retries and fallback for API failures

- [x] Implement NEI dictionary management
  - Create functions to extract NEIs from tags
  - Build and update a dictionary mapping NEIs to pronunciations
  - Track new vs. existing NEIs with the new="true" attribute

- [x] Implement XML merging
  - Create a function to merge LLM-tagged items back into the original XML
  - Maintain document structure while replacing content
  - Handle parsing errors gracefully

- [x] Add incremental saving
  - Implement saving after each chunk is processed
  - Save both the XML document and NEI dictionary
  - Add backup functionality with timestamps

### 1.3. Implement `orator.py` Module

Your reference is: `legacy_src/malmo_orator.py`

- [x] Create the basic module structure
  - Set up imports and constants
  - Define functions for processing documents and chunks

- [x] Implement sentence restructuring
  - Create a function that uses an LLM to divide long sentences into shorter ones
  - Ensure paragraphs and headings end with punctuation
  - Preserve all XML tags and structure
  - Extract and merge tagged items

- [x] Implement word normalization
  - Create a function that uses an LLM to convert digits to numeric words
  - Replace symbols like '/' and '&' with appropriate conjunctions
  - Spell out rare symbols
  - Preserve XML structure

- [x] Implement punctuation enhancement
  - Create a function that algorithmically enhances punctuation
  - Process parenthetical expressions
  - Replace em/en dashes with ellipses when used as pauses
  - Preserve XML structure

- [x] Implement emotional emphasis
  - Create a function that uses an LLM to add emotional emphasis
  - Add `<em>` tags to emphasize words/phrases
  - Insert `<hr/>` for dramatic pauses
  - Add exclamation marks for energy
  - Preserve XML structure

- [x] Add step selection logic
  - Implement logic to run specific steps based on configuration
  - Allow individual steps (sentences, words, punctuation, emotions) or all steps
  - Save after each step

### 1.4. Implement `tonedown.py` Module

Your reference is: `legacy_src/malmo_tonedown.py`

- [x] Create the basic module structure
  - Set up imports and constants
  - Define the main processing function

- [x] Implement NEI dictionary building
  - Create a function to extract NEIs from tags
  - Build a dictionary of NEIs with orig attribute and pronunciation cues
  - Set new="true" for first occurrences

- [x] Implement language detection
  - Create a function that uses an LLM to detect document language
  - Extract a representative sample from the document
  - Process the LLM's response to extract the language

- [x] Implement pronunciation cue review
  - Create a function that uses an LLM to review pronunciation cues
  - Send the NEI dictionary and detected language to the LLM
  - Process the LLM response to extract revised pronunciations
  - Update the NEI dictionary with revisions

- [x] Implement XML update
  - Create a function to update `<nei>` tags with revised pronunciations
  - Preserve tag attributes and structure
  - Update tag content with revised pronunciations

- [x] Implement emphasis reduction
  - Create a function to remove excessive `<em>` tags
  - Calculate word distances between emphasis tags
  - Remove tags that are too close together based on min_distance parameter
  - Add proper regex handling for complex XML

## 2. Add Testing and Documentation

### 2.1. Implement Unit Tests

- [x] Set up pytest framework
  - Create `tests/` directory with proper structure (already created)
  - Add `conftest.py` with common fixtures (needed)
  - Configure pytest in pyproject.toml (already configured)

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

- [x] Complete the pipeline integration
  - Verify all modules work together correctly
  - Test processing flow from chunking through ElevenLabs conversion
  - Ensure configuration is properly passed between modules

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
