# e11ocutionist

A multi-stage document processing pipeline for transforming literary content into enhanced speech synthesis markup.

## Overview

E11ocutionist processes text documents through a sequence of steps to prepare them for high-quality text-to-speech synthesis:

1. **Chunking**: Splits documents into semantic chunks based on content boundaries
2. **Entitizing**: Identifies and tags Named Entities of Interest (NEIs) for consistent pronunciation
3. **Orating**: Enhances text for spoken narrative (sentence restructuring, word normalization, etc.)
4. **Toning Down**: Reviews and refines NEI pronunciation cues and reduces excessive emphasis
5. **ElevenLabs Conversion**: Converts processed XML to ElevenLabs-compatible text format
6. **Speech Synthesis**: (Optional) Synthesizes text using personal ElevenLabs voices

## Features

- Sequential document processing pipeline with robust error handling
- Progress tracking and resumption capability
- LLM-powered semantic analysis for intelligent chunking
- Multi-language entity recognition and pronunciation mapping
- Speech enhancement through sentence restructuring and emphasis
- Direct integration with ElevenLabs API for speech synthesis

## Installation

```bash
pip install e11ocutionist
```

## Usage

### Command Line Interface

Process a document through the entire pipeline:

```bash
e11ocutionist process "input_document.txt" --output-dir "output_folder" --verbose
```

Run individual steps:

```bash
# Chunking step
e11ocutionist chunk "input.txt" "output_step1.xml" --model "gpt-4o" --temperature 0.2

# Entitizing step
e11ocutionist entitize "input_step1.xml" "output_step2.xml" --model "gpt-4o" --temperature 0.1

# Orating step
e11ocutionist orate "input_step2.xml" "output_step3.xml" --model "gpt-4o" --temperature 0.7 --all_steps

# Toning down step
e11ocutionist tonedown "input_step3.xml" "output_step4.xml" --model "gpt-4o" --temperature 0.1 --min_em_distance 10

# ElevenLabs conversion
e11ocutionist convert-11labs "input_step4.xml" "output_step5.txt" --dialog

# Speech synthesis (requires ElevenLabs API key)
e11ocutionist say --input-file "output_step5.txt" --output-dir "audio_files"
```

### Programmatic Usage

```python
from pathlib import Path
from e11ocutionist import E11ocutionistPipeline, PipelineConfig, ProcessingStep

# Configure the pipeline
config = PipelineConfig(
    input_file=Path("input_document.txt"),
    output_dir=Path("output_folder"),
    verbose=True
)

# Create and run the pipeline
pipeline = E11ocutionistPipeline(config)
result = pipeline.run()
```

## Development

This project uses [Hatch](https://hatch.pypa.io/) for development workflow management.

### Setup Development Environment

```bash
# Install hatch if you haven't already
pip install hatch

# Create and activate development environment
hatch shell

# Run tests
hatch run test

# Run tests with coverage
hatch run test-cov

# Run linting
hatch run lint

# Format code
hatch run format

# Apply auto-fixes (Ruff)
hatch run fix

# Run type checking (Mypy)
hatch run typecheck
```

## Requirements

- Python 3.10+
- LLM API access (via litellm)
- Optional: ElevenLabs API key for speech synthesis

## License

MIT License 