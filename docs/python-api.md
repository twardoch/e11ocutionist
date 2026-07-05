# Python API

e11ocutionist exposes a clean Python API for integration into larger pipelines.

## Running the full pipeline

```python
from pathlib import Path
from e11ocutionist import E11ocutionistPipeline, PipelineConfig

config = PipelineConfig(
    input_file=Path("my_chapter.txt"),
    output_dir=Path("output/"),
    verbose=True,
    # Optional overrides
    chunker_model="gpt-4o",
    chunker_temperature=0.2,
    chunk_size=12288,
)

pipeline = E11ocutionistPipeline(config)
final_path = pipeline.run()
print(f"Ready for ElevenLabs: {final_path}")
```

`pipeline.run()` returns the `Path` to the final `.txt` file.

### Resuming from a specific stage

```python
from e11ocutionist import ProcessingStep

config = PipelineConfig(
    input_file=Path("my_chapter.txt"),
    output_dir=Path("output/"),
    start_step=ProcessingStep.ORATING,  # skip chunking and entitizing
)
pipeline = E11ocutionistPipeline(config)
pipeline.run()
```

Available `ProcessingStep` values: `CHUNKING`, `ENTITIZING`, `ORATING`, `TONING_DOWN`, `ELEVENLABS_CONVERSION`.

---

## Running individual stages

Each stage is also available as a standalone function.

### Stage 1: Chunking

```python
from e11ocutionist.chunker import process_document as chunk_document

result = chunk_document(
    input_file="chapter.txt",
    output_file="chapter.chunked.xml",
    chunk_size=12288,
    model="gpt-4o",
    temperature=0.2,
    verbose=True,
    backup=False,
)
# result is a dict with keys: input_file, output_file, items_count,
# units_count, chunks_count, total_tokens, success
```

### Stage 2: Entitizing

```python
from e11ocutionist.entitizer import process_document as entitize_document

result = entitize_document(
    input_file="chapter.chunked.xml",
    output_file="chapter.entitized.xml",
    model="gpt-4o",
    temperature=0.2,
)
```

### Stage 3: Orating

```python
from e11ocutionist.orator import process_document as orate_document

result = orate_document(
    input_file="chapter.entitized.xml",
    output_file="chapter.orated.xml",
    steps=["sentences", "words", "punctuation", "emphasis"],  # or None for all
    model="gpt-4o",
    temperature=0.2,
)
# result dict keys: input_file, output_file, steps_performed,
# items_processed, success
```

### Stage 4: Toning Down

```python
from e11ocutionist.tonedown import process_document as tonedown_document

output_path = tonedown_document(
    input_file="chapter.orated.xml",
    output_file="chapter.toneddown.xml",
    min_em_distance=50,
    model="gpt-4o",
)
```

### Stage 5: ElevenLabs conversion

```python
from e11ocutionist.elevenlabs_converter import process_document as convert_document

convert_document(
    input_file="chapter.toneddown.xml",
    output_file="chapter_11labs.txt",
    dialog_mode=True,
)
```

---

## Synthesising audio

```python
from e11ocutionist.elevenlabs_synthesizer import synthesize_document

synthesize_document(
    input_file="chapter_11labs.txt",
    output_dir="audio/",
    voice="Rachel",
)
```

Requires `ELEVENLABS_API_KEY` to be set in the environment.

---

## Utility functions

```python
from e11ocutionist.utils import (
    count_tokens,       # count tokens in a string using tiktoken
    sanitize_filename,  # make a string safe to use as a filename
    parse_xml,          # parse XML with error recovery
    serialize_xml,      # serialise an lxml element back to a string
    escape_xml_chars,   # escape <, >, &, ", ' in text
    unescape_xml_chars, # reverse of escape_xml_chars
    generate_hash,      # 6-character base-36 hash of a string
    create_backup,      # timestamped copy of a file
)
```
