# Getting Started

## Requirements

- Python 3.10 or newer
- An API key for an LLM provider supported by [litellm](https://github.com/BerriAI/litellm) (OpenAI, Anthropic, Google Gemini, Cohere, and many others)
- *(Optional)* An [ElevenLabs](https://elevenlabs.io) API key if you want to synthesise audio directly from the tool

## Installation

```bash
pip install e11ocutionist
```

Or, if you use [uv](https://github.com/astral-sh/uv):

```bash
uv add e11ocutionist
```

## Setting up API keys

e11ocutionist uses environment variables for credentials. The easiest approach is to create a `.env` file in your working directory — the tool loads it automatically:

```dotenv
# .env
OPENAI_API_KEY=sk-...          # or any litellm-compatible key
ELEVENLABS_API_KEY=...         # only needed for the 'say' command
```

Alternatively, export the variables in your shell:

```bash
export OPENAI_API_KEY="sk-..."
```

## Running the full pipeline

The `process` command runs all five stages in sequence:

```bash
e11ocutionist process path/to/your_document.txt \
    --output-dir path/to/output/ \
    --verbose
```

After the command completes, `output/` will contain:

| File | Description |
|---|---|
| `your_document.chunked.xml` | After the chunking stage |
| `your_document.entitized.xml` | After entity tagging |
| `your_document.orated.xml` | After speech enhancement |
| `your_document.toneddown.xml` | After emphasis reduction |
| `your_document_step5_11labs.txt` | Final output, ready for ElevenLabs |

## Running individual stages

You can run each stage independently for fine-grained control:

```bash
# 1. Split into chunks
e11ocutionist chunk input.txt output_chunked.xml

# 2. Tag named entities
e11ocutionist entitize output_chunked.xml output_entitized.xml

# 3. Enhance for speech
e11ocutionist orate output_entitized.xml output_orated.xml

# 4. Reduce excessive emphasis
e11ocutionist tonedown output_orated.xml output_toneddown.xml

# 5. Convert for ElevenLabs
e11ocutionist convert-11labs output_toneddown.xml output_11labs.txt
```

## Synthesising audio

Once you have the final `.txt` file, use the `say` command:

```bash
e11ocutionist say \
    --input-file path/to/output/your_document_step5_11labs.txt \
    --output-dir path/to/audio/
```

This requires `ELEVENLABS_API_KEY` to be set and will produce `.mp3` files in the output directory.

## Choosing a model

By default the tool uses `gpt-4o`. To use a different model, pass the `--model` flag:

```bash
e11ocutionist process input.txt --output-dir out/ --model claude-3-5-sonnet-20241022
```

Any model identifier accepted by [litellm](https://github.com/BerriAI/litellm) works here.

## Resuming an interrupted run

If processing is interrupted (network error, API timeout, etc.) you can resume from the last completed stage by re-running the same `process` command. The tool detects the existing intermediate files and skips completed stages automatically.
