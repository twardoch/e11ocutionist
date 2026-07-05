# CLI Reference

All commands are accessed via the `e11ocutionist` entry point. Use `--help` on any subcommand for full option details.

## `process` — run the full pipeline

```
e11ocutionist process INPUT_FILE [OPTIONS]
```

Runs all five pipeline stages in sequence.

| Option | Default | Description |
|---|---|---|
| `--output-dir` | same dir as input | Directory for output files |
| `--model` | `gpt-4o` | litellm model identifier |
| `--temperature` | `0.2` | LLM sampling temperature |
| `--chunk-size` | `12288` | Maximum tokens per chunk |
| `--start-step` | *(beginning)* | Resume from this step name |
| `--force-restart` | `False` | Ignore existing progress, restart |
| `--verbose` | `False` | Enable detailed logging |
| `--backup` | `False` | Save a backup at each stage |

**Example:**
```bash
e11ocutionist process novel_chapter.txt \
    --output-dir ./output/ \
    --model gpt-4o-mini \
    --verbose
```

---

## `chunk` — stage 1: semantic chunking

```
e11ocutionist chunk INPUT_FILE OUTPUT_FILE [OPTIONS]
```

| Option | Default | Description |
|---|---|---|
| `--chunk-size` | `12288` | Maximum tokens per chunk |
| `--model` | `gpt-4o` | LLM model |
| `--temperature` | `0.2` | Sampling temperature |
| `--verbose` | `False` | Verbose logging |
| `--backup` | `False` | Save intermediate backups |

---

## `entitize` — stage 2: entity tagging

```
e11ocutionist entitize INPUT_FILE OUTPUT_FILE [OPTIONS]
```

| Option | Default | Description |
|---|---|---|
| `--model` | `gpt-4o` | LLM model |
| `--temperature` | `0.2` | Sampling temperature |
| `--verbose` | `False` | Verbose logging |
| `--backup` | `False` | Save intermediate backups |

---

## `orate` — stage 3: speech enhancement

```
e11ocutionist orate INPUT_FILE OUTPUT_FILE [OPTIONS]
```

| Option | Default | Description |
|---|---|---|
| `--steps` | `all` | Comma-separated list: `sentences`, `words`, `punctuation`, `emphasis` |
| `--model` | `gpt-4o` | LLM model |
| `--temperature` | `0.2` | Sampling temperature |
| `--verbose` | `False` | Verbose logging |
| `--backup` | `False` | Save intermediate backups |

**Example — run only sentence restructuring and word normalisation:**
```bash
e11ocutionist orate stage2.xml stage3.xml --steps sentences,words
```

---

## `tonedown` — stage 4: emphasis reduction

```
e11ocutionist tonedown INPUT_FILE OUTPUT_FILE [OPTIONS]
```

| Option | Default | Description |
|---|---|---|
| `--min-em-distance` | `50` | Minimum token gap required between `<em>` tags |
| `--model` | `gpt-4o` | LLM model (used for pronunciation review) |
| `--temperature` | `0.1` | Sampling temperature |
| `--verbose` | `False` | Verbose logging |
| `--backup` | `False` | Save intermediate backups |

---

## `convert-11labs` — stage 5: ElevenLabs conversion

```
e11ocutionist convert-11labs INPUT_FILE OUTPUT_FILE [OPTIONS]
```

| Option | Default | Description |
|---|---|---|
| `--dialog-mode` | `True` | Optimise for text containing dialogue |
| `--plaintext-mode` | `False` | Produce plain text without voice hints |
| `--verbose` | `False` | Verbose logging |

---

## `say` — synthesise audio via ElevenLabs

```
e11ocutionist say [OPTIONS]
```

Requires `ELEVENLABS_API_KEY` environment variable.

| Option | Description |
|---|---|
| `--input-file` | Path to the `.txt` file produced by stage 5 |
| `--output-dir` | Directory for audio output files |
| `--voice` | ElevenLabs voice name or ID (default: `Rachel`) |
| `--model` | ElevenLabs TTS model ID |
| `--verbose` | Verbose logging |
