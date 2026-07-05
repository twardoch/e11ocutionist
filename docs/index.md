# e11ocutionist

> Turn your written words into captivating, natural-sounding audio.

**e11ocutionist** is a Python tool that prepares written text for high-quality speech synthesis. Before sending your document to a text-to-speech service such as ElevenLabs, it runs a series of intelligent transformations that make the spoken result sound dramatically more natural and professional.

## What does it actually do?

Imagine handing a page of a novel to a human narrator. A skilled narrator doesn't just read every word exactly as printed — they adjust sentence rhythm, decide how to pronounce unusual names consistently, add subtle pauses, and avoid over-emphasising every other word. **e11ocutionist** does all of that automatically, using Large Language Models (LLMs) to understand the context of your text.

The tool processes your document through up to five sequential stages:

| Stage | What happens |
|---|---|
| **Chunking** | The text is split into semantically coherent sections. Long documents are divided at natural boundaries so each piece fits within LLM context limits. |
| **Entitizing** | Named entities (character names, places, technical terms) are identified and tagged so they can receive consistent pronunciation treatment throughout the document. |
| **Orating** | Sentences are restructured for natural spoken rhythm: long sentences are shortened, numbers are spelled out, abbreviations expanded, and emphasis markers added where a human narrator would stress a word. |
| **Toning Down** | The previous stage can be over-eager. This stage reviews the emphasis tags and removes any that are too close together or stylistically inappropriate, producing a calmer, more natural result. |
| **ElevenLabs Conversion** | The processed XML is converted to a plain-text format optimised for the ElevenLabs API, ready for speech synthesis. |

## Who is it for?

- **Authors and publishers** who want to turn books, stories, or articles into audiobooks without spending hours correcting a robotic-sounding TTS output.
- **Content creators** converting blog posts, scripts, or educational materials into podcast audio or video narration.
- **Developers** who need robust text pre-processing as part of a larger TTS pipeline.

## Quick start

```bash
pip install e11ocutionist
```

Set your LLM API key (the tool uses [litellm](https://github.com/BerriAI/litellm), so most major providers work):

```bash
export OPENAI_API_KEY="sk-..."
# or ANTHROPIC_API_KEY, GEMINI_API_KEY, etc.
```

Process a document:

```bash
e11ocutionist process my_chapter.txt --output-dir ./processed/
```

The output directory will contain the final `.txt` file ready for ElevenLabs, plus intermediate XML files for each stage.

To synthesise speech directly (requires `ELEVENLABS_API_KEY`):

```bash
e11ocutionist say --input-file ./processed/my_chapter_step5_11labs.txt \
                  --output-dir ./audio/
```

See [Getting Started](getting-started.md) for a complete walk-through, or [CLI Reference](cli-reference.md) for all available options.
