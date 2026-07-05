# How It Works

## The pipeline at a glance

```
Plain text / Markdown
        │
        ▼
┌─────────────────┐
│   1. Chunking   │  Splits the document into semantic sections
└────────┬────────┘
         │  XML with <chunk> and <item> elements
         ▼
┌──────────────────┐
│  2. Entitizing   │  Tags named entities (people, places, terms)
└────────┬─────────┘
         │  XML with <nei> tags
         ▼
┌─────────────────┐
│   3. Orating    │  Rewrites text for natural spoken delivery
└────────┬────────┘
         │  XML with <em> tags and restructured sentences
         ▼
┌──────────────────────┐
│   4. Toning Down     │  Removes excessive or clustered emphasis
└──────────┬───────────┘
           │  Refined XML
           ▼
┌─────────────────────────┐
│  5. ElevenLabs Convert  │  Serialises to plain text for TTS API
└─────────────────────────┘
           │
           ▼
   Ready-to-synthesise .txt
```

## Stage 1: Chunking (`chunker.py`)

**Input:** Plain text or Markdown file.

**What it does:** The document is first split into paragraph-level *items* (individual paragraphs, headings, list blocks, blockquotes, etc.). A semantic analysis step — powered by an LLM — then groups those items into *units* (a scene, a section, a chapter), and units are grouped into *chunks* that respect a token-size limit so no chunk exceeds the LLM's context window in later stages.

**Output:** An XML file with structure:
```xml
<doc>
  <chunk id="0001">
    <unit type="chapter">
      <item id="…">First paragraph text.</item>
      <item id="…">Second paragraph text.</item>
    </unit>
  </chunk>
</doc>
```

## Stage 2: Entitizing (`entitizer.py`)

**Input:** The chunked XML from stage 1.

**What it does:** The LLM reads each chunk and identifies *Named Entities of Interest* (NEIs): character names, place names, brand names, technical jargon — anything that appears multiple times and might be mispronounced by a generic TTS engine. Each occurrence is wrapped in a `<nei>` tag with a `type` attribute (`person`, `place`, `org`, …). New entities are flagged with `new="true"` on first occurrence.

**Output:** XML with inline entity tags:
```xml
<item id="…">
  <nei type="person">Hermione Granger</nei> raised her hand.
</item>
```

The resulting entity dictionary can be reviewed or edited before proceeding, giving you control over how unusual names will sound.

## Stage 3: Orating (`orator.py`)

**Input:** The entitized XML from stage 2.

**What it does:** This is the heart of the pipeline. The LLM rewrites the text inside each item to sound more natural when spoken aloud. Concretely it:

- **Restructures sentences** — splits run-on sentences, reorders clauses for spoken clarity.
- **Normalises words** — converts digits to words (`42` → `forty-two`), expands abbreviations (`Dr.` → `Doctor`), replaces symbols (`&` → `and`).
- **Adjusts punctuation** — adds pauses via ellipses or em-dashes at natural breath points.
- **Adds emphasis** — wraps key words in `<em>` tags so the TTS engine stresses them as a human narrator would.

All four sub-steps can be run individually or all at once (the default).

## Stage 4: Toning Down (`tonedown.py`)

**Input:** The orated XML from stage 3.

**What it does:** Stage 3 can be over-enthusiastic — it may place `<em>` tags too densely, creating an unnaturally staccato delivery. This stage enforces a minimum *token distance* between consecutive emphasis tags: any `<em>` that appears too soon after the previous one is removed and its text is preserved as plain text. The threshold (`min_em_distance`, default 50 tokens) is configurable.

It also reviews and consolidates Named Entity pronunciations across the whole document, ensuring consistency.

## Stage 5: ElevenLabs Conversion (`elevenlabs_converter.py`)

**Input:** The toned-down XML from stage 4.

**What it does:** Converts the internal XML format into the plain text (or lightly-tagged text) that the ElevenLabs API expects. Dialogue markers and voice-change hints are preserved; XML tags that have no ElevenLabs equivalent are stripped cleanly.

**Output:** A `.txt` file ready to paste into ElevenLabs or pass to `e11ocutionist say`.

## The orchestrator (`e11ocutionist.py`)

The `E11ocutionistPipeline` class ties all five stages together. It:

- Reads a `progress.json` file in the output directory to detect which stages have already completed.
- Skips completed stages, so an interrupted run can resume from where it left off.
- Passes the output path of each stage as the input to the next.
- Exposes a `run()` method that returns the path to the final output file.
