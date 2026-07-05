# e11ocutionist — Plan

The MVP streamlining and the modernization pass (strict typing, lint, CI/release
workflows, packaging) are complete. What remains are larger improvements that go
beyond housekeeping.

## 1. Restore the ElevenLabs `say` command

`elevenlabs_synthesizer.py` targets an old ElevenLabs SDK surface
(`elevenlabs.api.Voices`, top-level `generate`/`set_api_key`/`voices`). These were
removed in current `elevenlabs` releases, so the import falls back to placeholders
and the `say` command cannot synthesize audio.

- Port the synthesizer to the current `ElevenLabs()` client (`client.voices.get_all()`,
  `client.text_to_speech.convert(...)`).
- Keep the graceful-degradation path so the rest of the pipeline works without the
  optional dependency.
- Add tests that mock the client (no live API calls).

## 2. Document the intermediate contracts

The pipeline passes an XML document between stages and tracks state in
`progress.json`.

- Write a short spec of the XML schema (`<doc>/<chunk>/<unit>/<item>` plus `<nei>`,
  `<em>`, `<hr/>` inline markup) and where each stage adds or consumes attributes
  (`id`, `tok`, `type`, `new`, `orig`).
- Document the `progress.json` schema (per-step `output_file` / `completed`).

## 3. Pipeline robustness

- Confirm consistent use of `lxml` across stages (avoid mixing with `xml.etree`).
- Add an end-to-end test that runs all five steps with a fully mocked LLM.
- Add an offline/`--dry-run` mode that reports what each step would do without LLM calls.

## 4. Test coverage

- Revisit the 10 skipped tests (filesystem-permission and symlink cases) and enable
  them where the CI environment allows.
- Raise coverage on the orchestrator (`E11ocutionistPipeline`) step methods.
