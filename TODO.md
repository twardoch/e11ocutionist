# e11ocutionist — TODO

Streamlining and modernization are done. Remaining items (see PLAN.md for detail):

## ElevenLabs `say` command
- [ ] Port `elevenlabs_synthesizer.py` to the current `ElevenLabs()` client API
- [ ] Add client-mocked tests (no live API calls)

## Documentation
- [ ] Spec the inter-stage XML format (tags + attributes per stage)
- [ ] Document the `progress.json` schema

## Robustness
- [ ] Verify consistent `lxml` usage across stages
- [ ] Add a fully-mocked end-to-end pipeline test
- [ ] Add an offline/`--dry-run` mode

## Test coverage
- [ ] Re-enable the 10 skipped filesystem/symlink tests where CI allows
- [ ] Raise coverage on `E11ocutionistPipeline` step methods
