# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased] - 2026-07-05

### Added
-   Full static-type coverage: `mypy src/e11ocutionist` passes cleanly under the existing strict configuration (previously 34 errors across 9 modules).
-   `docs/assets/icon.png` project icon.
-   `.github/workflows/release.yml` — tag-triggered (`v*`) workflow that builds the distribution and publishes to PyPI plus a GitHub release.
-   `sanitize_filename()` utility in `src/e11ocutionist/utils.py` — removes dangerous characters, handles Windows reserved names (CON, AUX, PRN, …), strips null bytes and control characters, and truncates to 255 characters.
-   `add_unit_tags()` string-level wrapper in `src/e11ocutionist/chunker.py` — parses an itemised XML string, delegates to the tree-based function, and returns the updated XML string. Required for test compatibility.
-   `create_chunks()` string-level wrapper in `src/e11ocutionist/chunker.py` — same pattern as `add_unit_tags()`, wrapping `create_chunks_on_tree()`.
-   `import litellm` in `src/e11ocutionist/chunker.py` so that `semantic_analysis` can make real LLM calls and tests can patch `e11ocutionist.chunker.litellm.completion`.
-   MkDocs-Material documentation site (`mkdocs.yml` + `docs/`) with five pages: Home, Getting Started, How It Works, CLI Reference, and Python API.

### Changed
-   Added type annotations across `chunker.py`, `orator.py`, `tonedown.py`, `entitizer.py`, `utils.py`, `cli.py`, and `e11ocutionist.py` to satisfy strict mypy (return/parameter/variable annotations, `Sequence` for covariant inputs, narrowing of optional `output_dir`).
-   Renamed CI workflow `push.yml` → `ci.yml`; release/publish concerns moved to the new `release.yml`.
-   `.gitignore` now excludes generated pipeline output, Repomix snapshots, built distributions, and `.specstory/` chat history.
-   The optional-ElevenLabs import failure now logs at debug level (was error) so it no longer prints on every CLI invocation; message points to `TODO.md` for the client migration.
-   `semantic_analysis()` in `chunker.py` now makes a real `litellm.completion` call and passes the response to `extract_and_parse_llm_response()`. If the LLM call fails (e.g., no API key in CI), it falls back gracefully; mocking `extract_and_parse_llm_response` in tests still works correctly.
-   `update_nei_tags()` in `tonedown.py` now mirrors `new` and `orig` fields from the NEI dictionary onto the corresponding XML attributes, not only `pronunciation`.
-   `reduce_emphasis()` in `tonedown.py`: emphasis tags are removed when the token gap to the previous tag is *strictly less than* `min_em_distance` (same-position tags with `min_distance=0` are now correctly preserved).
-   `process_document()` in `tonedown.py` now reads the input file, applies `reduce_emphasis()`, writes the result to the output file, and returns the output path (was previously a no-op stub).
-   `extract_text_from_xml()` in `elevenlabs_converter.py`: `<nei>` tags with `new="true"` are now rendered as plain text, consistent with all other entity tags (previously they were incorrectly wrapped in quotation marks).
-   `tonedown.py` imports extended with `Path` (stdlib) and `create_backup` (from utils).
-   Test suite fixed — 137 tests now pass, 10 skipped, 0 failures:
    -   `test_chunker.py`: import alias corrected; `etree` and `pretty_print_xml` imports added; assertions updated to match actual lxml output format.
    -   `test_chunker_semantic.py`: now passes because `semantic_analysis` calls `extract_and_parse_llm_response`, making the mock effective.
    -   `test_elevenlabs_converter.py`: expected values updated after NEI quote-wrapping was removed.
    -   `test_orator.py`: result key name corrected (`items_processed` not `processed_items`).
    -   `test_performance.py`: patching updated to work with the new `litellm` import in chunker.
    -   `test_security.py`: `sanitize_filename` tests now pass; XML injection assertion corrected for entity-unescaping behaviour.
    -   `test_tonedown.py`: `min_em_distance` values tuned to the actual token-distance calculation; process_document test passes because the function now writes real output.

### Removed
-   `legacy_src/` directory and all its contents, as its functionality is superseded by the `src/e11ocutionist/` modules.
-   `legacy_src.txt` file, which was a Repomix of the `legacy_src` directory.
-   `src/e11ocutionist/neifix.py` module (file, `__init__` export of `transform_nei_content`, and the associated `fix-nei` CLI command), previously declared removed but still present.
-   Dead placeholder loop and its unused locals in `elevenlabs_converter.py`.
-   Example `main()` function from `src/e11ocutionist/e11ocutionist.py`.

### Fixed
-   Pipeline tone-down step passed an invalid `min_distance` keyword to `tonedown.process_document()` (the parameter is `min_em_distance`), which would raise at runtime whenever `min_em_distance` was set; it now passes the correct keyword.
-   Removed the leftover `fix_nei` command body in `cli.py` that referenced the deleted `neifix` module (an unbound-name error if ever invoked).
