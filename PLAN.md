# E11ocutionist Streamlining Plan (MVP Focus)

## Goal

To streamline the `e11ocutionist` codebase for a performant, focused v1.0 (MVP) that does its job very well. This involves removing redundant/obsolete code, consolidating where appropriate, and ensuring the remaining code is clear and essential for core functionality.

## Detailed Steps

1.  **Remove Legacy Code (`legacy_src` directory)**
    *   **Rationale:** Analysis confirmed that all essential functionalities of the scripts within `legacy_src` (including `malmo_all.py`, `malmo_chunker.py`, `malmo_entitizer.py`, `malmo_orator.py`, `malmo_tonedown.py`, `malmo_11labs.py`, `say_11labs.py`, `malmo_neifix.py`, and `sapko.sh`) are superseded by the modernized and refactored code in the `src/e11ocutionist/` directory. The new structure uses a dedicated pipeline orchestrator class (`E11ocutionistPipeline`) and a `python-fire` based CLI, offering better organization and maintainability. Features not critical for MVP (e.g., "benchmark mode" from legacy entitizer) have already been omitted in the `src` version.
    *   **Action:** Delete the entire `legacy_src` directory and associated files.
    *   **Files to delete:**
        *   `legacy_src/LEGACY.md`
        *   `legacy_src/malmo_11labs.py`
        *   `legacy_src/malmo_all.py`
        *   `legacy_src/malmo_chunker.py`
        *   `legacy_src/malmo_entitizer.py`
        *   `legacy_src/malmo_neifix.py`
        *   `legacy_src/malmo_orator.py`
        *   `legacy_src/malmo_tonedown.py`
        *   `legacy_src/sapko.sh`
        *   `legacy_src/say_11labs.py`
        *   The directory `legacy_src/` itself.
        *   `legacy_src.txt` (as it's a Repomix of the `legacy_src` directory).

2.  **Streamline `src/e11ocutionist/` for MVP**
    *   **Remove `neifix.py` and its CLI command `fix-nei`**
        *   **Rationale:** `neifix.py` provides rule-based post-processing for NEI tags. For an MVP, the core entitizing (`entitizer.py`) and toning down (`tonedown.py`) steps should ideally produce correctly formatted NEIs directly, possibly through improved LLM prompting. Relying on a separate fix-up script adds complexity to the user workflow and suggests a deficiency in the primary steps. Removing it encourages focus on making the core pipeline robust.
        *   **Action:**
            *   Delete the file `src/e11ocutionist/neifix.py`.
            *   Remove the `fix-nei` command from `src/e11ocutionist/cli.py` (from the `fire.Fire` call and the `fix_nei` function itself).
            *   Remove any tests specifically for `neifix.py` (e.g., `tests/test_neifix.py`).
    *   **Remove example `main()` function from `src/e11ocutionist/e11ocutionist.py`**
        *   **Rationale:** The `main()` function in `e11ocutionist.py` provides an example of programmatic usage. The `README.md` already covers programmatic usage. Removing this function will make the file cleaner and more focused on the `E11ocutionistPipeline` class definition.
        *   **Action:** Delete the `main()` function and its example `PipelineConfig` instantiation from `src/e11ocutionist/e11ocutionist.py`. Also remove the `if __name__ == "__main__": main()` block.

3.  **Update Documentation and Tests**
    *   **`README.md`:**
        *   Remove any references to `legacy_src` if they exist.
        *   Remove the `fix-nei` command from the CLI usage examples if it's listed.
    *   **Tests:**
        *   Remove `tests/test_neifix.py` (or any tests related to `neifix` functionality).
        *   Review other tests to ensure they align with the streamlined codebase. The primary functionality of the pipeline should remain well-tested.
        *   Run all tests to confirm no regressions.

4.  **Record Changes**
    *   Maintain `CHANGELOG.md` with all deletions and refactoring actions performed.
    *   Update `PLAN.md` and `TODO.md` to reflect progress.

5.  **Submit Changes**
    *   Commit all changes with a clear message detailing the streamlining efforts and rationale.
