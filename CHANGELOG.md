# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased] - YYYY-MM-DD

### Removed
-   `legacy_src/` directory and all its contents, as its functionality is superseded by the `src/e11ocutionist/` modules.
-   `legacy_src.txt` file, which was a Repomix of the `legacy_src` directory.
-   `src/e11ocutionist/neifix.py` module and the associated `fix-nei` CLI command. This functionality is considered non-MVP, and core NEI processing should handle formatting.
-   Example `main()` function from `src/e11ocutionist/e11ocutionist.py` to streamline the module, as programmatic usage is covered in README.

### Changed
-   Updated `src/e11ocutionist/cli.py` to remove import and registration of `neifix` module and `fix_nei` command.
