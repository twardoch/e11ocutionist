#!/usr/bin/env bash
echo
repomix -o src.txt src/
repomix -o legacy_src.txt legacy_src/
echo
echo
echo
echo "# PROJECT STRUCTURE:"
tree
echo
echo
echo
echo "# CAREFULLY REVIEW THE LEGACY SOURCE CODE:"
cat legacy_src.txt
echo
echo
echo
echo "# CAREFULLY REVIEW THEOUR NEW SOURCE CODE (and compare its functionality and completeness to the legacy code):"
cat src.txt
echo
echo
echo
echo "# THIS IS THE FORMATTING AND LINTING RESULTS:"

uv pip install -e .
echo "> hatch fmt -l"
hatch fmt -l
echo "> hatch fmt -f"
hatch fmt -f
echo "> autoflake"
fd -e py -x autoflake {}
echo "> pyupgrade"
fd -e py -x pyupgrade --py311-plus {}
echo "> ruff check"
fd -e py -x ruff check --output-format=github --fix --unsafe-fixes {}
echo "> ruff format"
fd -e py -x ruff format --respect-gitignore --target-version py311 {}
echo
echo
echo
echo "# THIS IS THE TEST RESULTS:"
hatch run test
hatch run test-cov
echo
echo
echo
echo "# THIS IS OUR CURRENT TODO:"
cat TODO.md
echo
echo
echo
echo "I think it’s time to make some more tests, and iteratively run ./UPDATE.sh until the tests pass."
echo "<task>"
echo "(1) Focus on implementing TODO items that are high priority, especially test failures and new tests."
echo "(2) Update the progress in the 'TODO.md' file: mark items as done."
echo "(3) Run ./UPDATE.sh again until and continue improving, until the tests pass."
echo "</task>"
echo
echo
