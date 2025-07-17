# Manual Workflow Setup

Due to GitHub App permissions, the workflow files need to be added manually. Here are the two workflow files that need to be created:

## 1. Create `.github/workflows/ci.yml`

```yaml
name: CI

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main, develop]

env:
  FORCE_COLOR: 1

concurrency:
  group: ${{ github.workflow }}-${{ github.ref }}
  cancel-in-progress: true

jobs:
  lint:
    name: Code Quality
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
        with:
          fetch-depth: 0

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: "3.12"

      - name: Install Hatch
        run: pip install hatch

      - name: Run linting
        run: hatch run lint

      - name: Run formatting check
        run: hatch run format --check

      - name: Run type checking
        run: hatch run typecheck

  test:
    name: Test Python ${{ matrix.python }} on ${{ matrix.os }}
    runs-on: ${{ matrix.os }}
    needs: lint
    strategy:
      fail-fast: false
      matrix:
        python: ["3.10", "3.11", "3.12"]
        os: [ubuntu-latest, windows-latest, macOS-latest]

    steps:
      - uses: actions/checkout@v4
        with:
          fetch-depth: 0

      - name: Set up Python ${{ matrix.python }}
        uses: actions/setup-python@v5
        with:
          python-version: ${{ matrix.python }}

      - name: Install Hatch
        run: pip install hatch

      - name: Run tests
        run: hatch run test-cov

      - name: Upload coverage to Codecov
        if: matrix.python == '3.12' && matrix.os == 'ubuntu-latest'
        uses: codecov/codecov-action@v4
        with:
          fail_ci_if_error: false

  build:
    name: Build package
    runs-on: ubuntu-latest
    needs: test
    
    steps:
      - uses: actions/checkout@v4
        with:
          fetch-depth: 0

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: "3.12"

      - name: Install Hatch
        run: pip install hatch

      - name: Build package
        run: hatch build

      - name: Upload build artifacts
        uses: actions/upload-artifact@v4
        with:
          name: dist
          path: dist/
```

## 2. Update `.github/workflows/release.yml`

Replace the existing release.yml with:

```yaml
name: Release

on:
  push:
    tags: ["v*"]

env:
  FORCE_COLOR: 1

permissions:
  contents: write
  id-token: write

jobs:
  test:
    name: Test Python ${{ matrix.python }} on ${{ matrix.os }}
    runs-on: ${{ matrix.os }}
    strategy:
      fail-fast: false
      matrix:
        python: ["3.10", "3.11", "3.12"]
        os: [ubuntu-latest, windows-latest, macOS-latest]

    steps:
      - uses: actions/checkout@v4
        with:
          fetch-depth: 0

      - name: Set up Python ${{ matrix.python }}
        uses: actions/setup-python@v5
        with:
          python-version: ${{ matrix.python }}

      - name: Install Hatch
        run: pip install hatch

      - name: Run linting
        run: hatch run lint

      - name: Run type checking
        run: hatch run typecheck

      - name: Run tests
        run: hatch run test-cov

  build:
    name: Build distributions
    runs-on: ubuntu-latest
    needs: test
    
    steps:
      - uses: actions/checkout@v4
        with:
          fetch-depth: 0

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: "3.12"

      - name: Install Hatch
        run: pip install hatch

      - name: Build distributions
        run: hatch build

      - name: Upload build artifacts
        uses: actions/upload-artifact@v4
        with:
          name: dist
          path: dist/

  build-binaries:
    name: Build binary for ${{ matrix.os }}
    runs-on: ${{ matrix.os }}
    needs: test
    strategy:
      fail-fast: false
      matrix:
        include:
          - os: ubuntu-latest
            target: linux-x86_64
            artifact_name: e11ocutionist-linux-x86_64
          - os: windows-latest
            target: windows-x86_64
            artifact_name: e11ocutionist-windows-x86_64.exe
          - os: macos-latest
            target: macos-x86_64
            artifact_name: e11ocutionist-macos-x86_64
          - os: macos-latest
            target: macos-arm64
            artifact_name: e11ocutionist-macos-arm64

    steps:
      - uses: actions/checkout@v4
        with:
          fetch-depth: 0

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: "3.12"

      - name: Install Hatch and PyInstaller
        run: pip install hatch pyinstaller

      - name: Install dependencies
        run: hatch run python -m pip install -e .

      - name: Build binary (Linux/macOS)
        if: runner.os != 'Windows'
        run: |
          hatch run pyinstaller --onefile --name ${{ matrix.artifact_name }} \
            --add-data "src/e11ocutionist:e11ocutionist" \
            --hidden-import e11ocutionist \
            --hidden-import e11ocutionist.__main__ \
            src/e11ocutionist/__main__.py

      - name: Build binary (Windows)
        if: runner.os == 'Windows'
        run: |
          hatch run pyinstaller --onefile --name ${{ matrix.artifact_name }} \
            --add-data "src/e11ocutionist;e11ocutionist" \
            --hidden-import e11ocutionist \
            --hidden-import e11ocutionist.__main__ \
            src/e11ocutionist/__main__.py

      - name: Upload binary artifact
        uses: actions/upload-artifact@v4
        with:
          name: ${{ matrix.artifact_name }}
          path: dist/${{ matrix.artifact_name }}*

  publish:
    name: Publish to PyPI
    runs-on: ubuntu-latest
    needs: [test, build, build-binaries]
    environment:
      name: pypi
      url: https://pypi.org/p/e11ocutionist

    steps:
      - uses: actions/checkout@v4
        with:
          fetch-depth: 0

      - name: Download build artifacts
        uses: actions/download-artifact@v4
        with:
          name: dist
          path: dist/

      - name: Download binary artifacts
        uses: actions/download-artifact@v4
        with:
          pattern: e11ocutionist-*
          path: binaries/
          merge-multiple: true

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: "3.12"

      - name: Install Hatch
        run: pip install hatch

      - name: Publish to PyPI
        uses: pypa/gh-action-pypi-publish@release/v1
        with:
          password: ${{ secrets.PYPI_TOKEN }}

      - name: Create GitHub Release
        env:
          GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}
        run: |
          TAG_NAME=${GITHUB_REF#refs/tags/}
          echo "Creating release for tag: $TAG_NAME"
          
          # Create release notes
          echo "## Release $TAG_NAME" > release_notes.md
          echo "" >> release_notes.md
          echo "### Installation" >> release_notes.md
          echo "" >> release_notes.md
          echo "#### Via pip" >> release_notes.md
          echo '```bash' >> release_notes.md
          echo "pip install e11ocutionist==$TAG_NAME" >> release_notes.md
          echo '```' >> release_notes.md
          echo "" >> release_notes.md
          echo "#### Download binaries" >> release_notes.md
          echo "Pre-compiled binaries are available for:" >> release_notes.md
          echo "- Linux x86_64" >> release_notes.md
          echo "- Windows x86_64" >> release_notes.md
          echo "- macOS x86_64" >> release_notes.md
          echo "- macOS ARM64" >> release_notes.md
          echo "" >> release_notes.md
          echo "### Changes" >> release_notes.md
          echo "See [CHANGELOG.md](CHANGELOG.md) for detailed changes." >> release_notes.md
          
          # Create release
          gh release create "$TAG_NAME" \
            --title "Release $TAG_NAME" \
            --notes-file release_notes.md \
            dist/* \
            binaries/*
```

## Setup Instructions

1. **Create the CI workflow file**:
   - Go to your GitHub repository
   - Navigate to `.github/workflows/`
   - Create a new file named `ci.yml`
   - Copy the first YAML content above into it

2. **Update the release workflow file**:
   - Edit the existing `.github/workflows/release.yml`
   - Replace its contents with the second YAML content above

3. **Configure PyPI publishing** (if not already done):
   - Go to your repository Settings > Secrets and variables > Actions
   - Add a secret named `PYPI_TOKEN` with your PyPI API token
   - Create a PyPI environment named `pypi` in Settings > Environments

4. **Test the workflows**:
   - The CI workflow will run on pushes to main/develop branches
   - The release workflow will run when you push a git tag like `v1.0.0`

## What These Workflows Do

### CI Workflow (`ci.yml`)
- **Triggers**: Push to main/develop, Pull requests
- **Jobs**: 
  - Code quality checks (lint, format, typecheck)
  - Matrix testing across Python 3.10/3.11/3.12 and Ubuntu/Windows/macOS
  - Package building

### Release Workflow (`release.yml`)
- **Triggers**: Git tags matching `v*`
- **Jobs**:
  - Full test suite across all platforms
  - Python package building (wheel + sdist)
  - Multi-platform binary building (Linux, Windows, macOS x86_64, macOS ARM64)
  - PyPI publication
  - GitHub release creation with all artifacts

Once you add these files manually, the complete CI/CD pipeline will be fully functional!