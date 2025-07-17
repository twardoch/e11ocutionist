# Build and Release Guide

This guide explains how to build, test, and release `e11ocutionist` using the included build system and CI/CD pipeline.

## Prerequisites

- Python 3.10 or higher
- [Hatch](https://hatch.pypa.io/) package manager
- Git
- (Optional) GitHub CLI for release management

Install Hatch:
```bash
pip install hatch
```

## Development Workflow

### 1. Setting up Development Environment

```bash
# Clone the repository
git clone https://github.com/twardoch/e11ocutionist.git
cd e11ocutionist

# Activate development environment
hatch shell

# Install development dependencies
hatch run python -m pip install -e .[dev,test]
```

### 2. Running Tests

```bash
# Run the full test suite
./scripts/test.sh

# Or use hatch directly
hatch run test        # Run tests
hatch run test-cov    # Run tests with coverage
hatch run lint        # Run linting
hatch run typecheck   # Run type checking
hatch run format      # Format code
```

### 3. Building Packages

```bash
# Build Python package (wheel and sdist)
./scripts/build.sh

# Build binary for current platform
./scripts/build-binary.sh --onefile

# Build everything
./scripts/build-all.sh
```

## Release Process

### Git-Tag-Based Semantic Versioning

The project uses git tags for versioning following semantic versioning (semver):

- `v1.0.0` - Major release
- `v1.1.0` - Minor release  
- `v1.1.1` - Patch release
- `v1.2.0-alpha` - Pre-release

### Creating a Release

1. **Prepare for Release**
   ```bash
   # Ensure you're on the main branch
   git checkout main
   git pull origin main
   
   # Run full test suite
   ./scripts/test.sh
   ```

2. **Create Release**
   ```bash
   # Create and push release tag
   ./scripts/release.sh -v v1.0.0
   ```

3. **GitHub Actions Automation**
   
   When you push a tag, GitHub Actions will automatically:
   - Run tests on Python 3.10, 3.11, 3.12
   - Test on Ubuntu, Windows, and macOS
   - Build Python packages (wheel and sdist)
   - Build binaries for all platforms:
     - Linux x86_64
     - Windows x86_64
     - macOS x86_64
     - macOS ARM64
   - Publish to PyPI
   - Create GitHub release with artifacts

## Available Scripts

### `./scripts/test.sh`
Runs the complete test suite including:
- Code linting with Ruff
- Type checking with MyPy
- Unit and integration tests with pytest
- Coverage reporting

### `./scripts/build.sh`
Builds Python packages:
- Wheel distribution
- Source distribution (sdist)

### `./scripts/build-binary.sh`
Builds standalone executables:
```bash
# Basic usage
./scripts/build-binary.sh

# Create single file executable
./scripts/build-binary.sh --onefile

# Custom output directory
./scripts/build-binary.sh -o build/

# Enable debug mode
./scripts/build-binary.sh --debug
```

### `./scripts/build-all.sh`
Builds everything:
```bash
# Build everything
./scripts/build-all.sh

# Skip tests
./scripts/build-all.sh --skip-tests

# Skip binaries
./scripts/build-all.sh --skip-binaries
```

### `./scripts/release.sh`
Manages releases:
```bash
# Create release with tag
./scripts/release.sh -v v1.0.0

# Dry run (test without making changes)
./scripts/release.sh --dry-run

# Help
./scripts/release.sh --help
```

## CI/CD Pipeline

### Continuous Integration (CI)

**Triggers**: Push to main/develop, Pull Requests

**Jobs**:
1. **Code Quality**: Linting, formatting, type checking
2. **Testing**: Matrix testing across Python versions and OS
3. **Build**: Package building and artifact storage

### Release Pipeline

**Triggers**: Git tags matching `v*`

**Jobs**:
1. **Testing**: Full test suite on all platforms
2. **Package Build**: Python wheel and source distributions
3. **Binary Build**: Standalone executables for all platforms
4. **Publishing**: 
   - PyPI package publication
   - GitHub release creation
   - Binary artifact attachment

## Multi-Platform Binary Distribution

The release process creates binaries for:

- **Linux x86_64**: `e11ocutionist-linux-x86_64`
- **Windows x86_64**: `e11ocutionist-windows-x86_64.exe`
- **macOS x86_64**: `e11ocutionist-macos-x86_64`
- **macOS ARM64**: `e11ocutionist-macos-arm64`

Users can download these from GitHub releases or install via pip.

## Installation Methods

### For End Users

**Via pip** (recommended):
```bash
pip install e11ocutionist
```

**Via binary download**:
1. Go to [GitHub Releases](https://github.com/twardoch/e11ocutionist/releases)
2. Download the binary for your platform
3. Make it executable (Linux/macOS): `chmod +x e11ocutionist-*`
4. Run: `./e11ocutionist-* --help`

### For Developers

```bash
# Development installation
git clone https://github.com/twardoch/e11ocutionist.git
cd e11ocutionist
hatch shell
hatch run python -m pip install -e .[dev,test]
```

## Version Management

The project uses `hatch-vcs` for automatic version management:

- Version is automatically derived from git tags
- No manual version bumping required
- Development versions include commit info
- Release versions match git tags

## Testing

### Test Categories

- **Unit Tests**: Individual function testing
- **Integration Tests**: Cross-module functionality
- **Performance Tests**: Speed and memory benchmarks
- **Security Tests**: Input validation and vulnerability checks
- **Error Handling Tests**: Edge cases and failure scenarios

### Running Specific Tests

```bash
# Run all tests
hatch run test

# Run with coverage
hatch run test-cov

# Run specific test file
hatch run pytest tests/test_chunker.py

# Run performance tests
hatch run pytest -m benchmark

# Run security tests
hatch run pytest tests/test_security.py
```

## Troubleshooting

### Common Issues

1. **Build fails with missing dependencies**
   ```bash
   hatch run python -m pip install -e .[dev,test]
   ```

2. **Binary build fails**
   ```bash
   hatch run python -m pip install pyinstaller
   ```

3. **Tests fail due to missing API keys**
   ```bash
   # Set environment variables for testing
   export OPENAI_API_KEY="test-key"
   export ELEVENLABS_API_KEY="test-key"
   ```

4. **Permission denied on scripts**
   ```bash
   chmod +x scripts/*.sh
   ```

### Getting Help

- Check the [README.md](README.md) for usage instructions
- Look at the [GitHub Issues](https://github.com/twardoch/e11ocutionist/issues) for known problems
- Check the [GitHub Actions](https://github.com/twardoch/e11ocutionist/actions) for CI/CD status

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests: `./scripts/test.sh`
5. Submit a pull request

The CI pipeline will automatically test your changes on all supported platforms.