#!/bin/bash

# Build all artifacts script for e11ocutionist
# This script builds Python packages and binaries for all supported platforms

set -e

echo "🚀 Building all e11ocutionist artifacts..."

# Change to project root
cd "$(dirname "$0")/.."

# Function to display help
show_help() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  -h, --help          Show this help message"
    echo "  --skip-tests        Skip running tests"
    echo "  --skip-binaries     Skip building binaries"
    echo "  --skip-package      Skip building Python package"
    echo "  --output-dir DIR    Output directory (default: dist/)"
    echo ""
    echo "Examples:"
    echo "  $0                  Build everything"
    echo "  $0 --skip-tests     Build without running tests"
    echo "  $0 --skip-binaries  Build only Python package"
}

# Parse command line arguments
SKIP_TESTS=false
SKIP_BINARIES=false
SKIP_PACKAGE=false
OUTPUT_DIR="dist"

while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            show_help
            exit 0
            ;;
        --skip-tests)
            SKIP_TESTS=true
            shift
            ;;
        --skip-binaries)
            SKIP_BINARIES=true
            shift
            ;;
        --skip-package)
            SKIP_PACKAGE=true
            shift
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
done

# Check if we have the required dependencies
if ! command -v hatch &> /dev/null; then
    echo "❌ Hatch is required but not installed. Please install it with: pip install hatch"
    exit 1
fi

# Clean previous builds
echo "🧹 Cleaning previous builds..."
rm -rf "$OUTPUT_DIR"/ build/ *.egg-info/

# Run tests if not skipped
if [[ "$SKIP_TESTS" == "false" ]]; then
    echo "🧪 Running tests..."
    ./scripts/test.sh
else
    echo "⏭️  Skipping tests..."
fi

# Build Python package if not skipped
if [[ "$SKIP_PACKAGE" == "false" ]]; then
    echo "📦 Building Python package..."
    ./scripts/build.sh
else
    echo "⏭️  Skipping Python package build..."
fi

# Build binaries if not skipped
if [[ "$SKIP_BINARIES" == "false" ]]; then
    echo "🔧 Building binaries..."
    
    # Build current platform binary
    ./scripts/build-binary.sh --onefile --output "$OUTPUT_DIR"
    
    # Note: Cross-platform builds would require additional setup
    echo "ℹ️  Cross-platform builds are handled by GitHub Actions"
else
    echo "⏭️  Skipping binary builds..."
fi

# Show results
echo ""
echo "✅ Build completed successfully!"
echo "📋 Build artifacts in $OUTPUT_DIR/:"
if [[ -d "$OUTPUT_DIR" ]]; then
    ls -la "$OUTPUT_DIR"/
else
    echo "  No artifacts directory found"
fi

echo ""
echo "🎯 Next steps:"
echo "  • Test the built artifacts"
echo "  • Run './scripts/release.sh -v vX.Y.Z' to create a release"
echo "  • Push a git tag to trigger GitHub Actions for multi-platform builds"