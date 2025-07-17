#!/bin/bash

# Build binary script for e11ocutionist
# This script builds standalone executables for different platforms

set -e

echo "🔧 Building e11ocutionist binary..."

# Change to project root
cd "$(dirname "$0")/.."

# Function to display help
show_help() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  -h, --help          Show this help message"
    echo "  -o, --output DIR    Output directory for binaries (default: dist/)"
    echo "  -n, --name NAME     Binary name (default: e11ocutionist)"
    echo "  --onefile           Create single file executable"
    echo "  --debug             Enable debug mode"
    echo ""
    echo "Examples:"
    echo "  $0                  Build binary with default settings"
    echo "  $0 --onefile        Build single file executable"
    echo "  $0 -o build/        Build binary to build/ directory"
}

# Parse command line arguments
OUTPUT_DIR="dist"
BINARY_NAME="e11ocutionist"
ONEFILE=""
DEBUG=""

while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            show_help
            exit 0
            ;;
        -o|--output)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        -n|--name)
            BINARY_NAME="$2"
            shift 2
            ;;
        --onefile)
            ONEFILE="--onefile"
            shift
            ;;
        --debug)
            DEBUG="--debug"
            shift
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

# Install dependencies
echo "📦 Installing dependencies..."
hatch run python -m pip install -e .
hatch run python -m pip install pyinstaller

# Determine platform-specific settings
PLATFORM=$(uname -s)
case $PLATFORM in
    Linux*)
        PLATFORM_NAME="linux"
        BINARY_EXT=""
        ;;
    Darwin*)
        PLATFORM_NAME="macos"
        BINARY_EXT=""
        ;;
    CYGWIN*|MINGW*|MSYS*)
        PLATFORM_NAME="windows"
        BINARY_EXT=".exe"
        ;;
    *)
        PLATFORM_NAME="unknown"
        BINARY_EXT=""
        ;;
esac

# Architecture detection
ARCH=$(uname -m)
case $ARCH in
    x86_64|amd64)
        ARCH_NAME="x86_64"
        ;;
    aarch64|arm64)
        ARCH_NAME="arm64"
        ;;
    *)
        ARCH_NAME=$ARCH
        ;;
esac

# Set binary name with platform suffix
FULL_BINARY_NAME="${BINARY_NAME}-${PLATFORM_NAME}-${ARCH_NAME}${BINARY_EXT}"

# Clean previous builds
echo "🧹 Cleaning previous builds..."
rm -rf build/ "$OUTPUT_DIR"/*.spec

# Build the binary
echo "🔨 Building binary: $FULL_BINARY_NAME"

# PyInstaller command
PYINSTALLER_CMD="hatch run pyinstaller"

# Add options
if [[ -n "$ONEFILE" ]]; then
    PYINSTALLER_CMD="$PYINSTALLER_CMD $ONEFILE"
fi

if [[ -n "$DEBUG" ]]; then
    PYINSTALLER_CMD="$PYINSTALLER_CMD $DEBUG"
fi

# Platform-specific data additions
if [[ "$PLATFORM_NAME" == "windows" ]]; then
    DATA_ARG="--add-data src/e11ocutionist;e11ocutionist"
else
    DATA_ARG="--add-data src/e11ocutionist:e11ocutionist"
fi

# Build command
$PYINSTALLER_CMD \
    --name "$FULL_BINARY_NAME" \
    --distpath "$OUTPUT_DIR" \
    "$DATA_ARG" \
    --hidden-import e11ocutionist \
    --hidden-import e11ocutionist.__main__ \
    --hidden-import lxml \
    --hidden-import lxml.etree \
    --hidden-import litellm \
    --hidden-import elevenlabs \
    --hidden-import loguru \
    --hidden-import fire \
    --hidden-import tenacity \
    --hidden-import backoff \
    --hidden-import tiktoken \
    --hidden-import rich \
    --collect-all e11ocutionist \
    src/e11ocutionist/__main__.py

# Test the binary
echo "🧪 Testing binary..."
if [[ "$ONEFILE" == "--onefile" ]]; then
    BINARY_PATH="$OUTPUT_DIR/$FULL_BINARY_NAME"
else
    BINARY_PATH="$OUTPUT_DIR/$FULL_BINARY_NAME/$FULL_BINARY_NAME"
fi

if [[ -f "$BINARY_PATH" ]]; then
    "$BINARY_PATH" --help > /dev/null 2>&1
    if [[ $? -eq 0 ]]; then
        echo "✅ Binary test passed!"
    else
        echo "⚠️  Binary test failed, but binary was created"
    fi
else
    echo "❌ Binary not found at expected location: $BINARY_PATH"
    exit 1
fi

# Show results
echo ""
echo "✅ Binary build completed successfully!"
echo "📍 Binary location: $BINARY_PATH"
echo "💾 Binary size: $(du -h "$BINARY_PATH" | cut -f1)"
echo ""
echo "🎯 To test the binary:"
echo "   $BINARY_PATH --help"