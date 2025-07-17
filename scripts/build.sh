#!/bin/bash

# Build script for e11ocutionist
# This script builds the package and ensures all dependencies are resolved

set -e

echo "🏗️  Building e11ocutionist..."

# Change to project root
cd "$(dirname "$0")/.."

# Clean previous builds
echo "🧹 Cleaning previous builds..."
rm -rf dist/ build/ *.egg-info/

# Build the package
echo "📦 Building package..."
hatch build

echo "✅ Build completed successfully!"
echo "📋 Build artifacts:"
ls -la dist/