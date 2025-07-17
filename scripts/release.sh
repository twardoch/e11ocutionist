#!/bin/bash

# Release script for e11ocutionist
# This script handles versioning, testing, building, and releasing

set -e

# Change to project root
cd "$(dirname "$0")/.."

# Function to display help
show_help() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  -h, --help          Show this help message"
    echo "  -v, --version TAG   Create a release for the specified version tag"
    echo "  --dry-run           Perform a dry run without making changes"
    echo ""
    echo "Examples:"
    echo "  $0 -v v1.0.0        Create a release for version v1.0.0"
    echo "  $0 --dry-run        Test the release process without making changes"
}

# Parse command line arguments
VERSION=""
DRY_RUN=false

while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            show_help
            exit 0
            ;;
        -v|--version)
            VERSION="$2"
            shift 2
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        *)
            echo "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
done

# Validate version format
if [[ -n "$VERSION" ]]; then
    if [[ ! "$VERSION" =~ ^v[0-9]+\.[0-9]+\.[0-9]+(-[a-zA-Z0-9]+)?$ ]]; then
        echo "❌ Invalid version format: $VERSION"
        echo "Expected format: v1.0.0 or v1.0.0-alpha"
        exit 1
    fi
fi

echo "🚀 Starting release process..."

# Check if we're in a git repository
if ! git rev-parse --is-inside-work-tree > /dev/null 2>&1; then
    echo "❌ Not in a git repository"
    exit 1
fi

# Check if working directory is clean
if [[ -n $(git status --porcelain) ]]; then
    echo "❌ Working directory is not clean. Please commit or stash changes."
    git status --short
    exit 1
fi

# Update to latest main branch
echo "🔄 Updating to latest main branch..."
if [[ "$DRY_RUN" == "false" ]]; then
    git checkout main
    git pull origin main
fi

# Run tests
echo "🧪 Running full test suite..."
./scripts/test.sh

# Build package
echo "📦 Building package..."
./scripts/build.sh

# Create and push tag if version is specified
if [[ -n "$VERSION" ]]; then
    echo "🏷️  Creating tag: $VERSION"
    if [[ "$DRY_RUN" == "false" ]]; then
        git tag -a "$VERSION" -m "Release $VERSION"
        git push origin "$VERSION"
        echo "✅ Tag $VERSION created and pushed"
    else
        echo "🔍 [DRY RUN] Would create tag: $VERSION"
    fi
else
    echo "ℹ️  No version specified, skipping tag creation"
fi

# Check if we should publish
if [[ "$DRY_RUN" == "false" ]]; then
    echo "📤 Package ready for release!"
    echo "🎯 To publish to PyPI:"
    echo "   hatch publish"
    echo ""
    echo "🎯 To publish to Test PyPI:"
    echo "   hatch publish -r test"
else
    echo "🔍 [DRY RUN] Release process completed successfully"
fi

echo "✅ Release process completed!"