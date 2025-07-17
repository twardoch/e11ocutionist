#!/bin/bash

# Test script for e11ocutionist
# This script runs the complete test suite with coverage

set -e

echo "🧪 Running e11ocutionist tests..."

# Change to project root
cd "$(dirname "$0")/.."

# Run linting
echo "🔍 Running linting..."
hatch run lint

# Run type checking
echo "🔍 Running type checking..."
hatch run typecheck

# Run tests with coverage
echo "🧪 Running tests with coverage..."
hatch run test-cov

echo "✅ All tests passed!"