#!/usr/bin/env bash
# =========================================================
# build.sh - Build and publish XRO package
# Usage:
#   ./build.sh local     # Build and install locally
#   ./build.sh test      # Upload to TestPyPI
#   ./build.sh pypi      # Upload to PyPI
# =========================================================

set -e  # exit on error

MODE=${1:-local}

echo "🚀 Starting build in mode: $MODE"

# Clean old builds
rm -rf build dist *.egg-info

# Build wheel + source distribution
python -m build

if [ "$MODE" = "local" ]; then
    echo "📦 Installing package locally (editable mode)..."
    pip install -e .
    echo "✅ Local install complete."
elif [ "$MODE" = "test" ]; then
    echo "📤 Uploading to TestPyPI..."
    twine upload --repository testpypi dist/*
    echo "✅ Uploaded to TestPyPI. Install with:"
    echo "    pip install -i https://test.pypi.org/simple XRO"
elif [ "$MODE" = "pypi" ]; then
    echo "📤 Uploading to PyPI..."
    twine upload dist/*
    echo "✅ Uploaded to PyPI. Install with:"
    echo "    pip install XRO"
else
    echo "❌ Unknown mode: $MODE"
    echo "Usage: ./build.sh [local|test|pypi]"
    exit 1
fi
