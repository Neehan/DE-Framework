#!/usr/bin/env bash
# Bootstrap the current harness, then delegate dataset and gateway setup to Python.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

exec uv run --locked python -m scripts.setup
