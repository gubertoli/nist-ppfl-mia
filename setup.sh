#!/bin/bash
set -e

echo "Downloading NIST dataset..."
wget -qnc https://github.com/usnistgov/genomics_ppfl/releases/download/Problem1/NIST_PPFL_problem1_202503.zip

echo "Unzipping dataset..."
unzip -o NIST_PPFL_problem1_202503.zip

echo "Checking for 'uv' installation..."
if ! command -v uv &> /dev/null; then
    echo "uv could not be found, installing..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi

uv sync
echo "Setup complete! Run: source .venv/bin/activate"
