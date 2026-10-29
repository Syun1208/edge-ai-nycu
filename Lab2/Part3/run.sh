#!/usr/bin/env bash
set -euo pipefail

MODEL=Qwen/Qwen3-4B
THRESHOLDS=thresholds.json

cd "$(dirname "$0")"
python main.py evaluate \
    --model "${MODEL}" \
    --method adaptquant \
    --thresholds "${THRESHOLDS}" \
    "$@"
