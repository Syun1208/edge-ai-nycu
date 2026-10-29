#!/usr/bin/env bash
cd "$(dirname "$0")"
mkdir -p logs checkpoints

pip install uv
uv pip install -r requirements.txt

export PYTHONPATH="$PWD/src"

compress() {
    python src/compress.py --case "$@" --output checkpoints/case$1.pt 2>&1 | tee logs/compress_case$1.log
}

evaluate() {
    python run_eval.py --case "$@" --model checkpoints/case$1.pt 2>&1 | tee logs/eval_case$1.log
}

report() {
    echo "Case $1"
    echo "Compression Ratio: $(awk '/^comp_rate:/ {print $2}' logs/eval_case$1.log)"
    echo "PPL: $(awk '/^ppl:/ {print $2}' logs/eval_case$1.log)"
    echo
}

compress 1 --rate 0.755 --wbits 4
compress 2 --rate 0.81 --wbits 3
compress 3 --rate 0.50
compress 4 --rate 0.85

evaluate 1 --rate 0.755 --wbits 4
evaluate 2 --rate 0.81 --wbits 3
evaluate 3 --rate 0.50
evaluate 4 --rate 0.85

echo "===== Final results ====="
report 1
report 2
report 3
report 4
