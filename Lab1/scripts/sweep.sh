#!/usr/bin/env bash
# Performance sweep over shapes -> results/sweep_<gpu>_<group>.csv
#   usage: scripts/sweep.sh [group] [binary]
#     group : all    every shape below (default; includes 8192^3, which takes
#                    several minutes with the naive kernel on slower GPUs)
#     binary: default ./sgemm
GROUP=${1:-all}
BIN=${2:-./sgemm}

case "$GROUP" in
  all)
    SHAPES=(
      # square
      "512 512 512" "1024 1024 1024" "2048 2048 2048" "4096 4096 4096" "8192 8192 8192"
      # not multiples of the tile size
      "1000 1000 1000" "3000 3000 3000"
      # GEMV / skinny
      "4096 1 4096" "8192 1 8192" "1 4096 4096" "1 8192 8192"
      "16 4096 4096" "4096 16 4096"
      # small K / large K
      "4096 4096 64" "64 64 65536"
    ) ;;
  *)
    echo "usage: scripts/sweep.sh [all] [binary]"; exit 1 ;;
esac

mkdir -p results
TMP=$(mktemp results/.sweep.XXXXXX)
echo "gpu,kernel,M,N,K,ms,gflops,gbps,pct_cublas,pct_peak_flops,pct_peak_bw,check" > "$TMP"
crashed=0
for s in "${SHAPES[@]}"; do
  echo ">> $s"
  $BIN all $s --csv --reps 10 | tee -a "$TMP"
  rc=${PIPESTATUS[0]}
  if [ $rc -ne 0 ]; then
    echo "!! harness exited with code $rc on '$s'; rows for this shape are incomplete"
    crashed=1
  fi
done

# Name the file after the GPU the harness actually ran on (honours
# CUDA_VISIBLE_DEVICES, unlike nvidia-smi).
GPU=$(awk -F, 'NR == 2 {print $1}' "$TMP" | tr ' /' '__')
OUT=results/sweep_${GPU:-unknown_gpu}_${GROUP}.csv
mv "$TMP" "$OUT"
echo "wrote $OUT"
[ $crashed -eq 0 ] || { echo "WARNING: some shapes crashed, see messages above"; exit 1; }
