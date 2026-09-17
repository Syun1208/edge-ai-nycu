#!/usr/bin/env bash
# Correctness sweep. This is what the graders run (on THEIR GPU).
# Speed is not graded. A kernel FAILS the sweep if, on any shape, it
#   * produces a wrong result (FAIL),
#   * crashes the harness (illegal address, bad launch config, ...), or
#   * returns SKIPPED although it is required to support that shape (below),
#   * writes outside C (the harness checks guard regions around every buffer), or
#   * shows out-of-bounds accesses or shared-memory races under compute-sanitizer.
# Kernels not listed below (reg2d variants, your own Part 4 kernels) may
# return SKIPPED for shapes they do not support.
#   usage: scripts/check_all.sh [binary]
BIN=${1:-./sgemm}


source "$(dirname "$0")/shapes.sh"  # SHAPES, SAN_RUNS, REQUIRED_ALL

if ! LIST=$($BIN --list 2>&1); then
  echo "cannot run '$BIN --list' (did you run make?)"; echo "$LIST"; exit 1
fi
NKERNELS=$(echo "$LIST" | grep -cE '^ *[0-9]+ ')
registered() { echo "$LIST" | awk '{print $2}' | grep -qxF "$1"; }

fail=0
for k in "${REQUIRED_ALL[@]}"; do
  if ! registered "$k"; then
    echo "MISSING: kernel '$k' is not registered in kKernels[]"; fail=1
  fi
done

for s in "${SHAPES[@]}"; do
  read -r M N K <<< "$s"
  for beta in 0 0.5; do
    out=$($BIN all $s --beta $beta --alpha 1.5 --reps 1 --warmup 0 --csv)
    rc=$?
    echo "$out" | awk -F, -v b=$beta '{printf "%-22s %6s %6s %6s beta=%-4s %s\n",$2,$3,$4,$5,b,$12}'
    if [ $rc -ne 0 ]; then
      # The harness aborts on the first CUDA error, so the kernel after the
      # last printed row is the one that crashed; later kernels never ran.
      done_rows=$(echo "$out" | grep -c ,)
      culprit=$(echo "$LIST" | awk -v id=$done_rows '$1 == id {print $2}')
      echo "CRASH: '${culprit:-?}' crashed the harness (exit $rc) on '$s'" \
           "beta=$beta; $((NKERNELS - done_rows - 1)) later kernel(s) did not run"
      fail=1
    fi
    echo "$out" | grep -q ",FAIL$" && fail=1
    for k in "${REQUIRED_ALL[@]}"; do
      if echo "$out" | grep -q "^[^,]*,$k,.*,SKIPPED$"; then
        echo "REQUIRED: '$k' was SKIPPED on '$s' but must support it"; fail=1
      fi
    done
  done
done
# compute-sanitizer stage: out-of-bounds accesses (memcheck) and shared-memory
# races (racecheck) on small shapes. --profile launches every kernel once with
# unguarded buffers, so the sanitizer sees the real bounds. Set NO_SANITIZER=1
# to skip this stage while iterating (graders do not skip it).
if [ -n "$NO_SANITIZER" ]; then
  echo "SANITIZER: skipped (NO_SANITIZER is set)"
elif ! command -v compute-sanitizer > /dev/null; then
  echo "SANITIZER: compute-sanitizer not found on PATH, stage skipped" \
       "(it ships with the CUDA toolkit: \$CUDA_HOME/bin)"
else
  for r in "${SAN_RUNS[@]}"; do
    read -r tool M N K <<< "$r"
    san=$(compute-sanitizer --tool $tool --error-exitcode 99 --print-limit 20 \
          $BIN all $M $N $K --profile 2>&1)
    rc=$?
    if [ $rc -eq 0 ]; then
      echo "SANITIZER $tool $M $N $K: clean"
    else
      kernels=$(echo "$san" | sed -nE 's/.* in [^ ]*:[0-9]+:(void )?([A-Za-z_][A-Za-z0-9_:]*(<[^>]*>)?)\(.*/\2/p' \
                | sort -u | paste -sd, -)
      echo "SANITIZER $tool $M $N $K: ERRORS in ${kernels:-?} (exit $rc);" \
           "rerun: compute-sanitizer --tool $tool $BIN <kernel> $M $N $K --profile"
      fail=1
    fi
  done
fi

[ $fail -eq 0 ] && echo "ALL PASSED" || { echo "SOME KERNELS FAILED"; exit 1; }
