#!/usr/bin/env bash
# Grader: correctness of Parts 2 and 3, and the Part 4 optimization chain.
#   usage: scripts/grade.sh [binary]          (run `make` first)
#   env:   NO_SANITIZER=1   skip the compute-sanitizer stage
#          GRADE_ROUNDS=3   timing rounds per chain entry (best median is kept)
#          GRADE_REPS=20    timed launches per round
# Pin an idle GPU first (export CUDA_VISIBLE_DEVICES=<id>); Part 4 compares
# timings, so a busy GPU gives meaningless results.
#
# Part 2 / 3 correctness: a kernel passes if it never FAILs or crashes on any
# shape in scripts/shapes.sh (both betas), never writes outside C, is clean
# under compute-sanitizer, and is not SKIPPED where it must support the shape.
#   Part 2: coalesced.            Part 3: smem16, smem32, and >= 6 reg2d_*
#   configurations, each correct and actually implemented (PASS on >= 1 shape).
#
# Part 4: kPart4Chain in src/kernels.cuh (printed by ./sgemm --chain). Each
# step i is timed against step i-1 at 4096^3: speedup >= 1.05 -> 5 points,
# otherwise 2.5 points (the report must explain it). An incorrect entry scores
# 0 and the next step is compared with the last correct entry. Capped at 10.
BIN=${1:-./sgemm}
source "$(dirname "$0")/shapes.sh"   # SHAPES, SAN_RUNS, REQUIRED_ALL
ROUNDS=${GRADE_ROUNDS:-3}
REPS=${GRADE_REPS:-20}
GRADE_SHAPE="4096 4096 4096"
THRESH=1.05

if ! LIST=$($BIN --list 2>&1); then
  echo "cannot run '$BIN --list' (did you run make?)"; echo "$LIST"; exit 1
fi
mapfile -t NAMES < <(echo "$LIST" | awk '$1 > 0 {print $2}')
mapfile -t CHAIN < <($BIN --chain)
declare -A BAD NBAD PASSES REGISTERED
for k in "${NAMES[@]}"; do REGISTERED[$k]=1; PASSES[$k]=0; done
is_required() { local r; for r in "${REQUIRED_ALL[@]}"; do [ "$r" = "$1" ] && return 0; done; return 1; }
note() { [ -z "${BAD[$1]}" ] && BAD[$1]=$2; NBAD[$1]=$((NBAD[$1] + 1)); }

# Parse --csv rows of one run: gpu,kernel,M,N,K,ms,...,status
parse_rows() {  # $1 = output, $2 = shape label
  local name st
  while IFS=, read -r _ name _ _ _ _ _ _ _ _ _ st; do
    [ -z "$name" ] || [ "$name" = cublas ] && continue
    case "$st" in
      PASS) PASSES[$name]=$((PASSES[$name] + 1)) ;;
      FAIL) note "$name" "FAIL on $2" ;;
      SKIPPED) is_required "$name" && note "$name" "SKIPPED on $2 (must support it)" ;;
    esac
  done <<< "$1"
}

# ---- correctness ------------------------------------------------------------
echo "== correctness: ${#SHAPES[@]} shapes x 2 betas"
for s in "${SHAPES[@]}"; do
  for beta in 0 0.5; do
    label="'$s' beta=$beta"
    out=$($BIN all $s --beta $beta --alpha 1.5 --reps 1 --warmup 0 --csv 2>/dev/null)
    rc=$?
    parse_rows "$out" "$label"
    if [ $rc -ne 0 ]; then
      # The harness stops at the first CUDA error: the kernel after the last
      # printed row crashed. Run the kernels after it one by one.
      done_rows=$(echo "$out" | grep -c ,)
      culprit=${NAMES[$((done_rows - 1))]}
      note "${culprit:-?}" "crashed on $label"
      for ((i = done_rows; i < ${#NAMES[@]}; i++)); do
        k=${NAMES[$i]}
        out=$($BIN "$k" $s --beta $beta --alpha 1.5 --reps 1 --warmup 0 --csv 2>/dev/null) \
          || note "$k" "crashed on $label"
        parse_rows "$out" "$label"
      done
    fi
  done
  echo "   $s done"
done

# ---- compute-sanitizer, one kernel at a time (exact attribution) ------------
SAN_KERNELS=(coalesced smem16 smem32)
for k in "${NAMES[@]}"; do [[ $k == reg2d_* ]] && SAN_KERNELS+=("$k"); done
for k in "${CHAIN[@]}"; do [ -n "${REGISTERED[$k]}" ] && SAN_KERNELS+=("$k"); done
mapfile -t SAN_KERNELS < <(printf '%s\n' "${SAN_KERNELS[@]}" | awk '!seen[$0]++')
if [ -n "$NO_SANITIZER" ]; then
  echo "== sanitizer: skipped (NO_SANITIZER is set)"
elif ! command -v compute-sanitizer > /dev/null; then
  echo "== sanitizer: compute-sanitizer not found on PATH, stage skipped"
else
  echo "== sanitizer: ${#SAN_KERNELS[@]} kernels x ${#SAN_RUNS[@]} runs"
  for k in "${SAN_KERNELS[@]}"; do
    [ -n "${REGISTERED[$k]}" ] || continue
    for r in "${SAN_RUNS[@]}"; do
      read -r tool M N K <<< "$r"
      compute-sanitizer --tool $tool --error-exitcode 99 --print-limit 5 \
        $BIN "$k" $M $N $K --profile > /dev/null 2>&1 \
        || note "$k" "$tool errors on $M $N $K"
    done
  done
fi

verdict() {  # kernel -> "PASS" or "FAIL (...)"
  if [ -z "${REGISTERED[$1]}" ]; then echo "FAIL (not registered)"
  elif [ -n "${BAD[$1]}" ]; then
    local more=$((NBAD[$1] - 1))
    echo "FAIL (${BAD[$1]}$([ $more -gt 0 ] && echo ", +$more more"))"
  elif [ "${PASSES[$1]}" -eq 0 ]; then echo "FAIL (not implemented: SKIPPED on every shape)"
  else echo "PASS"; fi
}

# ---- Part 2 / Part 3 ----------------------------------------------------------
echo
echo "== Part 2"
printf "   %-26s %s\n" coalesced "$(verdict coalesced)"
echo "== Part 3"
for k in smem16 smem32; do printf "   %-26s %s\n" "$k" "$(verdict $k)"; done
nreg=0; nreg_ok=0
for k in "${NAMES[@]}"; do
  [[ $k == reg2d_* ]] || continue
  v=$(verdict "$k"); nreg=$((nreg + 1)); [ "$v" = PASS ] && nreg_ok=$((nreg_ok + 1))
  printf "   %-26s %s\n" "$k" "$v"
done
printf "   %-26s %s\n" "reg2d configurations" \
  "$nreg_ok correct of $nreg registered (need >= 6): $([ $nreg_ok -ge 6 ] && echo PASS || echo FAIL)"

others=0
for k in "${NAMES[@]}"; do
  [[ $k == reg2d_* || $k == coalesced || $k == smem16 || $k == smem32 ]] && continue
  [ -n "${BAD[$k]}" ] || continue
  [ $others -eq 0 ] && echo "== Other kernels with problems (rule 3: every kernel must pass)"
  printf "   %-26s %s\n" "$k" "$(verdict "$k")"; others=1
done

# ---- Part 4 -------------------------------------------------------------------
echo "== Part 4 (chain at $GRADE_SHAPE, best of $ROUNDS x median of $REPS)"
if [ ${#CHAIN[@]} -lt 2 ]; then
  echo "   kPart4Chain has ${#CHAIN[@]} entries; need a starting kernel plus >= 1 optimization"
  echo "   Part 4 points: 0 / 10"
  exit 0
fi
declare -A SEEN BEST BADK  # BADK: kernel-level problems; SKIPI: per-entry problems
SKIPI=()
for ((i = 0; i < ${#CHAIN[@]}; i++)); do
  k=${CHAIN[$i]}
  if [ -z "${REGISTERED[$k]}" ]; then SKIPI[$i]="not registered in kKernels[]"
  elif [ -n "${SEEN[$k]}" ]; then SKIPI[$i]="already listed earlier in the chain"
  elif [ "$(verdict "$k")" != PASS ]; then BADK[$k]="incorrect: $(verdict "$k")"
  fi
  SEEN[$k]=1
done
for ((r = 1; r <= ROUNDS; r++)); do
  for k in "${!SEEN[@]}"; do
    [ -n "${REGISTERED[$k]}" ] && [ -z "${BADK[$k]}" ] || continue
    row=$($BIN "$k" $GRADE_SHAPE --csv --reps $REPS $([ $r -gt 1 ] && echo --no-check) 2>/dev/null)
    ms=$(echo "$row" | awk -F, '{print $6}'); st=$(echo "$row" | awk -F, '{print $12}')
    if [ $r -eq 1 ] && [ "$st" != PASS ]; then BADK[$k]="not PASS at $GRADE_SHAPE (${st:-crashed})"; continue; fi
    [ -z "${BEST[$k]}" ] || awk -v a="$ms" -v b="${BEST[$k]}" 'BEGIN{exit !(a < b)}' && BEST[$k]=$ms
  done
done
pts=0; prev=""
for ((i = 0; i < ${#CHAIN[@]}; i++)); do
  k=${CHAIN[$i]}
  why=${SKIPI[$i]:-${BADK[$k]}}
  if [ -n "$why" ]; then
    printf "   %d %-24s 0 pts   (%s)\n" $i "$k" "$why"; continue
  fi
  if [ -z "$prev" ]; then
    printf "   %d %-24s %8.3f ms  starting point\n" $i "$k" "${BEST[$k]}"
  else
    sp=$(awk -v a="${BEST[$prev]}" -v b="${BEST[$k]}" 'BEGIN{printf "%.3f", a / b}')
    if awk -v s="$sp" -v t=$THRESH 'BEGIN{exit !(s >= t)}'; then p=5; tag=successful
    else p=2.5; tag="failed (needs an explanation in the report)"; fi
    pts=$(awk -v a=$pts -v b=$p 'BEGIN{print a + b}')
    printf "   %d %-24s %8.3f ms  %sx vs %s: %s, %s pts\n" $i "$k" "${BEST[$k]}" "$sp" "$prev" "$tag" $p
  fi
  prev=$k
done
pts=$(awk -v a=$pts 'BEGIN{print (a > 10 ? 10 : a)}')
echo "   Part 4 points: $pts / 10"
