# Shapes shared by scripts/check_all.sh and scripts/grade.sh (sourced, not run).

# Correctness shapes: every kernel runs on each, with beta = 0 and beta = 0.5.
SHAPES=(
  "128 128 128"   "1024 1024 1024" "4096 4096 4096"
  "1000 1000 1000" "1023 257 511"  "1 1 1" "7 13 3"
  "4096 1 4096"   "1 4096 4096"    "8192 1 8192"
  "16 4096 4096"  "4096 4096 64"   "64 64 65536"
  "1000 1 1001"   "4097 1 4099"
)

# compute-sanitizer runs: out-of-bounds accesses (memcheck) and shared-memory
# races (racecheck). 256^3 gives the 128x128 reg2d tiles more than one block.
SAN_RUNS=(
  "memcheck 1023 257 511" "memcheck 7 13 3" "memcheck 1000 1 1001" "memcheck 128 128 128"
  "memcheck 256 256 256"
  "racecheck 128 128 128" "racecheck 1023 257 511" "racecheck 256 256 256"
)

# Kernels that must PASS on every shape (others may return SKIPPED).
REQUIRED_ALL=(naive coalesced smem16 smem32)
