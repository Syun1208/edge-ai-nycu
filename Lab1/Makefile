# Override ARCH if -arch=native is not supported by your nvcc (needs CUDA >= 11.6)
#   e.g.  make ARCH=-arch=sm_75
NVCC  ?= nvcc
ARCH  ?= -arch=native
FLAGS  = -O3 -std=c++17 $(ARCH) -use_fast_math -lineinfo -Xptxas -v
LIBS   = -lcublas

# -use_fast_math is on for everyone: fast intrinsics, flush denormals to zero.
# -lineinfo lets Nsight Compute map metrics to source lines.
# -Xptxas -v prints registers / shared memory / spills per kernel. Read it!
# Compiler warnings are shown above that list; kernel names are demangled.

sgemm: src/main.cu src/kernels.cuh src/common.cuh $(wildcard src/kernels/*.cuh)
	$(NVCC) $(FLAGS) -o $@ src/main.cu $(LIBS) 2> ptxas_info.txt || (cat ptxas_info.txt; false)
	@grep -vE "^ptxas info|bytes stack frame|^ *$$" ptxas_info.txt || true
	@grep -E "Compiling entry|registers|spill" ptxas_info.txt | sed 's/ptxas info    : //' | cu++filt \
	  | awk '/Compiling entry.*l2_evict/ {skip = 2; next} skip > 0 {skip--; next} {print}' || true

# Instructor only: build against the reference solutions.
solution: src/main.cu instructor/solution_kernels.cuh src/common.cuh
	$(NVCC) $(FLAGS) -DUSE_SOLUTION -o sgemm_solution src/main.cu $(LIBS)

clean:
	rm -f sgemm sgemm_solution ptxas_info.txt

.PHONY: clean solution
