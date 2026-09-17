#pragma once
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

#define CUDA_CHECK(x)                                                          \
  do {                                                                         \
    cudaError_t err_ = (x);                                                    \
    if (err_ != cudaSuccess) {                                                 \
      fprintf(stderr, "CUDA error '%s' at %s:%d\n", cudaGetErrorString(err_),  \
              __FILE__, __LINE__);                                             \
      exit(1);                                                                 \
    }                                                                          \
  } while (0)

__host__ __device__ inline int ceil_div(int a, int b) { return (a + b - 1) / b; }

// All matrices are ROW-MAJOR:
//   A is M x K, B is K x N, C is M x N
//   C = alpha * A @ B + beta * C
//
// Every kernel gets a *launcher* with this signature. The launcher picks the
// grid/block configuration and launches on `stream`.
// Return false if the kernel is not implemented yet or does not support this
// (M, N, K) (e.g. a tiled kernel that requires M % BM == 0). The harness will
// then print "SKIPPED" instead of a result.
using Launcher = bool (*)(int M, int N, int K, float alpha, const float* A,
                          const float* B, float beta, float* C,
                          cudaStream_t stream);

struct KernelEntry {
  const char* name;
  Launcher launch;
};
