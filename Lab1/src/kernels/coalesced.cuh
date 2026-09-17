#pragma once
// =============================================================================
//  coalesced.cuh  --  Kernel 2: coalesced global memory access (Part 2)
// =============================================================================
//  Row-major, FP32:  C (MxN) = alpha * A (MxK) @ B (KxN) + beta * C
//  YOU EDIT THIS FILE.
// =============================================================================
#include "../common.cuh"

// Same algorithm as the naive kernel. Change only how threads map to (row, col)
// so that consecutive threads in a warp touch consecutive addresses.
__global__ void sgemm_coalesced(int M, int N, int K, float alpha,
                                const float* __restrict__ A,
                                const float* __restrict__ B, float beta,
                                float* __restrict__ C) {
  // TODO(Part 2)
  const int col = blockIdx.x * blockDim.x + threadIdx.x;
  const int row = blockIdx.y * blockDim.y + threadIdx.y;
  if (row < M && col < N) {
    float acc = 0.f;
    for (int k = 0; k < K; ++k) {
      acc += A[(size_t)row * K + k] * B[(size_t)k * N + col];
    }
    const size_t idx = (size_t)row * N + col;
    C[idx] = alpha * acc + beta * C[idx];
  }
}

bool launch_coalesced(int M, int N, int K, float alpha, const float* A,
                      const float* B, float beta, float* C, cudaStream_t s) {
  // TODO(Part 2): choose grid/block, launch, and return true.
  dim3 block(32, 32);
  dim3 grid(ceil_div(N, 32), ceil_div(M, 32));
  sgemm_coalesced<<<grid, block, 0, s>>>(M, N, K, alpha, A, B, beta, C);
  return true;
}
