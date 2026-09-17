#pragma once
// =============================================================================
//  naive.cuh  --  Kernel 1: naive (GIVEN, do not modify: it is your baseline)
// =============================================================================
//  Row-major, FP32:  C (MxN) = alpha * A (MxK) @ B (KxN) + beta * C
//  Used in Part 1.
// =============================================================================
#include "../common.cuh"

// One thread computes one element of C.
// NOTE the index mapping: threadIdx.x -> ROW. Think about which addresses the
// 32 threads of one warp touch at the same moment.
__global__ void sgemm_naive(int M, int N, int K, float alpha,
                            const float* __restrict__ A,
                            const float* __restrict__ B, float beta,
                            float* __restrict__ C) {
  const int row = blockIdx.x * blockDim.x + threadIdx.x;
  const int col = blockIdx.y * blockDim.y + threadIdx.y;
  if (row < M && col < N) {
    float acc = 0.f;
    for (int k = 0; k < K; ++k) {
      acc += A[(size_t)row * K + k] * B[(size_t)k * N + col];
    }
    const size_t idx = (size_t)row * N + col;
    C[idx] = alpha * acc + beta * C[idx];
  }
}

bool launch_naive(int M, int N, int K, float alpha, const float* A,
                  const float* B, float beta, float* C, cudaStream_t s) {
  dim3 block(32, 32);
  dim3 grid(ceil_div(M, 32), ceil_div(N, 32));
  sgemm_naive<<<grid, block, 0, s>>>(M, N, K, alpha, A, B, beta, C);
  return true;
}
