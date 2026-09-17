#pragma once
// =============================================================================
//  smem.cuh  --  Kernel 3: shared-memory tiling, one output per thread (Part 3a)
// =============================================================================
//  Row-major, FP32:  C (MxN) = alpha * A (MxK) @ B (KxN) + beta * C
//  YOU EDIT THIS FILE.
// =============================================================================
#include "../common.cuh"

// Each block computes a TILE x TILE tile of C. Loop over K in steps of TILE:
// cooperatively load a tile of A and a tile of B into __shared__ memory,
// __syncthreads(), accumulate, __syncthreads().
// Must handle ANY M, N, K (zero-pad out-of-range tile elements).
template <int TILE>
__global__ void sgemm_smem(int M, int N, int K, float alpha,
                           const float* __restrict__ A,
                           const float* __restrict__ B, float beta,
                           float* __restrict__ C) {
  // TODO(Part 3a)
  __shared__ float As[TILE][TILE];
  __shared__ float Bs[TILE][TILE];
  const int tx = threadIdx.x;
  const int ty = threadIdx.y;
  const int row = blockIdx.y * TILE + ty;
  const int col = blockIdx.x * TILE + tx;
  float acc = 0.f;
  for (int k0 = 0; k0 < K; k0 += TILE) {
    const int a_col = k0 + tx;
    const int b_row = k0 + ty;
    As[ty][tx] = (row < M && a_col < K) ? A[(size_t)row * K + a_col] : 0.f;
    Bs[ty][tx] = (b_row < K && col < N) ? B[(size_t)b_row * N + col] : 0.f;
    __syncthreads();
    for (int k = 0; k < TILE; ++k) {
      acc += As[ty][k] * Bs[k][tx];
    }
    __syncthreads();
  }
  if (row < M && col < N) {
    const size_t idx = (size_t)row * N + col;
    C[idx] = alpha * acc + beta * C[idx];
  }
}

template <int TILE>
bool launch_smem(int M, int N, int K, float alpha, const float* A,
                 const float* B, float beta, float* C, cudaStream_t s) {
  // TODO(Part 3a)
  dim3 block(TILE, TILE);
  dim3 grid(ceil_div(N, TILE), ceil_div(M, TILE));
  sgemm_smem<TILE><<<grid, block, 0, s>>>(M, N, K, alpha, A, B, beta, C);
  return true;
}
