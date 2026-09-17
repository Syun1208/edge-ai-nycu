#pragma once
// =============================================================================
//  reg2d.cuh  --  Kernel 4: shared-memory tiling + 2D register blocking (Part 3b)
// =============================================================================
//  Row-major, FP32:  C (MxN) = alpha * A (MxK) @ B (KxN) + beta * C
//  YOU EDIT THIS FILE.
// =============================================================================
#include "../common.cuh"

// A block of (BM/TM)*(BN/TN) threads computes a BM x BN tile of C.
// Each thread computes a TM x TN sub-tile held in registers.
// Per K-step of size BK: load As (BM x BK) and Bs (BK x BN) into shared
// memory, then for each k in [0, BK): load TM values of A and TN values of B
// into registers and do TM*TN FMAs (an outer product). The outer-product loop
// is GIVEN below; you write the shared-memory loads and the write-back to C.
//
// Block -> tile mapping (see the launcher's grid): blockIdx.x selects the
// BN-wide column tile of C (columns blockIdx.x*BN ..), blockIdx.y the BM-tall
// row tile (rows blockIdx.y*BM ..). threadIdx.x is 1D, 0 .. NT-1.
//
// You may assume M % BM == 0, N % BN == 0, K % BK == 0 (the launcher below
// rejects other sizes).
template <int BM, int BN, int BK, int TM, int TN>
__global__ void __launch_bounds__((BM / TM) * (BN / TN))
    sgemm_reg2d(int M, int N, int K, float alpha, const float* __restrict__ A,
                const float* __restrict__ B, float beta,
                float* __restrict__ C) {
  constexpr int NT = (BM / TM) * (BN / TN);  // threads per block
  // As is stored TRANSPOSED: As[k][m] holds A[m][k] of the current tile.
  __shared__ float As[BK][BM];
  __shared__ float Bs[BK][BN];
  const int tid = threadIdx.x;
  // This thread computes rows tr*TM .. tr*TM+TM-1 and
  // columns tc*TN .. tc*TN+TN-1 of the block's BM x BN tile.
  const int tr = tid / (BN / TN), tc = tid % (BN / TN);
  float acc[TM][TN] = {};
  float ra[TM], rb[TN];
  for (int k0 = 0; k0 < K; k0 += BK) {
    // TODO(Part 3b): load the BM x BK tile of A into As (transposed) and the
    // BK x BN tile of B into Bs. There are NT threads but BM*BK and BK*BN
    // elements, so each thread must load several elements.
    for (int i = tid; i < BM * BK; i += NT) {
      const int m = i / BK;
      const int k = i % BK;
      As[k][m] = A[(size_t)(blockIdx.y * BM + m) * K + k0 + k];
    }
    for (int i = tid; i < BK * BN; i += NT) {
      const int k = i / BN;
      const int n = i % BN;
      Bs[k][n] = B[(size_t)(k0 + k) * N + blockIdx.x * BN + n];
    }

    __syncthreads();
    // GIVEN: outer product. Per k, each thread loads TM values of A and TN
    // values of B into registers and does TM*TN FMAs with them.
    for (int k = 0; k < BK; ++k) {
      for (int i = 0; i < TM; ++i) ra[i] = As[k][tr * TM + i];
      for (int j = 0; j < TN; ++j) rb[j] = Bs[k][tc * TN + j];
      for (int i = 0; i < TM; ++i)
        for (int j = 0; j < TN; ++j) acc[i][j] += ra[i] * rb[j];
    }
    __syncthreads();
  }
  // TODO(Part 3b): write acc[i][j] to C:  C = alpha * acc + beta * C
  for (int i = 0; i < TM; ++i) {
    const size_t row = (size_t)blockIdx.y * BM + tr * TM + i;
    for (int j = 0; j < TN; ++j) {
      const size_t idx = row * N + (size_t)blockIdx.x * BN + tc * TN + j;
      C[idx] = alpha * acc[i][j] + beta * C[idx];
    }
  }
}

// Set to true once your sgemm_reg2d works.
constexpr bool kReg2DImplemented = true;

template <int BM, int BN, int BK, int TM, int TN>
bool launch_reg2d(int M, int N, int K, float alpha, const float* A,
                  const float* B, float beta, float* C, cudaStream_t s) {
  static_assert(BM % TM == 0 && BN % TN == 0, "tile must divide block tile");
  constexpr int kThreads = (BM / TM) * (BN / TN);
  static_assert(kThreads <= 1024, "too many threads per block");
  if (!kReg2DImplemented) return false;
  if (M % BM || N % BN || K % BK) return false;
  dim3 grid(N / BN, M / BM);
  sgemm_reg2d<BM, BN, BK, TM, TN>
      <<<grid, kThreads, 0, s>>>(M, N, K, alpha, A, B, beta, C);
  return true;
}
