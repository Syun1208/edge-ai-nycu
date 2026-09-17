#pragma once
// =============================================================================
//  custom.cuh  --  Your own optimized kernels (Part 4)
// =============================================================================
//  Row-major, FP32:  C (MxN) = alpha * A (MxK) @ B (KxN) + beta * C
//  YOU EDIT THIS FILE.
// =============================================================================
#include "../common.cuh"

// e.g. a vectorized / double-buffered / swizzled variant of the reg2d kernel,
// a shape-dispatching launcher, ... Register each one in src/kernels.cuh.

template <int BM, int BN, int BK, int TM, int TN>
struct TileLoad {
  static constexpr int kThreads = (BM / TM) * (BN / TN);
  static constexpr int kA = BM * BK / 4 / kThreads;
  static constexpr int kB = BK * BN / 4 / kThreads;
  static_assert(kA * kThreads * 4 == BM * BK, "A tile must split evenly");
  static_assert(kB * kThreads * 4 == BK * BN, "B tile must split evenly");
  float4 a[kA];
  float4 b[kB];
};

template <int BM, int BN, int BK, int TM, int TN>
__device__ __forceinline__ void load_global_vec4(
    const float* __restrict__ A, const float* __restrict__ B, int N, int K,
    int k0, TileLoad<BM, BN, BK, TM, TN>& regs) {
  constexpr int NT = TileLoad<BM, BN, BK, TM, TN>::kThreads;
  const int tid = threadIdx.x;
#pragma unroll
  for (int r = 0; r < TileLoad<BM, BN, BK, TM, TN>::kA; ++r) {
    const int i = tid + r * NT;
    const int m = i / (BK / 4);
    const int k = (i % (BK / 4)) * 4;
    regs.a[r] = *reinterpret_cast<const float4*>(
        &A[(size_t)(blockIdx.y * BM + m) * K + k0 + k]);
  }
#pragma unroll
  for (int r = 0; r < TileLoad<BM, BN, BK, TM, TN>::kB; ++r) {
    const int i = tid + r * NT;
    const int k = i / (BN / 4);
    const int n = (i % (BN / 4)) * 4;
    regs.b[r] = *reinterpret_cast<const float4*>(
        &B[(size_t)(k0 + k) * N + blockIdx.x * BN + n]);
  }
}

template <int BM, int BN, int BK, int TM, int TN>
__device__ __forceinline__ void store_shared_vec4(
    const TileLoad<BM, BN, BK, TM, TN>& regs, float (*As)[BM],
    float (*Bs)[BN]) {
  constexpr int NT = TileLoad<BM, BN, BK, TM, TN>::kThreads;
  const int tid = threadIdx.x;
#pragma unroll
  for (int r = 0; r < TileLoad<BM, BN, BK, TM, TN>::kA; ++r) {
    const int i = tid + r * NT;
    const int m = i / (BK / 4);
    const int k = (i % (BK / 4)) * 4;
    As[k + 0][m] = regs.a[r].x;
    As[k + 1][m] = regs.a[r].y;
    As[k + 2][m] = regs.a[r].z;
    As[k + 3][m] = regs.a[r].w;
  }
#pragma unroll
  for (int r = 0; r < TileLoad<BM, BN, BK, TM, TN>::kB; ++r) {
    const int i = tid + r * NT;
    const int k = i / (BN / 4);
    const int n = (i % (BN / 4)) * 4;
    *reinterpret_cast<float4*>(&Bs[k][n]) = regs.b[r];
  }
}

template <int BM, int BN, int BK, int TM, int TN>
__device__ __forceinline__ void load_tile_vec4(const float* __restrict__ A,
                                               const float* __restrict__ B,
                                               int N, int K, int k0,
                                               float (*As)[BM],
                                               float (*Bs)[BN]) {
  TileLoad<BM, BN, BK, TM, TN> regs;
  load_global_vec4<BM, BN, BK, TM, TN>(A, B, N, K, k0, regs);
  store_shared_vec4<BM, BN, BK, TM, TN>(regs, As, Bs);
}

template <int BN, int TN, bool STRIDED_N>
__device__ __forceinline__ int thread_col(int tc, int j) {
  if (STRIDED_N) {
    return (j / 4) * (BN / TN) * 4 + tc * 4 + j % 4;
  }
  return tc * TN + j;
}

template <int BM, int BN, int BK, int TM, int TN, bool STRIDED_N>
__device__ __forceinline__ void outer_product_vec4(const float (*As)[BM],
                                                   const float (*Bs)[BN],
                                                   int tr, int tc,
                                                   float (&acc)[TM][TN]) {
  float ra[TM], rb[TN];
#pragma unroll
  for (int k = 0; k < BK; ++k) {
#pragma unroll
    for (int i = 0; i < TM; i += 4) {
      const float4 a = *reinterpret_cast<const float4*>(&As[k][tr * TM + i]);
      ra[i + 0] = a.x;
      ra[i + 1] = a.y;
      ra[i + 2] = a.z;
      ra[i + 3] = a.w;
    }
#pragma unroll
    for (int j = 0; j < TN; j += 4) {
      const float4 b = *reinterpret_cast<const float4*>(
          &Bs[k][thread_col<BN, TN, STRIDED_N>(tc, j)]);
      rb[j + 0] = b.x;
      rb[j + 1] = b.y;
      rb[j + 2] = b.z;
      rb[j + 3] = b.w;
    }
#pragma unroll
    for (int i = 0; i < TM; ++i)
#pragma unroll
      for (int j = 0; j < TN; ++j) acc[i][j] += ra[i] * rb[j];
  }
}

template <int BM, int BN, int TM, int TN, bool STRIDED_N>
__device__ __forceinline__ void store_tile_vec4(float* __restrict__ C, int N,
                                                float alpha, float beta,
                                                int tr, int tc,
                                                const float (&acc)[TM][TN]) {
#pragma unroll
  for (int i = 0; i < TM; ++i) {
    const size_t row = (size_t)blockIdx.y * BM + tr * TM + i;
#pragma unroll
    for (int j = 0; j < TN; j += 4) {
      float4* dst = reinterpret_cast<float4*>(
          &C[row * N + (size_t)blockIdx.x * BN +
             thread_col<BN, TN, STRIDED_N>(tc, j)]);
      float4 c = *dst;
      c.x = alpha * acc[i][j + 0] + beta * c.x;
      c.y = alpha * acc[i][j + 1] + beta * c.y;
      c.z = alpha * acc[i][j + 2] + beta * c.z;
      c.w = alpha * acc[i][j + 3] + beta * c.w;
      *dst = c;
    }
  }
}

template <int BM, int BN, int BK, int TM, int TN>
__global__ void __launch_bounds__((BM / TM) * (BN / TN))
    sgemm_vec4(int M, int N, int K, float alpha, const float* __restrict__ A,
               const float* __restrict__ B, float beta,
               float* __restrict__ C) {
  __shared__ __align__(16) float As[BK][BM];
  __shared__ __align__(16) float Bs[BK][BN];
  const int tid = threadIdx.x;
  const int tr = tid / (BN / TN);
  const int tc = tid % (BN / TN);
  float acc[TM][TN] = {};
  for (int k0 = 0; k0 < K; k0 += BK) {
    load_tile_vec4<BM, BN, BK, TM, TN>(A, B, N, K, k0, As, Bs);
    __syncthreads();
    outer_product_vec4<BM, BN, BK, TM, TN, false>(As, Bs, tr, tc, acc);
    __syncthreads();
  }
  store_tile_vec4<BM, BN, TM, TN, false>(C, N, alpha, beta, tr, tc, acc);
}

template <int BM, int BN, int BK, int TM, int TN, bool STRIDED_N>
__global__ void __launch_bounds__((BM / TM) * (BN / TN))
    sgemm_vec4_dbuf(int M, int N, int K, float alpha,
                    const float* __restrict__ A, const float* __restrict__ B,
                    float beta, float* __restrict__ C) {
  __shared__ __align__(16) float As[2][BK][BM];
  __shared__ __align__(16) float Bs[2][BK][BN];
  const int tid = threadIdx.x;
  const int tr = tid / (BN / TN);
  const int tc = tid % (BN / TN);
  float acc[TM][TN] = {};
  TileLoad<BM, BN, BK, TM, TN> next;
  load_global_vec4<BM, BN, BK, TM, TN>(A, B, N, K, 0, next);
  store_shared_vec4<BM, BN, BK, TM, TN>(next, As[0], Bs[0]);
  __syncthreads();
  int stage = 0;
  for (int k0 = 0; k0 < K; k0 += BK) {
    const bool has_next = k0 + BK < K;
    if (has_next) {
      load_global_vec4<BM, BN, BK, TM, TN>(A, B, N, K, k0 + BK, next);
    }
    outer_product_vec4<BM, BN, BK, TM, TN, STRIDED_N>(As[stage], Bs[stage], tr,
                                                     tc, acc);
    if (has_next) {
      store_shared_vec4<BM, BN, BK, TM, TN>(next, As[stage ^ 1], Bs[stage ^ 1]);
    }
    __syncthreads();
    stage ^= 1;
  }
  store_tile_vec4<BM, BN, TM, TN, STRIDED_N>(C, N, alpha, beta, tr, tc, acc);
}

template <int BM, int BN, int BK, int TM, int TN>
bool launch_vec4(int M, int N, int K, float alpha, const float* A,
                 const float* B, float beta, float* C, cudaStream_t s) {
  static_assert(TM % 4 == 0 && TN % 4 == 0 && BK % 4 == 0, "float4 tiles");
  constexpr int kThreads = (BM / TM) * (BN / TN);
  if (M % BM || N % BN || K % BK) return false;
  dim3 grid(N / BN, M / BM);
  sgemm_vec4<BM, BN, BK, TM, TN>
      <<<grid, kThreads, 0, s>>>(M, N, K, alpha, A, B, beta, C);
  return true;
}

template <int BM, int BN, int BK, int TM, int TN, bool STRIDED_N = false>
bool launch_vec4_dbuf(int M, int N, int K, float alpha, const float* A,
                      const float* B, float beta, float* C, cudaStream_t s) {
  static_assert(TM % 4 == 0 && TN % 4 == 0 && BK % 4 == 0, "float4 tiles");
  constexpr int kThreads = (BM / TM) * (BN / TN);
  if (M % BM || N % BN || K % BK) return false;
  dim3 grid(N / BN, M / BM);
  sgemm_vec4_dbuf<BM, BN, BK, TM, TN, STRIDED_N>
      <<<grid, kThreads, 0, s>>>(M, N, K, alpha, A, B, beta, C);
  return true;
}
