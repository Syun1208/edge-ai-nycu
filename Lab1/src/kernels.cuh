#pragma once
// =============================================================================
//  kernels.cuh  --  kernel registry
// =============================================================================
//  Row-major, FP32 only:  C (MxN) = alpha * A (MxK) @ B (KxN) + beta * C
//
//  Each kernel lives in its own file under src/kernels/:
//    naive.cuh      Kernel 1  (given, Part 1)
//    coalesced.cuh  Kernel 2  (Part 2)
//    smem.cuh       Kernel 3  (Part 3a)
//    reg2d.cuh      Kernel 4  (Part 3b)
//    custom.cuh     your own kernels (Part 4)
//
//  Rules:
//   * FP32 CUDA cores only. No tensor cores, no TF32, no WMMA / mma.sync.
//   * No cuBLAS / CUTLASS / other libraries inside your kernels.
//   * Every kernel must pass the harness correctness check for every size
//     its launcher accepts. Return false from the launcher for sizes you
//     do not support.
//
//  To add a kernel: write the __global__ function and a launcher in the
//  matching file under src/kernels/, then add one line to kKernels[] below.
// =============================================================================
#include <vector>

#include "common.cuh"

#include "kernels/naive.cuh"
#include "kernels/coalesced.cuh"
#include "kernels/smem.cuh"
#include "kernels/reg2d.cuh"
#include "kernels/custom.cuh"

// =============================================================================
// Registry. The harness refers to kernels by index or by name.
// Index 0 is always cuBLAS (provided by the harness).
// Add new configurations / kernels here.
// =============================================================================
static const KernelEntry kKernels[] = {
    {"naive", launch_naive},                                   // 1
    {"coalesced", launch_coalesced},                           // 2
    {"smem16", launch_smem<16>},                               // 3
    {"smem32", launch_smem<32>},                               // 4
    {"reg2d_128x128x8_8x8", launch_reg2d<128, 128, 8, 8, 8>},  // 5
    {"reg2d_64x64x8_4x4", launch_reg2d<64, 64, 8, 4, 4>},      // 6
    {"reg2d_128x64x8_8x4", launch_reg2d<128, 64, 8, 8, 4>},    // 7
    {"reg2d_128x128x16_8x8", launch_reg2d<128, 128, 16, 8, 8>},  // 8
    {"reg2d_64x64x32_4x4", launch_reg2d<64, 64, 32, 4, 4>},      // 9
    {"reg2d_128x128x8_4x4", launch_reg2d<128, 128, 8, 4, 4>},    // 10
    {"custom_vec4", launch_vec4<128, 128, 8, 8, 8>},              // 11
    {"custom_vec4_dbuf", launch_vec4_dbuf<128, 128, 8, 8, 8>},    // 12
    {"custom_vec4_dbuf_nbc", launch_vec4_dbuf<128, 128, 8, 8, 8, true>},  // 13
};

// =============================================================================
// Part 4: optimization chain, graded by scripts/grade.sh.
// List kernel names from kKernels[] in order. Entry 0 is your starting point
// (one of your Part 3 kernels); every later entry must be the previous entry
// plus ONE optimization. Each step is timed against the step before it at
// 4096x4096x4096: >= 1.05x faster = successful (5 points), otherwise failed
// (2.5 points, only with an explanation in your report). Capped at 10 points.
// Every entry must be correct. Example:
//   {"reg2d_128x128x8_8x8", "custom_vec4", "custom_vec4_dbuf"}
// =============================================================================
#define HAS_PART4_CHAIN 1
static const std::vector<const char*> kPart4Chain = {
    "reg2d_128x128x8_8x8",
    "custom_vec4",
    "custom_vec4_dbuf",
    "custom_vec4_dbuf_nbc",
};
