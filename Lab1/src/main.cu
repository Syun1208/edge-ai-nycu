// =============================================================================
//  SGEMM benchmark harness  (you should not need to edit this file)
// =============================================================================
//  ./sgemm --list
//  ./sgemm --chain   print the Part 4 optimization chain (kPart4Chain), one per line
//  ./sgemm <kernel> [M N K] [options]
//     <kernel>  : index, name (see --list), "cublas" (= 0) or "all"
//     M N K     : default 4096 4096 4096
//  options:
//     --reps R       timed repetitions (default 20), median is reported
//     --warmup W     untimed warmup launches (default 3)
//     --alpha a      default 1
//     --beta b       default 0
//     --no-check     skip the correctness check against cuBLAS
//     --no-flush     do NOT flush L2 before each timed launch
//     --csv          one machine-readable line per kernel (used by sweep.sh)
//     --profile      1 launch of the kernel only (no cuBLAS, no check);
//                    use this under nsys / ncu
// =============================================================================
#include <cublas_v2.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#include "common.cuh"
#ifdef USE_SOLUTION
#include "../instructor/solution_kernels.cuh"
#else
#include "kernels.cuh"
#endif

#define CUBLAS_CHECK(x)                                                      \
  do {                                                                       \
    cublasStatus_t st_ = (x);                                                \
    if (st_ != CUBLAS_STATUS_SUCCESS) {                                      \
      fprintf(stderr, "cuBLAS error %d at %s:%d\n", (int)st_, __FILE__,      \
              __LINE__);                                                     \
      exit(1);                                                               \
    }                                                                        \
  } while (0)

static const int kNumKernels = sizeof(kKernels) / sizeof(kKernels[0]);

// ----------------------------------------------------------------------------
// cuBLAS reference (id 0). cuBLAS is column-major; for row-major C = A @ B we
// compute C^T = B^T @ A^T, which in column-major is exactly our row-major data.
// Math mode is pinned to plain FP32 (no TF32) so the comparison is fair.
// ----------------------------------------------------------------------------
static cublasHandle_t g_cublas = nullptr;

static bool launch_cublas(int M, int N, int K, float alpha, const float* A,
                          const float* B, float beta, float* C,
                          cudaStream_t s) {
  CUBLAS_CHECK(cublasSetStream(g_cublas, s));
  CUBLAS_CHECK(cublasSgemm(g_cublas, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha,
                           B, N, A, K, &beta, C, N));
  return true;
}

static Launcher get_launcher(int id) {
  return id == 0 ? launch_cublas : kKernels[id - 1].launch;
}
static const char* get_name(int id) {
  return id == 0 ? "cublas" : kKernels[id - 1].name;
}

// ----------------------------------------------------------------------------
// Device info + (approximate) roofline numbers
// ----------------------------------------------------------------------------
static int fp32_cores_per_sm(int major, int minor) {
  switch (major * 10 + minor) {
    case 60: return 64;
    case 61: case 62: return 128;
    case 70: case 72: case 75: return 64;
    case 80: return 64;
    case 86: case 87: case 89: return 128;
    case 90: return 128;
    default: return major >= 10 ? 128 : -1;
  }
}

struct DeviceInfo {
  char name[256];
  int major, minor, sms, l2_bytes;
  double peak_gflops;  // FP32 FMA peak at max clock (approximate)
  double peak_gbps;    // DRAM bandwidth (approximate)
};

static DeviceInfo query_device() {
  DeviceInfo d{};
  int dev;
  CUDA_CHECK(cudaGetDevice(&dev));
  cudaDeviceProp p;
  CUDA_CHECK(cudaGetDeviceProperties(&p, dev));
  strncpy(d.name, p.name, sizeof(d.name) - 1);
  d.major = p.major;
  d.minor = p.minor;
  d.sms = p.multiProcessorCount;
  d.l2_bytes = p.l2CacheSize;
  int clk_khz = 0, memclk_khz = 0, bus_bits = 0;
  cudaDeviceGetAttribute(&clk_khz, cudaDevAttrClockRate, dev);
  cudaDeviceGetAttribute(&memclk_khz, cudaDevAttrMemoryClockRate, dev);
  cudaDeviceGetAttribute(&bus_bits, cudaDevAttrGlobalMemoryBusWidth, dev);
  const int cores = fp32_cores_per_sm(d.major, d.minor);
  d.peak_gflops = cores > 0 ? 2.0 * cores * d.sms * clk_khz * 1e3 / 1e9 : NAN;
  d.peak_gbps = 2.0 * memclk_khz * 1e3 * (bus_bits / 8.0) / 1e9;
  return d;
}

// ----------------------------------------------------------------------------
struct Options {
  std::string kernel = "all";
  int M = 4096, N = 4096, K = 4096;
  int reps = 20, warmup = 3;
  float alpha = 1.f, beta = 0.f;
  bool check = true, flush = true, csv = false, profile = false;
};

static void usage() {
  printf("usage: ./sgemm --list | --chain | <kernel|all|cublas> [M N K] [--reps R] "
         "[--warmup W] [--alpha a] [--beta b] [--no-check] [--no-flush] "
         "[--csv] [--profile]\n");
}

static void list_kernels() {
  printf("  0  cublas\n");
  for (int i = 0; i < kNumKernels; ++i) printf("%3d  %s\n", i + 1, kKernels[i].name);
}

// Part 4 optimization chain, read by scripts/grade.sh.
static void print_chain() {
#ifdef HAS_PART4_CHAIN
  for (const char* k : kPart4Chain) printf("%s\n", k);
#endif
}

static int resolve_kernel(const std::string& s) {
  if (s == "cublas") return 0;
  bool numeric = !s.empty() && std::all_of(s.begin(), s.end(), ::isdigit);
  if (numeric) {
    int id = atoi(s.c_str());
    if (id >= 0 && id <= kNumKernels) return id;
  }
  for (int i = 0; i < kNumKernels; ++i)
    if (s == kKernels[i].name) return i + 1;
  fprintf(stderr, "unknown kernel '%s'\n", s.c_str());
  list_kernels();
  exit(1);
}

// ----------------------------------------------------------------------------
// Timing: median over reps of per-launch cudaEvent time. Before each launch we
// (optionally) overwrite a buffer 2x the L2 size so every launch starts cold,
// which matters for small / GEMV shapes whose operands fit in L2. The memset
// leaves L2 full of dirty lines whose write-back would otherwise be charged to
// the timed kernel (large on GPUs with a big L2), so we then read a second,
// clean 2x-L2 buffer to evict them before the timer starts.
// ----------------------------------------------------------------------------
__global__ void l2_evict(const float4* __restrict__ p, size_t n, float* sink) {
  float acc = 0.f;
  for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n;
       i += (size_t)gridDim.x * blockDim.x)
    acc += p[i].x;
  if (acc == 1234.5f) *sink = acc;  // never true; keeps the loads alive
}

static float time_kernel(Launcher f, const Options& o, const float* A,
                         const float* B, float* C, void* flush_buf,
                         void* evict_buf, size_t flush_bytes, cudaStream_t s) {
  for (int i = 0; i < o.warmup; ++i)
    f(o.M, o.N, o.K, o.alpha, A, B, o.beta, C, s);
  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));
  std::vector<float> ms(o.reps);
  for (int r = 0; r < o.reps; ++r) {
    if (flush_buf) {
      CUDA_CHECK(cudaMemsetAsync(flush_buf, r & 0xff, flush_bytes, s));
      l2_evict<<<1024, 256, 0, s>>>((const float4*)evict_buf, flush_bytes / 16,
                                    (float*)flush_buf);
    }
    CUDA_CHECK(cudaEventRecord(start, s));
    f(o.M, o.N, o.K, o.alpha, A, B, o.beta, C, s);
    CUDA_CHECK(cudaEventRecord(stop, s));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventElapsedTime(&ms[r], start, stop));
  }
  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));
  std::sort(ms.begin(), ms.end());
  return ms[o.reps / 2];
}

// max_i |c - ref| / (1 + |ref|). Indexing bugs give O(1); FP32 reordering of
// the K-sum grows with K (~5e-5 at K = 2048, ~1e-3 at K = 65536), so the
// tolerance is check_tol(K) = 1e-3 * max(1, sqrt(K / 1024)).
static double check_tol(int K) { return 1e-3 * std::max(1.0, std::sqrt(K / 1024.0)); }

// ----------------------------------------------------------------------------
// Guarded buffers. Every matrix sits between two guard regions filled with
// 0xFF bytes (= NaN). A kernel that writes out of bounds changes a guard and is
// flagged FAIL itself, instead of silently corrupting a neighbouring buffer and
// making a later kernel fail. Out-of-bounds reads pull in NaN and fail the
// result check. Guards hold >= 128 rows of overrun for every matrix.
// ----------------------------------------------------------------------------
struct Guarded {
  char* base = nullptr;
  size_t guard = 0, bytes = 0;
  float* ptr() const { return (float*)(base + guard); }
};

static Guarded alloc_guarded(size_t n, size_t guard) {
  Guarded g;
  g.guard = guard;
  g.bytes = n * sizeof(float);
  CUDA_CHECK(cudaMalloc(&g.base, g.bytes + 2 * guard));
  CUDA_CHECK(cudaMemset(g.base, 0xff, g.bytes + 2 * guard));
  return g;
}

__global__ void count_bad_guard(const unsigned* __restrict__ p, size_t n,
                                unsigned long long* bad) {
  unsigned long long c = 0;
  for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n;
       i += (size_t)gridDim.x * blockDim.x)
    c += p[i] != 0xffffffffu;
  if (c) atomicAdd(bad, c);
}

// Returns the number of guard words that were overwritten, then restores them.
static unsigned long long check_guards(const std::vector<Guarded*>& bufs,
                                       unsigned long long* d_bad) {
  unsigned long long total = 0;
  for (Guarded* g : bufs) {
    CUDA_CHECK(cudaMemset(d_bad, 0, sizeof(*d_bad)));
    const char* regions[2] = {g->base, g->base + g->guard + g->bytes};
    for (const char* r : regions)
      count_bad_guard<<<256, 256>>>((const unsigned*)r, g->guard / 4, d_bad);
    unsigned long long bad = 0;
    CUDA_CHECK(cudaMemcpy(&bad, d_bad, sizeof(bad), cudaMemcpyDeviceToHost));
    if (bad) {
      CUDA_CHECK(cudaMemset(g->base, 0xff, g->guard));
      CUDA_CHECK(cudaMemset(g->base + g->guard + g->bytes, 0xff, g->guard));
    }
    total += bad;
  }
  return total;
}
static double check_result(const std::vector<float>& c,
                           const std::vector<float>& ref, size_t* worst) {
  double max_err = 0.0;
  *worst = 0;
  for (size_t i = 0; i < c.size(); ++i) {
    double e = std::fabs((double)c[i] - ref[i]) / (1.0 + std::fabs((double)ref[i]));
    if (std::isnan(c[i]) || std::isnan(ref[i])) e = INFINITY;
    if (e > max_err) { max_err = e; *worst = i; }
  }
  return max_err;
}

int main(int argc, char** argv) {
  // Line-buffer stdout even when piped, so every finished result row is out
  // before the next kernel runs. check_all.sh blames the kernel after the last
  // printed row; a host crash (segfault) would otherwise lose buffered rows.
  setvbuf(stdout, nullptr, _IOLBF, 0);
  Options o;
  std::vector<std::string> pos;
  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    auto next = [&]() { if (i + 1 >= argc) { usage(); exit(1); } return std::string(argv[++i]); };
    if (a == "--list") { list_kernels(); return 0; }
    else if (a == "--chain") { print_chain(); return 0; }
    else if (a == "--reps") o.reps = std::max(1, std::stoi(next()));
    else if (a == "--warmup") o.warmup = std::stoi(next());
    else if (a == "--alpha") o.alpha = std::stof(next());
    else if (a == "--beta") o.beta = std::stof(next());
    else if (a == "--no-check") o.check = false;
    else if (a == "--no-flush") o.flush = false;
    else if (a == "--csv") o.csv = true;
    else if (a == "--profile") o.profile = true;
    else if (a == "-h" || a == "--help") { usage(); return 0; }
    else pos.push_back(a);
  }
  if (pos.size() != 1 && pos.size() != 4) { usage(); return 1; }
  o.kernel = pos[0];
  if (pos.size() == 4) {
    o.M = std::stoi(pos[1]); o.N = std::stoi(pos[2]); o.K = std::stoi(pos[3]);
  }
  if (o.profile) { o.reps = 1; o.warmup = 0; o.check = false; o.flush = false; }

  const DeviceInfo dev = query_device();
  if (!o.csv) {
    printf("GPU: %s  (sm_%d%d, %d SMs, L2 %.1f MB)\n", dev.name, dev.major,
           dev.minor, dev.sms, dev.l2_bytes / 1048576.0);
    printf("Approx. peaks: FP32 %.0f GFLOP/s, DRAM %.0f GB/s  "
           "(verify against your spec sheet!)\n", dev.peak_gflops, dev.peak_gbps);
    printf("Problem: M=%d N=%d K=%d alpha=%g beta=%g\n\n", o.M, o.N, o.K,
           o.alpha, o.beta);
  }

  // ---- data ----------------------------------------------------------------
  const size_t nA = (size_t)o.M * o.K, nB = (size_t)o.K * o.N, nC = (size_t)o.M * o.N;
  std::vector<float> hA(nA), hB(nB), hC0(nC);
  std::mt19937 rng(1234);
  std::uniform_real_distribution<float> U(-1.f, 1.f);
  for (auto& x : hA) x = U(rng);
  for (auto& x : hB) x = U(rng);
  for (auto& x : hC0) x = U(rng);

  // Guard: 128 rows of the widest matrix + 1 MB, rounded up to 256 bytes.
  // None in --profile mode, so compute-sanitizer sees the real buffer bounds.
  const size_t guard = o.profile ? 0 :
      ((128 * (size_t)std::max({o.M, o.N, o.K}) * sizeof(float) + (1 << 20)) + 255) & ~(size_t)255;
  Guarded gA = alloc_guarded(nA, guard), gB = alloc_guarded(nB, guard),
          gC = alloc_guarded(nC, guard), gC0 = alloc_guarded(nC, guard),
          gCref = alloc_guarded(nC, guard);
  float *A = gA.ptr(), *B = gB.ptr(), *C = gC.ptr(), *C0 = gC0.ptr(), *Cref = gCref.ptr();
  const std::vector<Guarded*> guarded = {&gA, &gB, &gC, &gC0, &gCref};
  unsigned long long* d_bad;
  CUDA_CHECK(cudaMalloc(&d_bad, sizeof(*d_bad)));
  CUDA_CHECK(cudaMemcpy(A, hA.data(), nA * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(B, hB.data(), nB * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(C0, hC0.data(), nC * sizeof(float), cudaMemcpyHostToDevice));

  void *flush_buf = nullptr, *evict_buf = nullptr;
  size_t flush_bytes = 0;
  if (o.flush && dev.l2_bytes > 0) {
    flush_bytes = 2 * (size_t)dev.l2_bytes;
    CUDA_CHECK(cudaMalloc(&flush_buf, flush_bytes));
    CUDA_CHECK(cudaMalloc(&evict_buf, flush_bytes));
    CUDA_CHECK(cudaMemset(evict_buf, 0, flush_bytes));
  }

  cudaStream_t s;
  CUDA_CHECK(cudaStreamCreate(&s));
  CUBLAS_CHECK(cublasCreate(&g_cublas));
  CUBLAS_CHECK(cublasSetMathMode(g_cublas, CUBLAS_DEFAULT_MATH));  // no TF32

  // ---- which kernels -------------------------------------------------------
  std::vector<int> ids;
  if (o.kernel == "all") for (int i = 0; i <= kNumKernels; ++i) ids.push_back(i);
  else ids.push_back(resolve_kernel(o.kernel));

  // ---- profile mode: exactly one launch, nothing else ------------------------
  if (o.profile) {
    for (int id : ids) {
      CUDA_CHECK(cudaMemcpy(C, C0, nC * sizeof(float), cudaMemcpyDeviceToDevice));
      bool ok = get_launcher(id)(o.M, o.N, o.K, o.alpha, A, B, o.beta, C, s);
      CUDA_CHECK(cudaStreamSynchronize(s));
      CUDA_CHECK(cudaGetLastError());
      printf("%-24s %s\n", get_name(id), ok ? "launched" : "SKIPPED");
    }
    return 0;
  }

  // ---- reference -------------------------------------------------------------
  std::vector<float> hRef(nC), hOut(nC);
  if (o.check) {
    CUDA_CHECK(cudaMemcpy(Cref, C0, nC * sizeof(float), cudaMemcpyDeviceToDevice));
    launch_cublas(o.M, o.N, o.K, o.alpha, A, B, o.beta, Cref, s);
    CUDA_CHECK(cudaStreamSynchronize(s));
    CUDA_CHECK(cudaMemcpy(hRef.data(), Cref, nC * sizeof(float), cudaMemcpyDeviceToHost));
  }
  const float cublas_ms =
      time_kernel(launch_cublas, o, A, B, C, flush_buf, evict_buf, flush_bytes, s);

  const double flops = 2.0 * o.M * o.N * (double)o.K;
  // Minimum possible DRAM traffic: read A, B (and C if beta != 0), write C.
  const double bytes = 4.0 * (nA + nB + nC * (o.beta != 0.f ? 2.0 : 1.0));

  if (!o.csv)
    printf("%-3s %-24s %10s %10s %9s %8s %8s %8s  %s\n", "id", "kernel", "ms",
           "GFLOP/s", "GB/s", "%cuBLAS", "%peakF", "%peakBW", "check");
  for (int id : ids) {
    Launcher f = get_launcher(id);
    // correctness run (also tells us whether the kernel is supported)
    CUDA_CHECK(cudaMemcpy(C, C0, nC * sizeof(float), cudaMemcpyDeviceToDevice));
    bool ok = f(o.M, o.N, o.K, o.alpha, A, B, o.beta, C, s);
    if (!ok) {
      if (o.csv)
        printf("%s,%s,%d,%d,%d,,,,,,,SKIPPED\n", dev.name, get_name(id), o.M, o.N, o.K);
      else
        printf("%-3d %-24s   SKIPPED (not implemented / unsupported size)\n", id, get_name(id));
      continue;
    }
    CUDA_CHECK(cudaStreamSynchronize(s));
    CUDA_CHECK(cudaGetLastError());
    std::string status = "not checked";
    if (o.check) {
      CUDA_CHECK(cudaMemcpy(hOut.data(), C, nC * sizeof(float), cudaMemcpyDeviceToHost));
      size_t worst;
      double err = check_result(hOut, hRef, &worst);
      char buf[160];
      if (err < check_tol(o.K))
        snprintf(buf, sizeof(buf), "PASS (err %.1e)", err);
      else
        snprintf(buf, sizeof(buf), "FAIL (err %.1e at row %zu col %zu: got %g want %g)",
                 err, worst / o.N, worst % o.N, hOut[worst], hRef[worst]);
      status = buf;
    }
    // Out-of-bounds writes, during the correctness run or the timed launches.
    unsigned long long oob = check_guards(guarded, d_bad);
    const float ms = id == 0 ? cublas_ms
                             : time_kernel(f, o, A, B, C, flush_buf, evict_buf, flush_bytes, s);
    oob += check_guards(guarded, d_bad);
    if (oob) {
      char buf[160];
      snprintf(buf, sizeof(buf), "FAIL (out-of-bounds write: %llu guard words changed)", oob);
      status = buf;
    }
    const double gflops = flops / (ms * 1e6);
    const double gbps = bytes / (ms * 1e6);
    if (o.csv)
      printf("%s,%s,%d,%d,%d,%.5f,%.1f,%.1f,%.1f,%.1f,%.1f,%s\n", dev.name,
             get_name(id), o.M, o.N, o.K, ms, gflops, gbps,
             100.0 * cublas_ms / ms, 100.0 * gflops / dev.peak_gflops,
             100.0 * gbps / dev.peak_gbps,
             status.rfind("FAIL", 0) == 0 ? "FAIL" : !o.check ? "UNCHECKED" : "PASS");
    else
      printf("%-3d %-24s %10.4f %10.1f %9.1f %7.1f%% %7.1f%% %7.1f%%  %s\n", id,
             get_name(id), ms, gflops, gbps, 100.0 * cublas_ms / ms,
             100.0 * gflops / dev.peak_gflops, 100.0 * gbps / dev.peak_gbps,
             status.c_str());
  }

  cublasDestroy(g_cublas);
  for (Guarded* g : guarded) cudaFree(g->base);
  cudaFree(d_bad);
  if (flush_buf) cudaFree(flush_buf);
  if (evict_buf) cudaFree(evict_buf);
  cudaStreamDestroy(s);
  return 0;
}
