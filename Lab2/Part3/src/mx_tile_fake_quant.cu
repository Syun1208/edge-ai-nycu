#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <vector>

namespace {

constexpr int TILE_ROWS = 16;
constexpr int TILE_COLS = 32;
constexpr float TILE_SIZE = 512.0f;
constexpr unsigned FULL_MASK = 0xffffffffu;

struct ElementFormat {
    int mantissa_bits;
    int min_exponent;
    int max_exponent;
    float max_value;
};

__constant__ ElementFormat FORMATS[3] = {
    {1, 0, 2, 6.0f},
    {3, 0, 2, 7.5f},
    {3, -6, 8, 448.0f},
};

__device__ __forceinline__ int floor_log2(float value) {
    int exponent;
    frexpf(value, &exponent);
    return exponent - 1;
}

__device__ __forceinline__ float quantize_element(float value, const ElementFormat& element) {
    float magnitude = fabsf(value);
    int exponent = max(floor_log2(magnitude), element.min_exponent);
    float step = ldexpf(1.0f, exponent - element.mantissa_bits);
    float rounded = fminf(rintf(magnitude / step) * step, element.max_value);
    return copysignf(rounded, value);
}

template <typename scalar_t>
__global__ void mx_tile_fake_quant_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    int8_t* __restrict__ formats,
    int cols,
    float lower,
    float upper
) {
    const int lane = threadIdx.x;
    const int tile_row = threadIdx.y;
    const long row = static_cast<long>(blockIdx.y) * TILE_ROWS + tile_row;
    const long col = static_cast<long>(blockIdx.x) * TILE_COLS + lane;
    const long offset = row * cols + col;

    const float value = static_cast<float>(input[offset]);
    float block_max = fabsf(value);
    float block_square = value * value;
    for (int shift = 16; shift > 0; shift >>= 1) {
        block_max = fmaxf(block_max, __shfl_xor_sync(FULL_MASK, block_max, shift));
        block_square += __shfl_xor_sync(FULL_MASK, block_square, shift);
    }

    __shared__ float row_max[TILE_ROWS];
    __shared__ float row_square[TILE_ROWS];
    if (lane == 0) {
        row_max[tile_row] = block_max;
        row_square[tile_row] = block_square;
    }
    __syncthreads();

    float tile_max = 0.0f;
    float tile_square = 0.0f;
    for (int index = 0; index < TILE_ROWS; ++index) {
        tile_max = fmaxf(tile_max, row_max[index]);
        tile_square += row_square[index];
    }

    int format_index = 0;
    if (tile_max > 0.0f) {
        const float statistic = log2f(tile_max / sqrtf(tile_square / TILE_SIZE));
        format_index = statistic < lower ? 0 : (statistic < upper ? 1 : 2);
    }
    if (lane == 0 && tile_row == 0) {
        formats[static_cast<long>(blockIdx.y) * gridDim.x + blockIdx.x] = static_cast<int8_t>(format_index);
    }

    float result = 0.0f;
    if (block_max > 0.0f) {
        const ElementFormat element = FORMATS[format_index];
        const int exponent = min(max(floor_log2(block_max) - element.max_exponent, -127), 127);
        result = ldexpf(quantize_element(ldexpf(value, -exponent), element), exponent);
    }
    output[offset] = static_cast<scalar_t>(result);
}

}

std::vector<torch::Tensor> mx_tile_fake_quant(torch::Tensor input, double lower, double upper) {
    TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
    TORCH_CHECK(input.dim() == 2, "input must be 2-D");
    TORCH_CHECK(input.size(0) % TILE_ROWS == 0, "rows must be a multiple of 16");
    TORCH_CHECK(input.size(1) % TILE_COLS == 0, "cols must be a multiple of 32");

    auto contiguous = input.contiguous();
    auto output = torch::empty_like(contiguous);
    const int tile_rows = contiguous.size(0) / TILE_ROWS;
    const int tile_cols = contiguous.size(1) / TILE_COLS;
    auto formats = torch::empty({tile_rows, tile_cols}, contiguous.options().dtype(torch::kInt8));

    const dim3 grid(tile_cols, tile_rows);
    const dim3 block(TILE_COLS, TILE_ROWS);
    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, contiguous.scalar_type(), "mx_tile_fake_quant", [&] {
        mx_tile_fake_quant_kernel<scalar_t><<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
            contiguous.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            formats.data_ptr<int8_t>(),
            static_cast<int>(contiguous.size(1)),
            static_cast<float>(lower),
            static_cast<float>(upper)
        );
    });
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {output, formats};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
    module.def("mx_tile_fake_quant", &mx_tile_fake_quant, "Per-tile MX format selection and fake quantization");
}
