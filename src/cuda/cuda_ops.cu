#ifdef USE_CUDA

#include "cuda_ops.hpp"
#include "cuda_utils.hpp"
#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>

#define __threads_per_block 256
#define __mm_block_dim 16

// Helper macro to launch unary kernels
#define LAUNCH_UNARY_KERNEL(kernel, ...)                                       \
  do {                                                                         \
    int threadsPerBlock = __threads_per_block;                                 \
    int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;        \
    kernel<<<blocksPerGrid, threadsPerBlock>>>(__VA_ARGS__);                   \
    CUDA_CHECK(cudaGetLastError());                                            \
    CUDA_CHECK(cudaDeviceSynchronize());                                       \
  } while (0)

__global__ void tensor_add_kernel(const float *a, const float *b, float *result, int size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    result[idx] = a[idx] + b[idx];
  }
}

void cu_tensor_add(const float *a, const float *b, float *result, int size) {
  int threadsPerBlock = __threads_per_block;
  int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;

  tensor_add_kernel<<<blocksPerGrid, threadsPerBlock>>>(a, b, result, size);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());
}

__global__ void cu_tensor_add_backward_kernel(float *a_grad, float *b_grad, const float *result_grad, int size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    a_grad[idx] += result_grad[idx];
    b_grad[idx] += result_grad[idx];
  }
}

void cu_tensor_add_backward(float *a_grad, float *b_grad, const float *result_grad, int size) {
  int threadsPerBlock = __threads_per_block;
  int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;

  cu_tensor_add_backward_kernel<<<blocksPerGrid, threadsPerBlock>>>(a_grad, b_grad, result_grad, size);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());
}

__global__ void tensor_sub_kernel(const float *a, const float *b, float *result, int size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    result[idx] = a[idx] - b[idx];
  }
}

void cu_tensor_sub(const float *a, const float *b, float *result, int size) {
  int threadsPerBlock = __threads_per_block;
  int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;

  tensor_sub_kernel<<<blocksPerGrid, threadsPerBlock>>>(a, b, result, size);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());
}

__global__ void cu_tensor_sub_backward_kernel(float *a_grad, float *b_grad, const float *result_grad, int size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    a_grad[idx] += result_grad[idx];
    b_grad[idx] -= result_grad[idx];
  }
}

void cu_tensor_sub_backward(float *a_grad, float *b_grad, const float *result_grad, int size) {
  int threadsPerBlock = __threads_per_block;
  int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;

  cu_tensor_sub_backward_kernel<<<blocksPerGrid, threadsPerBlock>>>(a_grad, b_grad, result_grad, size);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());
}

__global__ void tensor_mm_kernel(const float *a, const float *b, float *result,
                                 int m, int k, int n) {
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  if (row < m && col < n) {
    float value = 0.0f;
    for (int i = 0; i < k; i++) {
      value += a[row * k + i] * b[i * n + col];
    }
    result[row * n + col] = value;
  }
}

__global__ void tensor_mm_grad_a_kernel(float *a_grad,
                                        const float *result_grad,
                                        const float *b, int m, int k, int n) {
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  if (row < m && col < k) {
    float sum = 0.0f;
    for (int i = 0; i < n; i++) {
      sum += result_grad[row * n + i] * b[col * n + i];
    }
    a_grad[row * k + col] += sum;
  }
}

__global__ void tensor_mm_grad_b_kernel(float *b_grad, const float *a,
                                        const float *result_grad, int m, int k,
                                        int n) {
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  if (row < k && col < n) {
    float sum = 0.0f;
    for (int i = 0; i < m; i++) {
      sum += a[i * k + row] * result_grad[i * n + col];
    }
    b_grad[row * n + col] += sum;
  }
}

void cu_tensor_mm(const float *a, const float *b, float *result, int m, int k,
                  int n) {
  dim3 blockDim(__mm_block_dim, __mm_block_dim);
  dim3 gridDim((n + blockDim.x - 1) / blockDim.x,
               (m + blockDim.y - 1) / blockDim.y);

  tensor_mm_kernel<<<gridDim, blockDim>>>(a, b, result, m, k, n);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());
}

void cu_tensor_mm_backward(float *a_grad, float *b_grad, const float *a,
                           const float *b, const float *result_grad, int m,
                           int k, int n) {
  dim3 blockDim(__mm_block_dim, __mm_block_dim);
  dim3 gridGradA((k + blockDim.x - 1) / blockDim.x,
                 (m + blockDim.y - 1) / blockDim.y);
  dim3 gridGradB((n + blockDim.x - 1) / blockDim.x,
                 (k + blockDim.y - 1) / blockDim.y);

  tensor_mm_grad_a_kernel<<<gridGradA, blockDim>>>(a_grad, result_grad, b, m, k,
                                                   n);
  CUDA_CHECK(cudaGetLastError());

  tensor_mm_grad_b_kernel<<<gridGradB, blockDim>>>(b_grad, a, result_grad, m, k,
                                                   n);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());
}

// ============================================================================
// Unary Operations
// ============================================================================

// Scale
__global__ void tensor_scale_kernel(const float *a, float *result, float k,
                                    int size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    result[idx] = k * a[idx];
  }
}

__global__ void tensor_scale_backward_kernel(float *a_grad,
                                             const float *result_grad, float k,
                                             int size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    a_grad[idx] += k * result_grad[idx];
  }
}

void cu_tensor_scale(const float *a, float *result, float k, int size) {
  LAUNCH_UNARY_KERNEL(tensor_scale_kernel, a, result, k, size);
}

void cu_tensor_scale_backward(float *a_grad, const float *result_grad, float k,
                              int size) {
  LAUNCH_UNARY_KERNEL(tensor_scale_backward_kernel, a_grad, result_grad, k,
                      size);
}

// Neg
__global__ void tensor_neg_kernel(const float *a, float *result, int size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    result[idx] = -a[idx];
  }
}

__global__ void tensor_neg_backward_kernel(float *a_grad,
                                           const float *result_grad, int size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    a_grad[idx] -= result_grad[idx];
  }
}

void cu_tensor_neg(const float *a, float *result, int size) {
  LAUNCH_UNARY_KERNEL(tensor_neg_kernel, a, result, size);
}

void cu_tensor_neg_backward(float *a_grad, const float *result_grad, int size) {
  LAUNCH_UNARY_KERNEL(tensor_neg_backward_kernel, a_grad, result_grad, size);
}

// Log2
__global__ void tensor_log2_kernel(const float *a, float *result, int size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    result[idx] = log2f(a[idx]);
  }
}

__global__ void tensor_log2_backward_kernel(float *a_grad, const float *a,
                                            const float *result_grad,
                                            int size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  float ln2 = 0.693147180559945f; // logf(2.0f)
  if (idx < size) {
    float val = a[idx];
    if (val > 0.0f) {
      a_grad[idx] += result_grad[idx] / (val * ln2);
    }
  }
}

void cu_tensor_log2(const float *a, float *result, int size) {
  LAUNCH_UNARY_KERNEL(tensor_log2_kernel, a, result, size);
}

void cu_tensor_log2_backward(float *a_grad, const float *a,
                             const float *result_grad, int size) {
  LAUNCH_UNARY_KERNEL(tensor_log2_backward_kernel, a_grad, a, result_grad,
                      size);
}

// Exp2
__global__ void tensor_exp2_kernel(const float *a, float *result, int size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    result[idx] = exp2f(a[idx]);
  }
}

__global__ void tensor_exp2_backward_kernel(float *a_grad,
                                            const float *result_data,
                                            const float *result_grad,
                                            int size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  float ln2 = 0.693147180559945f;
  if (idx < size) {
    a_grad[idx] += result_grad[idx] * result_data[idx] * ln2;
  }
}

void cu_tensor_exp2(const float *a, float *result, int size) {
  LAUNCH_UNARY_KERNEL(tensor_exp2_kernel, a, result, size);
}

void cu_tensor_exp2_backward(float *a_grad, const float *result_data,
                             const float *result_grad, int size) {
  LAUNCH_UNARY_KERNEL(tensor_exp2_backward_kernel, a_grad, result_data,
                      result_grad, size);
}

// Sqrt
__global__ void tensor_sqrt_kernel(const float *a, float *result, int size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    result[idx] = sqrtf(a[idx]);
  }
}

__global__ void tensor_sqrt_backward_kernel(float *a_grad, const float *a,
                                            const float *result_data,
                                            const float *result_grad,
                                            int size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    float val = a[idx];
    float res = result_data[idx];
    if (val > 0.0f && res != 0.0f) {
      a_grad[idx] += result_grad[idx] * 0.5f / res;
    }
  }
}

void cu_tensor_sqrt(const float *a, float *result, int size) {
  LAUNCH_UNARY_KERNEL(tensor_sqrt_kernel, a, result, size);
}

void cu_tensor_sqrt_backward(float *a_grad, const float *a,
                             const float *result_data,
                             const float *result_grad, int size) {
  LAUNCH_UNARY_KERNEL(tensor_sqrt_backward_kernel, a_grad, a, result_data,
                      result_grad, size);
}

// Sin
__global__ void tensor_sin_kernel(const float *a, float *result, int size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    result[idx] = sinf(a[idx]);
  }
}

__global__ void tensor_sin_backward_kernel(float *a_grad, const float *a,
                                           const float *result_grad, int size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    a_grad[idx] += result_grad[idx] * cosf(a[idx]);
  }
}

void cu_tensor_sin(const float *a, float *result, int size) {
  LAUNCH_UNARY_KERNEL(tensor_sin_kernel, a, result, size);
}

void cu_tensor_sin_backward(float *a_grad, const float *a,
                            const float *result_grad, int size) {
  LAUNCH_UNARY_KERNEL(tensor_sin_backward_kernel, a_grad, a, result_grad, size);
}

// Cos
__global__ void tensor_cos_kernel(const float *a, float *result, int size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    result[idx] = cosf(a[idx]);
  }
}

__global__ void tensor_cos_backward_kernel(float *a_grad, const float *a,
                                           const float *result_grad, int size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    a_grad[idx] -= result_grad[idx] * sinf(a[idx]);
  }
}

void cu_tensor_cos(const float *a, float *result, int size) {
  LAUNCH_UNARY_KERNEL(tensor_cos_kernel, a, result, size);
}

void cu_tensor_cos_backward(float *a_grad, const float *a,
                            const float *result_grad, int size) {
  LAUNCH_UNARY_KERNEL(tensor_cos_backward_kernel, a_grad, a, result_grad, size);
}

// Tan
__global__ void tensor_tan_kernel(const float *a, float *result, int size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    result[idx] = tanf(a[idx]);
  }
}

__global__ void tensor_tan_backward_kernel(float *a_grad,
                                           const float *result_data,
                                           const float *result_grad, int size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    float t = result_data[idx];
    a_grad[idx] += result_grad[idx] * (1.0f + t * t);
  }
}

void cu_tensor_tan(const float *a, float *result, int size) {
  LAUNCH_UNARY_KERNEL(tensor_tan_kernel, a, result, size);
}

void cu_tensor_tan_backward(float *a_grad, const float *result_data,
                            const float *result_grad, int size) {
  LAUNCH_UNARY_KERNEL(tensor_tan_backward_kernel, a_grad, result_data,
                      result_grad, size);
}

// Trunc
__global__ void tensor_trunc_kernel(const float *a, float *result, int size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    result[idx] = truncf(a[idx]);
  }
}

void cu_tensor_trunc(const float *a, float *result, int size) {
  LAUNCH_UNARY_KERNEL(tensor_trunc_kernel, a, result, size);
}

// Ceil
__global__ void tensor_ceil_kernel(const float *a, float *result, int size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    result[idx] = ceilf(a[idx]);
  }
}

void cu_tensor_ceil(const float *a, float *result, int size) {
  LAUNCH_UNARY_KERNEL(tensor_ceil_kernel, a, result, size);
}

// Floor
__global__ void tensor_floor_kernel(const float *a, float *result, int size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    result[idx] = floorf(a[idx]);
  }
}

void cu_tensor_floor(const float *a, float *result, int size) {
  LAUNCH_UNARY_KERNEL(tensor_floor_kernel, a, result, size);
}

// Round
__global__ void tensor_round_kernel(const float *a, float *result, int size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    result[idx] = roundf(a[idx]);
  }
}

void cu_tensor_round(const float *a, float *result, int size) {
  LAUNCH_UNARY_KERNEL(tensor_round_kernel, a, result, size);
}

// Square
__global__ void tensor_square_kernel(const float *a, float *result, int size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    float val = a[idx];
    result[idx] = val * val;
  }
}

__global__ void tensor_square_backward_kernel(float *a_grad, const float *a,
                                              const float *result_grad,
                                              int size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    a_grad[idx] += result_grad[idx] * 2.0f * a[idx];
  }
}

void cu_tensor_square(const float *a, float *result, int size) {
  LAUNCH_UNARY_KERNEL(tensor_square_kernel, a, result, size);
}

void cu_tensor_square_backward(float *a_grad, const float *a,
                               const float *result_grad, int size) {
  LAUNCH_UNARY_KERNEL(tensor_square_backward_kernel, a_grad, a, result_grad,
                      size);
}

// Sign
__global__ void tensor_sign_kernel(const float *a, float *result, int size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    float val = a[idx];
    result[idx] = (val > 0.0f) ? 1.0f : ((val < 0.0f) ? -1.0f : 0.0f);
  }
}

void cu_tensor_sign(const float *a, float *result, int size) {
  LAUNCH_UNARY_KERNEL(tensor_sign_kernel, a, result, size);
}

// Abs
__global__ void tensor_abs_kernel(const float *a, float *result, int size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    result[idx] = fabsf(a[idx]);
  }
}

__global__ void tensor_abs_backward_kernel(float *a_grad, const float *a,
                                           const float *result_grad, int size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    float val = a[idx];
    float sign = (val > 0.0f) ? 1.0f : ((val < 0.0f) ? -1.0f : 0.0f);
    a_grad[idx] += result_grad[idx] * sign;
  }
}

void cu_tensor_abs(const float *a, float *result, int size) {
  LAUNCH_UNARY_KERNEL(tensor_abs_kernel, a, result, size);
}

void cu_tensor_abs_backward(float *a_grad, const float *a,
                            const float *result_grad, int size) {
  LAUNCH_UNARY_KERNEL(tensor_abs_backward_kernel, a_grad, a, result_grad, size);
}

// Reciprocal
__global__ void tensor_reciprocal_kernel(const float *a, float *result,
                                         int size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    float val = a[idx];
    result[idx] = (val == 0.0f) ? INFINITY : (1.0f / val);
  }
}

__global__ void tensor_reciprocal_backward_kernel(float *a_grad, const float *a,
                                                  const float *result_grad,
                                                  int size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    float val = a[idx];
    if (val != 0.0f) {
      a_grad[idx] += result_grad[idx] * (-1.0f / (val * val));
    }
  }
}

void cu_tensor_reciprocal(const float *a, float *result, int size) {
  LAUNCH_UNARY_KERNEL(tensor_reciprocal_kernel, a, result, size);
}

void cu_tensor_reciprocal_backward(float *a_grad, const float *a,
                                   const float *result_grad, int size) {
  LAUNCH_UNARY_KERNEL(tensor_reciprocal_backward_kernel, a_grad, a, result_grad,
                      size);
}

// Pow
__global__ void tensor_pow_kernel(const float *a, float *result, float exponent,
                                  int size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    result[idx] = powf(a[idx], exponent);
  }
}

__global__ void tensor_pow_backward_kernel(float *a_grad, const float *a,
                                           const float *result_grad,
                                           float exponent, int size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    float base = a[idx];
    float local_grad = 0.0f;
    if (base == 0.0f) {
      if (exponent > 1.0f) {
        local_grad = 0.0f;
      } else if (exponent == 1.0f) {
        local_grad = 1.0f;
      }
    } else {
      local_grad = exponent * powf(base, exponent - 1.0f);
    }
    a_grad[idx] += result_grad[idx] * local_grad;
  }
}

void cu_tensor_pow(const float *a, float *result, float exponent, int size) {
  LAUNCH_UNARY_KERNEL(tensor_pow_kernel, a, result, exponent, size);
}

void cu_tensor_pow_backward(float *a_grad, const float *a,
                            const float *result_grad, float exponent,
                            int size) {
  LAUNCH_UNARY_KERNEL(tensor_pow_backward_kernel, a_grad, a, result_grad,
                      exponent, size);
}

// Exp
__global__ void tensor_exp_kernel(const float *a, float *result, int size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    result[idx] = expf(a[idx]);
  }
}

__global__ void tensor_exp_backward_kernel(float *a_grad,
                                           const float *result_data,
                                           const float *result_grad, int size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    a_grad[idx] += result_grad[idx] * result_data[idx];
  }
}

void cu_tensor_exp(const float *a, float *result, int size) {
  LAUNCH_UNARY_KERNEL(tensor_exp_kernel, a, result, size);
}

void cu_tensor_exp_backward(float *a_grad, const float *result_data,
                            const float *result_grad, int size) {
  LAUNCH_UNARY_KERNEL(tensor_exp_backward_kernel, a_grad, result_data,
                      result_grad, size);
}

// Log
__global__ void tensor_log_kernel(const float *a, float *result, int size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    float val = a[idx];
    result[idx] = (val <= 0.0f) ? -INFINITY : logf(val);
  }
}

__global__ void tensor_log_backward_kernel(float *a_grad, const float *a,
                                           const float *result_grad, int size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    float val = a[idx];
    if (val > 0.0f) {
      a_grad[idx] += result_grad[idx] / val;
    }
  }
}

void cu_tensor_log(const float *a, float *result, int size) {
  LAUNCH_UNARY_KERNEL(tensor_log_kernel, a, result, size);
}

void cu_tensor_log_backward(float *a_grad, const float *a,
                            const float *result_grad, int size) {
  LAUNCH_UNARY_KERNEL(tensor_log_backward_kernel, a_grad, a, result_grad, size);
}

// Aggregate (sum reduction)
__global__ void tensor_aggregate_kernel(const float *a, float *result,
                                        int size) {
  __shared__ float sdata[256];
  int tid = threadIdx.x;
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  sdata[tid] = (idx < size) ? a[idx] : 0.0f;
  __syncthreads();

  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) {
      sdata[tid] += sdata[tid + s];
    }
    __syncthreads();
  }

  if (tid == 0) {
    atomicAdd(result, sdata[0]);
  }
}

__global__ void tensor_aggregate_backward_kernel(float *a_grad, float grad_val,
                                                 int size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    a_grad[idx] += grad_val;
  }
}

void cu_tensor_aggregate(const float *a, float *result, int size) {
  // Zero the result first
  CUDA_CHECK(cudaMemset(result, 0, sizeof(float)));
  LAUNCH_UNARY_KERNEL(tensor_aggregate_kernel, a, result, size);
}

void cu_tensor_aggregate_backward(float *a_grad, float grad_val, int size) {
  LAUNCH_UNARY_KERNEL(tensor_aggregate_backward_kernel, a_grad, grad_val, size);
}

// Dot product
__global__ void tensor_dot_kernel(const float *a, const float *b, float *result,
                                  int size) {
  __shared__ float sdata[256];
  int tid = threadIdx.x;
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  sdata[tid] = (idx < size) ? (a[idx] * b[idx]) : 0.0f;
  __syncthreads();

  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) {
      sdata[tid] += sdata[tid + s];
    }
    __syncthreads();
  }

  if (tid == 0) {
    atomicAdd(result, sdata[0]);
  }
}

__global__ void tensor_dot_backward_kernel(float *a_grad, float *b_grad,
                                           const float *a, const float *b,
                                           float grad_val, int size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    a_grad[idx] += b[idx] * grad_val;
    b_grad[idx] += a[idx] * grad_val;
  }
}

void cu_tensor_dot(const float *a, const float *b, float *result, int size) {
  CUDA_CHECK(cudaMemset(result, 0, sizeof(float)));
  LAUNCH_UNARY_KERNEL(tensor_dot_kernel, a, b, result, size);
}

void cu_tensor_dot_backward(float *a_grad, float *b_grad, const float *a,
                            const float *b, float grad_val, int size) {
  LAUNCH_UNARY_KERNEL(tensor_dot_backward_kernel, a_grad, b_grad, a, b,
                      grad_val, size);
}

#endif // USE_CUDA

