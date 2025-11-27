#ifdef USE_CUDA

#include "cuda_ops.hpp"
#include "cuda_utils.hpp"
#include <cuda_runtime.h>
#include <stdio.h>


#define __threads_per_block 256
#define __mm_block_dim 16

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
    a_grad[idx] -= result_grad[idx];
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

#endif // USE_CUDA

