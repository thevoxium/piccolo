#ifdef USE_CUDA

#include "cuda_ops.hpp"
#include "cuda_utils.hpp"
#include <cuda_runtime.h>
#include <stdio.h>


#define __threads_per_block 256

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

#endif // USE_CUDA

