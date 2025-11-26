#ifdef USE_CUDA

#include "cuda_ops.hpp"
#include "cuda_utils.hpp"
#include <cuda_runtime.h>
#include <stdio.h>

// CUDA kernel for element-wise addition
__global__ void tensor_add_kernel(const float *a, const float *b, float *result, int size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    result[idx] = a[idx] + b[idx];
  }
}

// Host function to launch the CUDA kernel
void cu_tensor_add(const float *a, const float *b, float *result, int size) {
  // Calculate grid and block dimensions
  int threadsPerBlock = 256;
  int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;

  // Launch the kernel
  tensor_add_kernel<<<blocksPerGrid, threadsPerBlock>>>(a, b, result, size);

  // Check for kernel launch errors
  CUDA_CHECK(cudaGetLastError());
  
  // Synchronize to ensure kernel completes
  CUDA_CHECK(cudaDeviceSynchronize());
}

// CUDA kernel for backward pass of addition (accumulates gradients to both parents)
__global__ void cu_tensor_add_backward_kernel(float *a_grad, float *b_grad, const float *result_grad, int size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    a_grad[idx] += result_grad[idx];
    b_grad[idx] += result_grad[idx];
  }
}

// Host function to launch the backward pass CUDA kernel for addition
void cu_tensor_add_backward(float *a_grad, float *b_grad, const float *result_grad, int size) {
  // Calculate grid and block dimensions
  int threadsPerBlock = 256;
  int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;

  // Launch the kernel
  cu_tensor_add_backward_kernel<<<blocksPerGrid, threadsPerBlock>>>(a_grad, b_grad, result_grad, size);

  // Check for kernel launch errors
  CUDA_CHECK(cudaGetLastError());
  
  // Synchronize to ensure kernel completes
  CUDA_CHECK(cudaDeviceSynchronize());
}

#endif // USE_CUDA

