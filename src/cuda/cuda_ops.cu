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
void tensor_add_cuda(const float *a, const float *b, float *result, int size) {
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

#endif // USE_CUDA

