#ifndef CUDA_OPS_H
#define CUDA_OPS_H

#ifdef USE_CUDA

// CUDA kernel for element-wise addition of two tensors
void tensor_add_cuda(const float *a, const float *b, float *result, int size);

#endif // USE_CUDA

#endif // CUDA_OPS_H

