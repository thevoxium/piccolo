#ifndef CUDA_OPS_H
#define CUDA_OPS_H

#ifdef USE_CUDA

// CUDA kernel for element-wise addition of two tensors
void cu_tensor_add(const float *a, const float *b, float *result, int size);

// CUDA kernel for backward pass of addition (gradient accumulation)
void cu_tensor_add_backward(float *a_grad, float *b_grad,
                            const float *result_grad, int size);

#endif // USE_CUDA

#endif // CUDA_OPS_H
