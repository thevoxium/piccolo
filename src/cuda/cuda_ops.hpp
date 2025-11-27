#ifndef CUDA_OPS_H
#define CUDA_OPS_H

#ifdef USE_CUDA

void cu_tensor_add(const float *a, const float *b, float *result, int size);
void cu_tensor_add_backward(float *a_grad, float *b_grad,
                            const float *result_grad, int size);

void cu_tensor_sub(const float *a, const float *b, float *result, int size);
void cu_tensor_sub_backward(float *a_grad, float *b_grad,
                            const float *result_grad, int size);

void cu_tensor_mm(const float *a, const float *b, float *result, int m, int k,
                  int n);
void cu_tensor_mm_backward(float *a_grad, float *b_grad, const float *a,
                           const float *b, const float *result_grad, int m,
                           int k, int n);

#endif // USE_CUDA
#endif // CUDA_OPS_H
