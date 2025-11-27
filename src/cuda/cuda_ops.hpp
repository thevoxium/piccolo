#ifndef CUDA_OPS_H
#define CUDA_OPS_H

#ifdef USE_CUDA

// Binary operations
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

void cu_tensor_dot(const float *a, const float *b, float *result, int size);
void cu_tensor_dot_backward(float *a_grad, float *b_grad, const float *a,
                            const float *b, float grad_val, int size);

// Unary operations
void cu_tensor_scale(const float *a, float *result, float k, int size);
void cu_tensor_scale_backward(float *a_grad, const float *result_grad, float k,
                              int size);

void cu_tensor_neg(const float *a, float *result, int size);
void cu_tensor_neg_backward(float *a_grad, const float *result_grad, int size);

void cu_tensor_log2(const float *a, float *result, int size);
void cu_tensor_log2_backward(float *a_grad, const float *a,
                             const float *result_grad, int size);

void cu_tensor_exp2(const float *a, float *result, int size);
void cu_tensor_exp2_backward(float *a_grad, const float *result_data,
                             const float *result_grad, int size);

void cu_tensor_sqrt(const float *a, float *result, int size);
void cu_tensor_sqrt_backward(float *a_grad, const float *a,
                             const float *result_data,
                             const float *result_grad, int size);

void cu_tensor_sin(const float *a, float *result, int size);
void cu_tensor_sin_backward(float *a_grad, const float *a,
                            const float *result_grad, int size);

void cu_tensor_cos(const float *a, float *result, int size);
void cu_tensor_cos_backward(float *a_grad, const float *a,
                            const float *result_grad, int size);

void cu_tensor_tan(const float *a, float *result, int size);
void cu_tensor_tan_backward(float *a_grad, const float *result_data,
                            const float *result_grad, int size);

void cu_tensor_trunc(const float *a, float *result, int size);
void cu_tensor_ceil(const float *a, float *result, int size);
void cu_tensor_floor(const float *a, float *result, int size);
void cu_tensor_round(const float *a, float *result, int size);

void cu_tensor_square(const float *a, float *result, int size);
void cu_tensor_square_backward(float *a_grad, const float *a,
                               const float *result_grad, int size);

void cu_tensor_sign(const float *a, float *result, int size);

void cu_tensor_abs(const float *a, float *result, int size);
void cu_tensor_abs_backward(float *a_grad, const float *a,
                            const float *result_grad, int size);

void cu_tensor_reciprocal(const float *a, float *result, int size);
void cu_tensor_reciprocal_backward(float *a_grad, const float *a,
                                   const float *result_grad, int size);

void cu_tensor_pow(const float *a, float *result, float exponent, int size);
void cu_tensor_pow_backward(float *a_grad, const float *a,
                            const float *result_grad, float exponent,
                            int size);

void cu_tensor_exp(const float *a, float *result, int size);
void cu_tensor_exp_backward(float *a_grad, const float *result_data,
                            const float *result_grad, int size);

void cu_tensor_log(const float *a, float *result, int size);
void cu_tensor_log_backward(float *a_grad, const float *a,
                            const float *result_grad, int size);

void cu_tensor_aggregate(const float *a, float *result, int size);
void cu_tensor_aggregate_backward(float *a_grad, float grad_val, int size);

#endif // USE_CUDA
#endif // CUDA_OPS_H
