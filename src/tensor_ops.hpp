#ifndef TENSOR_OPS_H
#define TENSOR_OPS_H
#include "tensor.hpp"

#define TENSOR_OPS_COMPATIBLE_CHECK(a, b)                                      \
  do {                                                                         \
    CHECK_NULL_T(a);                                                           \
    CHECK_NULL_T(b);                                                           \
                                                                               \
    if ((a)->ndim != (b)->ndim) {                                              \
      ERROR_RETURN_NULL("Error: Tensor n-dim does not match\n");               \
    }                                                                          \
                                                                               \
    for (int _i = 0; _i < (a)->ndim; _i++) {                                   \
      if ((a)->shape[_i] != (b)->shape[_i]) {                                  \
        ERROR_RETURN_NULL("Error: Tensor shapes do not match\n");              \
      }                                                                        \
    }                                                                          \
                                                                               \
    if ((a)->data == NULL || (b)->data == NULL || (a)->grad == NULL ||         \
        (b)->grad == NULL) {                                                   \
      ERROR_RETURN_NULL("Error: Tensor data or grad is Null\n");               \
    }                                                                          \
                                                                               \
    if ((a)->device != (b)->device) {                                          \
      ERROR_RETURN_NULL("Error: Tensor devices do not match\n");               \
    }                                                                          \
  } while (0)

Tensor *tensor_add(Tensor *a, Tensor *b);
Tensor *tensor_sub(Tensor *a, Tensor *b);
Tensor *tensor_scale(Tensor *a, float k);
Tensor *tensor_dot(Tensor *a, Tensor *b);
Tensor *tensor_neg(Tensor *a);
Tensor *tensor_log2(Tensor *a);
Tensor *tensor_exp2(Tensor *a);
Tensor *tensor_sqrt(Tensor *a);
Tensor *tensor_sin(Tensor *a);
Tensor *tensor_cos(Tensor *a);
Tensor *tensor_tan(Tensor *a);
Tensor *tensor_trunc(Tensor *a);
Tensor *tensor_ceil(Tensor *a);
Tensor *tensor_floor(Tensor *a);
Tensor *tensor_round(Tensor *a);
Tensor *tensor_square(Tensor *a);
Tensor *tensor_sign(Tensor *a);
Tensor *tensor_abs(Tensor *a);
Tensor *tensor_reciprocal(Tensor *a);
Tensor *tensor_pow(Tensor *a, float exponent);
Tensor *tensor_exp(Tensor *a);
Tensor *tensor_log(Tensor *a);
Tensor *tensor_mm(Tensor *a, Tensor *b);
Tensor *tensor_aggregate(Tensor *a);

#endif
