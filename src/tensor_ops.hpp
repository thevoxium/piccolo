#ifndef TENSOR_OPS_H
#define TENSOR_OPS_H
#include "tensor.hpp"


Tensor* tensor_add(Tensor* a, Tensor* b);
Tensor* tensor_sub(Tensor* a, Tensor* b);
Tensor* tensor_scale(Tensor* a, float k);
Tensor* tensor_dot(Tensor* a, Tensor* b);
Tensor* tensor_neg(Tensor* a);
Tensor* tensor_log2(Tensor* a);
Tensor* tensor_exp2(Tensor* a);
Tensor* tensor_sqrt(Tensor* a);
Tensor* tensor_sin(Tensor* a);
Tensor* tensor_cos(Tensor* a);
Tensor* tensor_tan(Tensor* a);
Tensor* tensor_trunc(Tensor* a);
Tensor* tensor_ceil(Tensor* a);
Tensor* tensor_floor(Tensor* a);
Tensor* tensor_round(Tensor* a);
Tensor* tensor_square(Tensor* a);
Tensor* tensor_sign(Tensor* a);
Tensor* tensor_abs(Tensor* a);
Tensor* tensor_reciprocal(Tensor* a);
Tensor* tensor_pow(Tensor* a, float exponent);
Tensor* tensor_exp(Tensor* a);
Tensor* tensor_log(Tensor* a);


#endif