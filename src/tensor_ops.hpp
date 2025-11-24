#ifndef TENSOR_OPS_H
#define TENSOR_OPS_H
#include "tensor.hpp"


Tensor* tensor_add(Tensor* a, Tensor* b);
Tensor* tensor_sub(Tensor* a, Tensor* b);
Tensor* tensor_scale(Tensor* a, float k);
Tensor* tensor_dot(Tensor* a, Tensor* b);


#endif