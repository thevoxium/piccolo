#ifndef LOSS_H
#define LOSS_H
#include "tensor.hpp"

Tensor* loss_mse(Tensor* pred, Tensor* y);

#endif