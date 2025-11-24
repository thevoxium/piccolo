#include "loss.hpp"
#include "tensor_ops.hpp"

Tensor* loss_mse(Tensor* pred, Tensor* y){
    Tensor* diff = tensor_sub(pred, y);
    Tensor* diff_squared = tensor_square(diff);
    Tensor* sum = tensor_aggregate(diff_squared);
    Tensor* mean = tensor_scale(sum, 1.0f / diff_squared->capacity);
    return mean;
}