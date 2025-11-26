#include "src/engine.hpp"
#include "src/loss.hpp"
#include "src/tensor.hpp"
#include "src/tensor_ops.hpp"
#include <iostream>

int main() {
  Tensor *x = tensor_random(2, new int[2]{1000, 1}, DEVICE_GPU);
  tensor_free(x);
  return 0;
}
