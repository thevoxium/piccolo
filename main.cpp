#include "src/engine.hpp"
#include "src/loss.hpp"
#include "src/tensor.hpp"
#include "src/tensor_ops.hpp"
#include <iostream>

int main() {
  Tensor *x = tensor_random(2, new int[2]{1000, 1}, DEVICE_GPU);
  Tensor *y = tensor_sin(x);

  Tensor *w = tensor_random(2, new int[2]{1, 1});
  Tensor *b = tensor_random(2, new int[2]{1000, 1});

  int epochs = 100;
  for (int i = 0; i < epochs; i++) {
    Tensor *z = tensor_mm(x, w);
    Tensor *pred = tensor_add(z, b);
    Tensor *loss = loss_mse(pred, y);
    backward(loss);
    w = tensor_sub(w, tensor_scale(w, 0.01f));
    b = tensor_sub(b, tensor_scale(b, 0.01f));
    std::cout << "Epoch " << i << " Loss: " << loss->data[0] << std::endl;
    tensor_free(z);
    tensor_free(pred);
    tensor_free(loss);
  }

  tensor_free(x);
  tensor_free(y);
  tensor_free(w);
  tensor_free(b);

  return 0;
}
