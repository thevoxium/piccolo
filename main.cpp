#include "src/engine.hpp"
#include "src/loss.hpp"
#include "src/tensor.hpp"
#include "src/tensor_ops.hpp"
#include <iostream>

int main() {
  Tensor *x = tensor_random(2, new int[2]{1000, 1}, DEVICE_GPU);
  Tensor *y = tensor_random(2, new int[2]{1000, 1}, DEVICE_GPU);
  
  Tensor *z = tensor_add(x, y);

  std::cout << *z << std::endl;

  tensor_free(x);
  tensor_free(y);
  tensor_free(z);
  // Tensor *x = tensor_random(2, new int[2]{1000, 1}, DEVICE_GPU);
  // Tensor *y = tensor_sin(x);
  //
  // Tensor *w = tensor_random(2, new int[2]{1, 1}, DEVICE_CPU);
  // Tensor *b = tensor_random(2, new int[2]{1000, 1}, DEVICE_CPU);
  //
  // int epochs = 100;
  // for (int i = 0; i < epochs; i++) {
  //   Tensor *z = tensor_mm(x, w);
  //   Tensor *pred = tensor_add(z, b);
  //   Tensor *loss = loss_mse(pred, y);
  //   backward(loss);
  //
  //   // Update parameters in-place to keep them as leaf nodes
  //   // and prevent the computation graph from growing indefinitely
  //   float learning_rate = 0.01f;
  //   for (int j = 0; j < w->capacity; j++) {
  //       w->data[j] -= learning_rate * w->grad[j];
  //       w->grad[j] = 0.0f; // Zero gradients for next epoch
  //   }
  //   for (int j = 0; j < b->capacity; j++) {
  //       b->data[j] -= learning_rate * b->grad[j];
  //       b->grad[j] = 0.0f; // Zero gradients for next epoch
  //   }
  //
  //   std::cout << "Epoch " << i << " Loss: " << loss->data[0] << std::endl;
  //   tensor_free(z);
  //   tensor_free(pred);
  //   tensor_free(loss);
  // }
  //
  // tensor_free(x);
  // tensor_free(y);
  // tensor_free(w);
  // tensor_free(b);
  //
  // return 0;
}
