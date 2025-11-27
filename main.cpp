#include "src/engine.hpp"
#include "src/loss.hpp"
#include "src/tensor.hpp"
#include "src/tensor_ops.hpp"
#include <cmath>

int main() {
#ifdef USE_CUDA
  // Linear regression example using GPU
  int x_shape[2] = {1000, 1};
  int w_shape[2] = {1, 1};
  int b_shape[2] = {1000, 1};

  // Create tensors on CPU first
  Tensor *x = tensor_random(2, x_shape, DEVICE_CPU);
  Tensor *y = tensor_sin(x);
  realize(y);  // Realize the computation graph (x -> y)

  Tensor *w = tensor_random(2, w_shape, DEVICE_CPU);
  Tensor *b = tensor_random(2, b_shape, DEVICE_CPU);

  // Move all tensors to GPU
  set_graph_device(y, DEVICE_GPU);
  set_graph_device(w, DEVICE_GPU);
  set_graph_device(b, DEVICE_GPU);

  int epochs = 1000;
  float learning_rate = 0.001f;

  for (int i = 0; i < epochs; i++) {
    // Zero gradients for parameters before backward pass
    zero_grad(w);
    zero_grad(b);

    // Build computation graph and realize at the end
    Tensor *z = tensor_mm(x, w);
    Tensor *pred = tensor_add(z, b);
    Tensor *loss = loss_mse(pred, y);
    realize(loss);  // This realizes the entire computation graph
    backward(loss);

    // Sync loss to host for printing
    sync_to_host(loss);
    float loss_val = loss->data[0];

    // Sync parameter gradients to host for update
    sync_to_host(w);
    sync_to_host(b);

    // Update parameters on host
    for (int j = 0; j < w->capacity; j++) {
      w->data[j] -= learning_rate * w->grad[j];
    }
    for (int j = 0; j < b->capacity; j++) {
      b->data[j] -= learning_rate * b->grad[j];
    }

    // Sync updated parameters back to device
    sync_to_device(w);
    sync_to_device(b);

    if (i % 10 == 0 || i == epochs - 1) {
      std::cout << "Epoch " << i << " Loss: " << loss_val << std::endl;
    }

    tensor_free(z);
    tensor_free(pred);
    tensor_free(loss);
  }

  tensor_free(x);
  tensor_free(y);
  tensor_free(w);
  tensor_free(b);

  std::cout << "GPU Linear Regression completed!" << std::endl;
  return 0;

#else
  // CPU Linear regression example
  int x_shape[2] = {1000, 1};
  int w_shape[2] = {1, 1};
  int b_shape[2] = {1000, 1};

  Tensor *x = tensor_random(2, x_shape, DEVICE_CPU);
  Tensor *y = tensor_sin(x);
  realize(y);  // Realize the computation graph (x -> y)

  Tensor *w = tensor_random(2, w_shape, DEVICE_CPU);
  Tensor *b = tensor_random(2, b_shape, DEVICE_CPU);

  int epochs = 1000;
  float learning_rate = 0.001f;

  for (int i = 0; i < epochs; i++) {
    // Zero gradients for parameters before backward pass
    zero_grad(w);
    zero_grad(b);

    // Build computation graph and realize at the end
    Tensor *z = tensor_mm(x, w);
    Tensor *pred = tensor_add(z, b);
    Tensor *loss = loss_mse(pred, y);
    realize(loss);  // This realizes the entire computation graph
    backward(loss);

    // Update parameters in-place
    for (int j = 0; j < w->capacity; j++) {
      w->data[j] -= learning_rate * w->grad[j];
    }
    for (int j = 0; j < b->capacity; j++) {
      b->data[j] -= learning_rate * b->grad[j];
    }

    if (i % 10 == 0 || i == epochs - 1) {
      std::cout << "Epoch " << i << " Loss: " << loss->data[0] << std::endl;
    }

    tensor_free(z);
    tensor_free(pred);
    tensor_free(loss);
  }

  tensor_free(x);
  tensor_free(y);
  tensor_free(w);
  tensor_free(b);

  std::cout << "CPU Linear Regression completed!" << std::endl;
  return 0;
#endif
}
