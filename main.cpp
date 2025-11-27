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

  Tensor *x = tensor_random(2, x_shape, DEVICE_GPU);
  realize(x);
  Tensor *y = tensor_sin(x);
  realize(y);

  Tensor *w = tensor_random(2, w_shape, DEVICE_GPU);
  realize(w);
  Tensor *b = tensor_random(2, b_shape, DEVICE_GPU);
  realize(b);

  int epochs = 100;
  float learning_rate = 0.01f;

  for (int i = 0; i < epochs; i++) {
    Tensor *z = tensor_mm(x, w);
    Tensor *pred = tensor_add(z, b);
    Tensor *loss = loss_mse(pred, y);
    realize(loss);
    backward(loss);

    // Copy loss to host for printing
    float loss_val;
    cudaMemcpy(&loss_val, loss->d_data, sizeof(float), cudaMemcpyDeviceToHost);

    // Update parameters on GPU
    // Copy gradients to host, update, copy back
    float w_grad;
    cudaMemcpy(&w_grad, w->d_grad, sizeof(float), cudaMemcpyDeviceToHost);

    float w_val;
    cudaMemcpy(&w_val, w->d_data, sizeof(float), cudaMemcpyDeviceToHost);
    w_val -= learning_rate * w_grad;
    cudaMemcpy(w->d_data, &w_val, sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(w->d_grad, 0, sizeof(float));

    // Update bias (element-wise on GPU would be better, but for simplicity...)
    float *b_grad_host = new float[b->capacity];
    float *b_data_host = new float[b->capacity];
    cudaMemcpy(b_grad_host, b->d_grad, b->capacity * sizeof(float),
               cudaMemcpyDeviceToHost);
    cudaMemcpy(b_data_host, b->d_data, b->capacity * sizeof(float),
               cudaMemcpyDeviceToHost);
    for (int j = 0; j < b->capacity; j++) {
      b_data_host[j] -= learning_rate * b_grad_host[j];
    }
    cudaMemcpy(b->d_data, b_data_host, b->capacity * sizeof(float),
               cudaMemcpyHostToDevice);
    cudaMemset(b->d_grad, 0, b->capacity * sizeof(float));
    delete[] b_grad_host;
    delete[] b_data_host;

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
  realize(x);
  Tensor *y = tensor_sin(x);
  realize(y);

  Tensor *w = tensor_random(2, w_shape, DEVICE_CPU);
  realize(w);
  Tensor *b = tensor_random(2, b_shape, DEVICE_CPU);
  realize(b);

  int epochs = 1000;
  float learning_rate = 0.001f;

  for (int i = 0; i < epochs; i++) {
    Tensor *z = tensor_mm(x, w);
    Tensor *pred = tensor_add(z, b);
    Tensor *loss = loss_mse(pred, y);
    realize(loss);
    backward(loss);

    // Update parameters in-place
    for (int j = 0; j < w->capacity; j++) {
      w->data[j] -= learning_rate * w->grad[j];
      w->grad[j] = 0.0f;
    }
    for (int j = 0; j < b->capacity; j++) {
      b->data[j] -= learning_rate * b->grad[j];
      b->grad[j] = 0.0f;
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
