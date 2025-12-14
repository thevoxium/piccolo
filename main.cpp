#include "src/engine.hpp"
#include "src/loss.hpp"
#include "src/tensor.hpp"
#include "src/tensor_ops.hpp"

int main() {
  // Linear regression example
  int x_shape[2] = {1000, 1};
  int w_shape[2] = {1, 1};
  int b_shape[2] = {1000, 1};

  Tensor *x = tensor_random(2, x_shape);
  Tensor *y = tensor_sin(x);
  realize(y);

  Tensor *w = tensor_random(2, w_shape);
  Tensor *b = tensor_random(2, b_shape);

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
    realize(loss);
    backward(loss);

    // Update parameters in-place
    for (int j = 0; j < w->capacity; j++) {
      w->data[j] -= learning_rate * w->grad[j];
    }
    for (int j = 0; j < b->capacity; j++) {
      b->data[j] -= learning_rate * b->grad[j];
    }

    if (i % 100 == 0 || i == epochs - 1) {
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

  std::cout << "Linear Regression completed!" << std::endl;
  return 0;
}
