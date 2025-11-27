---
title: Linear Regression
order: 1
---

# Linear Regression Tutorial

This tutorial demonstrates how to implement a simple linear regression model using Tensory. We'll fit a model to approximate $y = \sin(x)$ (locally linear approximation).

## The Code

Here is the complete example from `main.cpp`:

```cpp
#include "src/engine.hpp"
#include "src/loss.hpp"
#include "src/tensor.hpp"
#include "src/tensor_ops.hpp"
#include <cmath>
#include <iostream>

int main() {
  // 1. Setup Data Shapes
  int x_shape[2] = {1000, 1};
  int w_shape[2] = {1, 1};
  int b_shape[2] = {1000, 1};

  // 2. Initialize Data on CPU
  Tensor *x = tensor_random(2, x_shape, DEVICE_CPU);
  Tensor *y = tensor_sin(x); // Target: y = sin(x)
  realize(y); // Compute y values

  // 3. Initialize Parameters
  Tensor *w = tensor_random(2, w_shape, DEVICE_CPU);
  Tensor *b = tensor_random(2, b_shape, DEVICE_CPU);

  // 4. Move to GPU (Optional)
  // set_graph_device(y, DEVICE_GPU);
  // set_graph_device(w, DEVICE_GPU);
  // set_graph_device(b, DEVICE_GPU);

  // 5. Training Loop
  int epochs = 1000;
  float learning_rate = 0.001f;

  for (int i = 0; i < epochs; i++) {
    // Zero gradients
    zero_grad(w);
    zero_grad(b);

    // Forward Pass: z = x * w + b
    Tensor *z = tensor_mm(x, w);
    Tensor *pred = tensor_add(z, b);
    
    // Compute Loss (MSE)
    Tensor *loss = loss_mse(pred, y);
    
    // Execute Graph
    realize(loss);
    
    // Backward Pass
    backward(loss);

    // Update Parameters (SGD)
    // Note: If on GPU, need sync_to_host(w) before and sync_to_device(w) after
    for (int j = 0; j < w->capacity; j++) {
      w->data[j] -= learning_rate * w->grad[j];
    }
    for (int j = 0; j < b->capacity; j++) {
      b->data[j] -= learning_rate * b->grad[j];
    }

    // Print progress
    if (i % 100 == 0) {
      std::cout << "Epoch " << i << " Loss: " << loss->data[0] << std::endl;
    }

    // Cleanup intermediate tensors
    tensor_free(z);
    tensor_free(pred);
    tensor_free(loss);
  }

  // Final Cleanup
  tensor_free(x);
  tensor_free(y);
  tensor_free(w);
  tensor_free(b);

  return 0;
}
```

## Key Concepts

1.  **Defining the Graph**: Operations like `tensor_mm` and `tensor_add` do not execute immediately. They build a graph.
2.  **`realize(loss)`**: This triggers the actual computation.
3.  **`backward(loss)`**: Computes gradients for all tensors involved in the loss calculation.
4.  **Parameter Updates**: Currently done manually by accessing the `data` and `grad` arrays.
5.  **Memory**: Intermediate tensors created inside the loop (`z`, `pred`, `loss`) must be freed to avoid memory leaks.

