#include "src/engine.hpp"
#include "src/tensor.hpp"
#include "src/tensor_ops.hpp"
#include <cmath>

int main() {
  int shape_a[2] = {2, 3};
  int shape_b[2] = {3, 2};

  float data_a[6] = {1.0f, 2.0f, 3.0f, //
                     4.0f, 5.0f, 6.0f};
  float data_b[6] = {7.0f, 8.0f, //
                     9.0f, 10.0f, //
                     11.0f, 12.0f};
  float expected[4] = {58.0f, 64.0f, //
                       139.0f, 154.0f};

  Tensor *a_cpu = tensor_from_data(2, shape_a, data_a, DEVICE_CPU);
  Tensor *b_cpu = tensor_from_data(2, shape_b, data_b, DEVICE_CPU);
  Tensor *res_cpu = tensor_mm(a_cpu, b_cpu);
  realize(res_cpu);

  bool cpu_ok = true;
  for (int i = 0; i < res_cpu->capacity; i++) {
    if (std::fabs(res_cpu->data[i] - expected[i]) > 1e-4f) {
      cpu_ok = false;
      break;
    }
  }

  std::cout << "CPU tensor_mm result:\n" << *res_cpu << std::endl;
  std::cout << "CPU check: " << (cpu_ok ? "PASSED" : "FAILED") << std::endl;

#ifdef USE_CUDA
  Tensor *a_gpu = tensor_from_data(2, shape_a, data_a, DEVICE_GPU);
  Tensor *b_gpu = tensor_from_data(2, shape_b, data_b, DEVICE_GPU);
  Tensor *res_gpu = tensor_mm(a_gpu, b_gpu);
  realize(res_gpu);

  CUDA_CHECK(cudaMemcpy(res_gpu->data, res_gpu->d_data,
                        res_gpu->capacity * sizeof(float),
                        cudaMemcpyDeviceToHost));

  bool gpu_ok = true;
  for (int i = 0; i < res_gpu->capacity; i++) {
    if (std::fabs(res_gpu->data[i] - expected[i]) > 1e-4f) {
      gpu_ok = false;
      break;
    }
  }

  std::cout << "GPU tensor_mm result:\n" << *res_gpu << std::endl;
  std::cout << "GPU check: " << (gpu_ok ? "PASSED" : "FAILED") << std::endl;
#endif

  tensor_free(res_cpu);
  tensor_free(a_cpu);
  tensor_free(b_cpu);

#ifdef USE_CUDA
  tensor_free(res_gpu);
  tensor_free(a_gpu);
  tensor_free(b_gpu);

  return (cpu_ok && gpu_ok) ? 0 : 1;
#else
  return cpu_ok ? 0 : 1;
#endif

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
