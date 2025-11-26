#ifndef TENSOR_H
#define TENSOR_H

#ifdef USE_CUDA
#include "cuda/cuda_utils.hpp"
#endif //

#include <functional>
#include <iostream>

#define CHECK_NULL_T(t)                                                        \
  if (t == NULL) {                                                             \
    fprintf(stderr, "Error: Tensor is Null\n");                                \
    return NULL;                                                               \
  }

#define ERROR_RETURN_NULL(s)                                                   \
  fprintf(stderr, s);                                                          \
  return NULL;

#define ERROR_MSG(s) fprintf(stderr, s);

typedef enum Device { DEVICE_CPU, DEVICE_GPU } Device;

typedef struct Tensor {
  float *data;
  float *grad;
  int ndim;
  int *shape;
  int *strides;
  int capacity;
  Device device;

  Tensor **_parents;
  std::function<void()> _backward;

#ifdef USE_CUDA
  float *d_data;
  float *d_grad;
#endif //

} Tensor;

Tensor *tensor_create(int ndim, int *shape, Device = DEVICE_CPU);
void tensor_free(Tensor *t);
std::ostream &operator<<(std::ostream &os, const Tensor &t);

Tensor *tensor_ones(int ndim, int *shape, Device = DEVICE_CPU);
Tensor *tensor_zeroes(int ndim, int *shape, Device = DEVICE_CPU);
Tensor *tensor_random(int ndim, int *shape, Device = DEVICE_CPU);
Tensor *tensor_from_data(int ndim, int *shape, float *data,
                         Device = DEVICE_CPU);

#endif
