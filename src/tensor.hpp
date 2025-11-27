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
  std::function<void()> _forward;
  bool _realized;
  bool _host_dirty;
  bool _device_dirty;

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

// Realize the tensor and all its dependencies in the computation graph
void realize(Tensor *t);
void ensure_realized(Tensor *t);

// Set device for the entire computation graph rooted at t
void set_graph_device(Tensor *root, Device device);

// Sync data/grad between host (CPU) and device (GPU) for a single tensor
void sync_to_host(Tensor *t);
void sync_to_device(Tensor *t);

// Sync data/grad between host and device for entire computation graph
void sync_graph_to_host(Tensor *root);
void sync_graph_to_device(Tensor *root);

// Zero gradients for a single tensor
void zero_grad(Tensor *t);

// Zero gradients for entire computation graph
void zero_graph_grad(Tensor *root);

#endif
