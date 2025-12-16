#ifndef TENSOR_H
#define TENSOR_H

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

typedef struct Tensor {
  float *data;
  float *grad;
  int ndim;
  int *shape;
  int *strides;
  int capacity;

  Tensor **_parents;
  std::function<void()> _backward;
  std::function<void()> _forward;
  bool _realized;
} Tensor;

Tensor *tensor_create(int ndim, int *shape);
void tensor_free(Tensor *t);
std::ostream &operator<<(std::ostream &os, const Tensor &t);

Tensor *tensor_ones(int ndim, int *shape);
Tensor *tensor_zeroes(int ndim, int *shape);
Tensor *tensor_random(int ndim, int *shape);
Tensor *tensor_from_data(int ndim, int *shape, float *data);

// Realize the tensor and all its dependencies in the computation graph
void realize(Tensor *t);

// Zero gradients for a single tensor
void zero_grad(Tensor *t);

// Zero gradients for entire computation graph
void zero_graph_grad(Tensor *root);

#endif
