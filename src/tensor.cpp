#include "tensor.hpp"
#include <climits>
#include <functional>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <string>
#include <vector>

Tensor *tensor_create(int ndim, int *shape, Device device) {
  if (ndim <= 0 || shape == NULL) {
    fprintf(stderr, "Error: Check the input parameters, ndim: %d, shape: %p\n",
            ndim, shape);
    return NULL;
  }

  for (int i = 0; i < ndim; i++) {
    if (shape[i] <= 0) {
      fprintf(stderr, "Error: shape[%d] must be positive, got %d\n", i,
              shape[i]);
      return NULL;
    }
  }

  // Calculate capacity from shapes with overflow protection
  int capacity = 1;
  for (int i = 0; i < ndim; i++) {
    // Check for overflow before multiplying
    if (shape[i] > 0 && capacity > INT_MAX / shape[i]) {
      fprintf(stderr, "Error: Capacity calculation would overflow\n");
      return NULL;
    }
    capacity *= shape[i];
  }

  Tensor *t = (Tensor *)malloc(sizeof(Tensor));
  if (t == NULL) {
    fprintf(stderr, "Error: Failed to allocate memory for Tensor\n");
    return NULL;
  }

  t->data = (float *)malloc(capacity * sizeof(float));
  if (t->data == NULL) {
    fprintf(stderr, "Error: Failed to allocate memory for data\n");
    free(t);
    return NULL;
  }

  // Allocate and copy the shape array
  t->shape = (int *)malloc(ndim * sizeof(int));
  if (t->shape == NULL) {
    fprintf(stderr, "Error: Failed to allocate memory for shape\n");
    free(t->data);
    free(t);
    return NULL;
  }
  memcpy(t->shape, shape, ndim * sizeof(int));

  // Allocate and calculate strides (row-major order)
  t->strides = (int *)malloc(ndim * sizeof(int));
  if (t->strides == NULL) {
    fprintf(stderr, "Error: Failed to allocate memory for strides\n");
    free(t->shape);
    free(t->data);
    free(t);
    return NULL;
  }

  // Calculate strides: stride[i] = product of shape[i+1] to shape[ndim-1]
  // For row-major order, last dimension has stride 1
  t->strides[ndim - 1] = 1;
  for (int i = ndim - 2; i >= 0; i--) {
    t->strides[i] = t->strides[i + 1] * shape[i + 1];
  }

  t->ndim = ndim;
  t->capacity = capacity;

  t->_parents = (Tensor **)malloc(2 * sizeof(Tensor *));
  if (t->_parents == NULL) {
    fprintf(stderr, "Error: Failed to allocate memory for parents\n");
    free(t->strides);
    free(t->shape);
    free(t->data);
    free(t);
    return NULL;
  }
  t->_parents[0] = NULL;
  t->_parents[1] = NULL;
  t->_backward = NULL;

  t->grad = (float *)malloc(capacity * sizeof(float));
  if (t->grad == NULL) {
    fprintf(stderr, "Error: Failed to allocate memory for grad\n");
    free(t->_parents);
    free(t->strides);
    free(t->shape);
    free(t->data);
    free(t);
    return NULL;
  }
  memset(t->grad, 0, capacity * sizeof(float));

  t->device = device;

  if (device == DEVICE_GPU) {
#ifdef USE_CUDA
    t->d_data = nullptr;
    t->d_grad = nullptr;
    printf("cuda available, allocating on device\n");
#else
    fprintf(
        stderr,
        "Warning: CUDA is not available. Falling back to CPU allocation.\n");
    t->device = DEVICE_CPU;
#endif
  }

  return t;
}

Tensor *tensor_ones(int ndim, int *shape, Device device) {
  Tensor *t = tensor_create(ndim, shape, device);
  if (t == NULL) {
    return NULL;
  }

  for (int i = 0; i < t->capacity; i++) {
    t->data[i] = 1.0f;
  }

  return t;
}

Tensor *tensor_zeroes(int ndim, int *shape, Device device) {
  Tensor *t = tensor_create(ndim, shape, device);
  if (t == NULL) {
    return NULL;
  }

  memset(t->data, 0, t->capacity * sizeof(float));

  return t;
}

Tensor *tensor_random(int ndim, int *shape, Device device) {
  Tensor *t = tensor_create(ndim, shape, device);
  if (t == NULL) {
    return NULL;
  }

  for (int i = 0; i < t->capacity; i++) {
    t->data[i] = (float)rand() / (float)RAND_MAX;
  }

  return t;
}

Tensor *tensor_from_data(int ndim, int *shape, float *data, Device device) {
  if (data == NULL) {
    fprintf(stderr, "Error: data array is NULL\n");
    return NULL;
  }

  Tensor *t = tensor_create(ndim, shape, device);
  if (t == NULL) {
    return NULL;
  }

  memcpy(t->data, data, t->capacity * sizeof(float));

  return t;
}

void tensor_free(Tensor *t) {
  if (t == NULL) {
    return;
  }
  if (t->data != NULL) {
    free(t->data);
  }
  if (t->shape != NULL) {
    free(t->shape);
  }
  if (t->strides != NULL) {
    free(t->strides);
  }
  if (t->_parents != NULL) {
    free(t->_parents);
  }
  if (t->grad != NULL) {
    free(t->grad);
  }
  free(t);
}

std::ostream &operator<<(std::ostream &os, const Tensor &t) {
  os << "Tensor(";

  // Check for NULL pointers before accessing
  if (t.data == NULL || t.shape == NULL || t.strides == NULL) {
    os << "INVALID_TENSOR";
    return os;
  }

  std::function<void(size_t, size_t, const std::vector<size_t> &)>
      print_recursive =
          [&](size_t offset, size_t dim, const std::vector<size_t> &coords) {
            if (dim >= static_cast<size_t>(t.ndim))
              return;
            if (dim == static_cast<size_t>(t.ndim) - 1) {
              // Print the innermost dimension
              os << "[";
              for (size_t i = 0; i < static_cast<size_t>(t.shape[dim]); ++i) {
                if (offset + i < static_cast<size_t>(t.capacity)) {
                  os << t.data[offset + i];
                }
                if (i < static_cast<size_t>(t.shape[dim]) - 1)
                  os << ", ";
              }
              os << "]";
            } else {
              os << "[";
              for (size_t i = 0; i < static_cast<size_t>(t.shape[dim]); ++i) {
                if (i > 0)
                  os << ",\n" << std::string(dim + 1, ' ');
                size_t new_offset = offset + i * t.strides[dim];
                if (new_offset < static_cast<size_t>(t.capacity)) {
                  print_recursive(new_offset, dim + 1, coords);
                }
              }
              os << "]";
            }
          };

  print_recursive(0, 0, {});
  os << ", shape: (";
  for (int i = 0; i < t.ndim; ++i) {
    os << t.shape[i];
    if (i < t.ndim - 1)
      os << ", ";
  }
  os << "))";

  return os;
}
