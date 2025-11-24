#include "tensor.hpp"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <functional>
#include <vector>
#include <string>

Tensor* tensor_create(int ndim, int* shape) {
    if(ndim <= 0 || shape == NULL) {
        fprintf(stderr, "Error: Check the input parameters, ndim: %d, shape: %p", ndim, shape);
        return NULL;
    }
    
    for(int i = 0; i < ndim; i++) {
        if(shape[i] <= 0) {
            fprintf(stderr, "Error: shape[%d] must be positive, got %d\n", i, shape[i]);
            return NULL;
        }
    }
    
    // Calculate capacity from shapes
    int capacity = 1;
    for(int i = 0; i < ndim; i++) {
        capacity *= shape[i];
    }
    
    Tensor* t = (Tensor*)malloc(sizeof(Tensor));
    if(t == NULL) {
        fprintf(stderr, "Error: Failed to allocate memory for Tensor\n");
        return NULL;
    }
    
    t->data = (float*)malloc(capacity * sizeof(float));
    if(t->data == NULL) {
        fprintf(stderr, "Error: Failed to allocate memory for data\n");
        free(t);
        return NULL;
    }
    
    // Allocate and copy the shape array
    t->shape = (int*)malloc(ndim * sizeof(int));
    if(t->shape == NULL) {
        fprintf(stderr, "Error: Failed to allocate memory for shape\n");
        free(t->data);
        free(t);
        return NULL;
    }
    memcpy(t->shape, shape, ndim * sizeof(int));
    
    // Allocate and calculate strides (row-major order)
    t->strides = (int*)malloc(ndim * sizeof(int));
    if(t->strides == NULL) {
        fprintf(stderr, "Error: Failed to allocate memory for strides\n");
        free(t->shape);
        free(t->data);
        free(t);
        return NULL;
    }
    
    // Calculate strides: stride[i] = product of shape[i+1] to shape[ndim-1]
    // For row-major order, last dimension has stride 1
    t->strides[ndim - 1] = 1;
    for(int i = ndim - 2; i >= 0; i--) {
        t->strides[i] = t->strides[i + 1] * shape[i + 1];
    }
    
    t->ndim = ndim;
    t->capacity = capacity;
    
    t->_parents = (Tensor**)malloc(2 * sizeof(Tensor*));
    if(t->_parents == NULL) {
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
    
    t->grad = (float*)malloc(capacity * sizeof(float));
    if(t->grad == NULL) {
        fprintf(stderr, "Error: Failed to allocate memory for grad\n");
        free(t->data);
        free(t->shape);
        free(t->strides);
        free(t->_parents);
        free(t);
        return NULL;
    }
    memset(t->grad, 0, capacity * sizeof(float));


    return t;
}

void tensor_free(Tensor* t) {
    if(t == NULL) {
        return;
    }
    if(t->data != NULL) {
        free(t->data);
    }
    if(t->shape != NULL) {
        free(t->shape);
    }
    if(t->strides != NULL) {
        free(t->strides);
    }
    if(t->_parents != NULL) {
        free(t->_parents);
    }
    if(t->grad != NULL) {
        free(t->grad);
    }
    free(t);
}

std::ostream &operator<<(std::ostream &os, const Tensor &t) {
  os << "Tensor(";

  std::function<void(size_t, size_t, const std::vector<size_t> &)>
      print_recursive =
          [&](size_t offset, size_t dim, const std::vector<size_t> &coords) {
            if (dim == static_cast<size_t>(t.ndim) - 1) {
              // Print the innermost dimension
              os << "[";
              for (size_t i = 0; i < static_cast<size_t>(t.shape[dim]); ++i) {
                os << t.data[offset + i];
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
                print_recursive(new_offset, dim + 1, coords);
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
