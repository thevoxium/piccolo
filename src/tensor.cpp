#include "tensor.hpp"
#include <algorithm>
#include <climits>
#include <functional>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unordered_set>
#include <vector>

Tensor *tensor_create(int ndim, int *shape) {
  if (ndim <= 0 || shape == NULL) {
    ERROR_RETURN_NULL("Error: Check for ndim or shape of the tensor \n");
  }

  for (int i = 0; i < ndim; i++) {
    if (shape[i] <= 0) {
      ERROR_RETURN_NULL("Error: Shape can't be negative\n");
    }
  }

  int capacity = 1;
  for (int i = 0; i < ndim; i++) {
    if (shape[i] > 0 && capacity > INT_MAX / shape[i]) {
      ERROR_RETURN_NULL("Error: Capacity overflow for tensor allocation\n");
    }
    capacity *= shape[i];
  }

  // Using new instead of malloc because std::function requires proper
  // constructor/destructor calls
  Tensor *t = new Tensor();
  CHECK_NULL_T(t);

  t->data = (float *)malloc(capacity * sizeof(float));
  if (t->data == NULL) {
    delete t;
    return NULL;
  }

  t->shape = (int *)malloc(ndim * sizeof(int));
  if (t->shape == NULL) {
    free(t->data);
    delete t;
    return NULL;
  }
  memcpy(t->shape, shape, ndim * sizeof(int));

  t->strides = (int *)malloc(ndim * sizeof(int));
  if (t->strides == NULL) {
    free(t->shape);
    free(t->data);
    delete t;
    return NULL;
  }

  t->strides[ndim - 1] = 1;
  for (int i = ndim - 2; i >= 0; i--) {
    t->strides[i] = t->strides[i + 1] * shape[i + 1];
  }

  t->ndim = ndim;
  t->capacity = capacity;

  t->_parents = (Tensor **)malloc(2 * sizeof(Tensor *));
  if (t->_parents == NULL) {
    free(t->strides);
    free(t->shape);
    free(t->data);
    delete t;
    return NULL;
  }
  t->_parents[0] = NULL;
  t->_parents[1] = NULL;
  t->_backward = NULL;
  t->_forward = NULL;
  t->_realized = false;

  t->grad = (float *)malloc(capacity * sizeof(float));
  if (t->grad == NULL) {
    free(t->_parents);
    free(t->strides);
    free(t->shape);
    free(t->data);
    delete t;
    return NULL;
  }
  memset(t->grad, 0, capacity * sizeof(float));

  return t;
}

Tensor *tensor_ones(int ndim, int *shape) {
  Tensor *t = tensor_create(ndim, shape);
  CHECK_NULL_T(t);

  for (int i = 0; i < t->capacity; i++) {
    t->data[i] = 1.0f;
  }

  return t;
}

Tensor *tensor_zeroes(int ndim, int *shape) {
  Tensor *t = tensor_create(ndim, shape);
  CHECK_NULL_T(t);

  memset(t->data, 0, t->capacity * sizeof(float));

  return t;
}

Tensor *tensor_random(int ndim, int *shape) {
  Tensor *t = tensor_create(ndim, shape);
  CHECK_NULL_T(t);

  for (int i = 0; i < t->capacity; i++) {
    t->data[i] = (float)rand() / (float)RAND_MAX;
  }

  return t;
}

Tensor *tensor_from_data(int ndim, int *shape, float *data) {
  if (data == NULL) {
    ERROR_RETURN_NULL("Error: data array is NULL\n");
  }

  Tensor *t = tensor_create(ndim, shape);
  CHECK_NULL_T(t);

  memcpy(t->data, data, t->capacity * sizeof(float));

  return t;
}

// Helper function to build topological order for graph traversal
static void build_topo_order(Tensor *root, std::vector<Tensor *> &topo,
                             std::unordered_set<Tensor *> &visited) {
  if (root == NULL || visited.find(root) != visited.end()) {
    return;
  }
  visited.insert(root);

  // Visit parents first (dependencies)
  if (root->_parents != NULL) {
    for (int i = 0; i < 2; i++) {
      if (root->_parents[i] != NULL) {
        build_topo_order(root->_parents[i], topo, visited);
      }
    }
  }

  // Add this node after its dependencies
  topo.push_back(root);
}

void realize(Tensor *t) {
  if (t == NULL) {
    ERROR_MSG("Error: Realizing a NULL Tensor\n");
    return;
  }

  // Build topological order: leaves first, root last
  std::vector<Tensor *> topo;
  std::unordered_set<Tensor *> visited;
  build_topo_order(t, topo, visited);

  // Realize tensors in topological order (leaves first)
  for (Tensor *tensor : topo) {
    if (tensor->_realized) {
      continue;
    }
    if (tensor->_forward != NULL) {
      tensor->_forward();
    }
    tensor->_realized = true;
  }
}

void ensure_realized(Tensor *t) {
  if (t == NULL) {
    ERROR_MSG("Error: ensure_realized called with NULL tensor\n");
    return;
  }
  if (!t->_realized) {
    realize(t);
  }
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

  delete t;
}

void zero_grad(Tensor *t) {
  if (t == NULL) {
    return;
  }

  if (t->grad != NULL) {
    memset(t->grad, 0, t->capacity * sizeof(float));
  }
}

void zero_graph_grad(Tensor *root) {
  if (root == NULL) {
    return;
  }

  std::vector<Tensor *> topo;
  std::unordered_set<Tensor *> visited;
  build_topo_order(root, topo, visited);

  for (Tensor *t : topo) {
    zero_grad(t);
  }
}
