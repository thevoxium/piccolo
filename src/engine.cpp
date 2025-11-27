#include "engine.hpp"
#include <algorithm>
#include <functional>
#include <unordered_set>

#ifdef USE_CUDA
#include <cuda_runtime.h>
#endif

static void build_topological_order(Tensor *root, std::vector<Tensor *> &topo) {
  if (root == NULL)
    return;

  std::unordered_set<Tensor *> visited;
  std::function<void(Tensor *)> dfs = [&](Tensor *v) {
    if (v == NULL || visited.find(v) != visited.end())
      return;
    visited.insert(v);
    for (int i = 0; i < 2; i++) {
      dfs(v->_parents[i]);
    }
    topo.push_back(v);
  };

  dfs(root);
  std::reverse(topo.begin(), topo.end());
}

void backward(Tensor *root) {
  if (root == NULL)
    return;
  std::vector<Tensor *> topo;
  build_topological_order(root, topo);

  realize(root);

  // Initialize root gradient to 1.0f for each element
#ifdef USE_CUDA
  if (root->device == DEVICE_GPU) {
    if (root->d_grad != NULL) {
      float *ones = new float[root->capacity];
      for (int i = 0; i < root->capacity; i++) {
        ones[i] = 1.0f;
      }
      cudaMemcpy(root->d_grad, ones, root->capacity * sizeof(float),
                 cudaMemcpyHostToDevice);
      delete[] ones;
    }
  } else
#endif
  {
    if (root->grad != NULL) {
      for (int i = 0; i < root->capacity; i++) {
        root->grad[i] = 1.0f;
      }
    }
  }

  for (Tensor *t : topo) {
    if (t != NULL && t->_backward) {
      t->_backward();
    }
  }
}

void free_graph(Tensor *root) {
  if (root == NULL)
    return;
  std::vector<Tensor *> topo;
  build_topological_order(root, topo);

  for (Tensor *t : topo) {
    if (t != NULL) {
      tensor_free(t);
    }
  }
}
