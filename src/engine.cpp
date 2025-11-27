#include "engine.hpp"
#include <unordered_set>

void backward(Tensor *root) {
  if (root == NULL)
    return;
  std::vector<Tensor *> topo;
  std::unordered_set<Tensor *> visited;

  std::function<void(Tensor *)> build_topo = [&](Tensor *v) {
    if (v == NULL)
      return;
    if (visited.find(v) == visited.end()) {
      visited.insert(v);
      for (int i = 0; i < 2; i++) {
        if (v->_parents[i] != NULL) {
          build_topo(v->_parents[i]);
        }
      }
      topo.push_back(v);
    }
  };

  build_topo(root);

  // Initialize root gradient to 1.0f for each element
  if (root->grad != NULL) {
    for (int i = 0; i < root->capacity; i++) {
      root->grad[i] = 1.0f;
    }
  }

  std::reverse(topo.begin(), topo.end());

  for (Tensor *t : topo) {
    if (t != NULL && t->_backward) {
      realize(t);
      t->_backward();
    }
  }
}

void free_graph(Tensor *root) {
  if (root == NULL)
    return;
  std::vector<Tensor *> topo;
  std::unordered_set<Tensor *> visited;

  std::function<void(Tensor *)> build_topo = [&](Tensor *v) {
    if (v == NULL)
      return;
    if (visited.find(v) == visited.end()) {
      visited.insert(v);
      for (int i = 0; i < 2; i++) {
        if (v->_parents[i] != NULL) {
          build_topo(v->_parents[i]);
        }
      }
      topo.push_back(v);
    }
  };

  build_topo(root);

  std::reverse(topo.begin(), topo.end());

  for (Tensor *t : topo) {
    if (t != NULL) {
      tensor_free(t);
    }
  }
}
