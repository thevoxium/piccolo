#ifndef ENGINE_H
#define ENGINE_H
#include "tensor.hpp"

void backward(Tensor* root);
void free_graph(Tensor* root);

#endif