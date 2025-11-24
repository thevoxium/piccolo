#ifndef TENSOR_H
#define TENSOR_H

#include <iostream>
#include <functional>

typedef struct Tensor{
    float* data;
    int ndim;
    int* shape;
    int* strides;
    int capacity;

    Tensor** _parents;
    std::function<void()> _backward;
} Tensor;

Tensor* tensor_create(int ndim, int* shape);
void tensor_free(Tensor* t);
std::ostream &operator<<(std::ostream &os, const Tensor &t);

#endif