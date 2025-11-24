#ifndef TENSOR_H
#define TENSOR_H

#include <iostream>
#include <functional>

typedef struct Tensor{
    float* data;
    float* grad;
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

Tensor* tensor_ones(int ndim, int* shape);
Tensor* tensor_zeroes(int ndim, int* shape);
Tensor* tensor_random(int ndim, int* shape);
Tensor* tensor_from_data(int ndim, int* shape, float* data);



#endif