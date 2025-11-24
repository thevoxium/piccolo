#ifndef TENSOR_H
#define TENSOR_H

#include <stdlib.h>
#include <stdio.h>
#include <string.h>

typedef struct Tensor{
    float* data;
    int ndim;
    int* shape;
    int* strides;
    int capacity;
} Tensor;

Tensor* tensor_create(int ndim, int* shape);
void tensor_free(Tensor* t);

#endif