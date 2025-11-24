#include "tensor_ops.hpp"
#include <stdio.h>

Tensor* tensor_add(Tensor* a, Tensor* b){
    if(a == NULL || b == NULL){
        fprintf(stderr, "Error: Tensor is NULL");
        return NULL;
    }

    if(a->ndim != b->ndim){
        fprintf(stderr, "Error: Tensor dimensions must match");
        return NULL;
    }
    for(int i = 0; i < a->ndim; i++){
        if(a->shape[i] != b->shape[i]){
            fprintf(stderr, "Error: Tensor shapes must match");
            return NULL;
        }
    }
    Tensor* result = tensor_create(a->ndim, a->shape);
    for(int i = 0; i < a->capacity; i++){
        result->data[i] = a->data[i] + b->data[i];
    }

    result->_parents[0] = (Tensor*)a;
    result->_parents[1] = (Tensor*)b;
    
    result->_backward = [=](){
        for (int i=0; i < a->capacity; i++){
            a->grad[i] += result->grad[i];
            b->grad[i] += result->grad[i];
        }
    };
    return result;
}