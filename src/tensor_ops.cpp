#include "tensor_ops.hpp"
#include <stdio.h>

Tensor* tensor_add(const Tensor* a, const Tensor* b){
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
    return result;
}