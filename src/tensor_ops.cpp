#include "tensor_ops.hpp"
#include <stdio.h>

Tensor* tensor_add(Tensor* a, Tensor* b){
    if(a == NULL || b == NULL){
        fprintf(stderr, "Error: Tensor is NULL\n");
        return NULL;
    }

    if(a->ndim != b->ndim){
        fprintf(stderr, "Error: Tensor dimensions must match\n");
        return NULL;
    }
    for(int i = 0; i < a->ndim; i++){
        if(a->shape[i] != b->shape[i]){
            fprintf(stderr, "Error: Tensor shapes must match\n");
            return NULL;
        }
    }
    if(a->data == NULL || b->data == NULL || a->grad == NULL || b->grad == NULL){
        fprintf(stderr, "Error: Tensor data or grad arrays are NULL\n");
        return NULL;
    }
    Tensor* result = tensor_create(a->ndim, a->shape);
    if(result == NULL){
        return NULL;
    }
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

Tensor* tensor_sub(Tensor* a, Tensor* b){
    if(a == NULL || b == NULL){
        fprintf(stderr, "Error: Tensor is NULL\n");
        return NULL;
    }

    if(a->ndim != b->ndim){
        fprintf(stderr, "Error: Tensor dimensions must match\n");
        return NULL;
    }
    for(int i = 0; i < a->ndim; i++){
        if(a->shape[i] != b->shape[i]){
            fprintf(stderr, "Error: Tensor shapes must match\n");
            return NULL;
        }
    }
    if(a->data == NULL || b->data == NULL || a->grad == NULL || b->grad == NULL){
        fprintf(stderr, "Error: Tensor data or grad arrays are NULL\n");
        return NULL;
    }
    Tensor* result = tensor_create(a->ndim, a->shape);
    if(result == NULL){
        return NULL;
    }
    for(int i = 0; i < a->capacity; i++){
        result->data[i] = a->data[i] - b->data[i];
    }

    result->_parents[0] = (Tensor*)a;
    result->_parents[1] = (Tensor*)b;
    
    result->_backward = [=](){
        for (int i=0; i < a->capacity; i++){
            a->grad[i] += result->grad[i];
            b->grad[i] -= result->grad[i];
        }
    };
    return result;
}

Tensor* tensor_scale(Tensor* a, float k){
    if(a == NULL){
        fprintf(stderr, "Error: Tensor is NULL\n");
        return NULL;
    }

    if(a->data == NULL || a->grad == NULL){
        fprintf(stderr, "Error: Tensor data or grad arrays are NULL\n");
        return NULL;
    }
    Tensor* result = tensor_create(a->ndim, a->shape);
    if(result == NULL){
        return NULL;
    }
    for(int i = 0; i < a->capacity; i++){
        result->data[i] = k * a->data[i];
    }

    result->_parents[0] = (Tensor*)a;
    result->_parents[1] = NULL;
    
    result->_backward = [=](){
        for (int i=0; i < a->capacity; i++){
            a->grad[i] += k * result->grad[i];
        }
    };
    return result;
}

Tensor* tensor_dot(Tensor* a, Tensor* b){
    if(a == NULL || b == NULL){
        fprintf(stderr, "Error: Tensor is NULL\n");
        return NULL;
    }

    // Check if both tensors are 1D
    if(a->ndim != 1 || b->ndim != 1){
        fprintf(stderr, "Error: tensor_dot only supports 1D vectors\n");
        return NULL;
    }

    if(a->shape[0] != b->shape[0]){
        fprintf(stderr, "Error: Vector lengths must match for dot product\n");
        return NULL;
    }
    
    if(a->data == NULL || b->data == NULL || a->grad == NULL || b->grad == NULL){
        fprintf(stderr, "Error: Tensor data or grad arrays are NULL\n");
        return NULL;
    }
    
    float dot_result = 0.0f;
    for(int i = 0; i < a->capacity; i++){
        dot_result += a->data[i] * b->data[i];
    }
    
    int result_shape[] = {1};
    Tensor* result = tensor_create(1, result_shape);
    if(result == NULL){
        return NULL;
    }
    result->data[0] = dot_result;
    
    result->_parents[0] = (Tensor*)a;
    result->_parents[1] = (Tensor*)b;
    
    result->_backward = [=](){
        float grad_val = result->grad[0];
        for (int i=0; i < a->capacity; i++){
            a->grad[i] += b->data[i] * grad_val;
            b->grad[i] += a->data[i] * grad_val;
        }
    };
    return result;
}