#include "tensor_ops.hpp"
#include <stdio.h>
#include <math.h>
#if __has_include(<cblas.h>)
extern "C" {
#include <cblas.h>
}
#elif __has_include(<Accelerate/Accelerate.h>)
#include <Accelerate/Accelerate.h>
#else
#error "cblas interface not available"
#endif

static Tensor* tensor_unary_result(Tensor* a, const char* op_name){
    if(a == NULL){
        fprintf(stderr, "Error (%s): Tensor is NULL\n", op_name);
        return NULL;
    }
    if(a->data == NULL || a->grad == NULL){
        fprintf(stderr, "Error (%s): Tensor data or grad arrays are NULL\n", op_name);
        return NULL;
    }
    Tensor* result = tensor_create(a->ndim, a->shape);
    if(result == NULL){
        fprintf(stderr, "Error (%s): Failed to create result tensor\n", op_name);
        return NULL;
    }
    result->_parents[0] = (Tensor*)a;
    result->_parents[1] = NULL;
    return result;
}

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

Tensor* tensor_mm(Tensor* a, Tensor* b){
    if(a == NULL || b == NULL){
        fprintf(stderr, "Error (tensor_mm): Tensor is NULL\n");
        return NULL;
    }
    if(a->ndim != 2 || b->ndim != 2){
        fprintf(stderr, "Error (tensor_mm): Both tensors must be 2D\n");
        return NULL;
    }
    if(a->shape[1] != b->shape[0]){
        fprintf(stderr, "Error (tensor_mm): Incompatible shapes (%d, %d) x (%d, %d)\n",
                a->shape[0], a->shape[1], b->shape[0], b->shape[1]);
        return NULL;
    }
    if(a->data == NULL || b->data == NULL || a->grad == NULL || b->grad == NULL){
        fprintf(stderr, "Error (tensor_mm): Tensor data or grad arrays are NULL\n");
        return NULL;
    }

    const int m = a->shape[0];
    const int k = a->shape[1];
    const int n = b->shape[1];

    int result_shape[2] = {m, n};
    Tensor* result = tensor_create(2, result_shape);
    if(result == NULL){
        fprintf(stderr, "Error (tensor_mm): Failed to create result tensor\n");
        return NULL;
    }

    cblas_sgemm(CblasRowMajor,
                CblasNoTrans,
                CblasNoTrans,
                m, n, k,
                1.0f,
                a->data, k,
                b->data, n,
                0.0f,
                result->data, n);

    result->_parents[0] = a;
    result->_parents[1] = b;

    result->_backward = [=](){
        if(a->grad != NULL){
            cblas_sgemm(CblasRowMajor,
                        CblasNoTrans,
                        CblasTrans,
                        m, k, n,
                        1.0f,
                        result->grad, n,
                        b->data, n,
                        1.0f,
                        a->grad, k);
        }
        if(b->grad != NULL){
            cblas_sgemm(CblasRowMajor,
                        CblasTrans,
                        CblasNoTrans,
                        k, n, m,
                        1.0f,
                        a->data, k,
                        result->grad, n,
                        1.0f,
                        b->grad, n);
        }
    };

    return result;
}

Tensor* tensor_neg(Tensor* a){
    Tensor* result = tensor_unary_result(a, "tensor_neg");
    if(result == NULL){
        return NULL;
    }
    for(int i = 0; i < a->capacity; i++){
        result->data[i] = -a->data[i];
    }
    result->_backward = [=](){
        for(int i = 0; i < a->capacity; i++){
            a->grad[i] -= result->grad[i];
        }
    };
    return result;
}

Tensor* tensor_log2(Tensor* a){
    Tensor* result = tensor_unary_result(a, "tensor_log2");
    if(result == NULL){
        return NULL;
    }
    for(int i = 0; i < a->capacity; i++){
        float val = a->data[i];
        if(val <= 0.0f){
            fprintf(stderr, "Warning (tensor_log2): log2 undefined for non-positive values, got %f\n", val);
            result->data[i] = -INFINITY;
        }else{
            result->data[i] = log2f(val);
        }
    }
    const float ln2 = logf(2.0f);
    result->_backward = [=](){
        for(int i = 0; i < a->capacity; i++){
            float val = a->data[i];
            if(val <= 0.0f){
                continue;
            }
            a->grad[i] += result->grad[i] / (val * ln2);
        }
    };
    return result;
}

Tensor* tensor_exp2(Tensor* a){
    Tensor* result = tensor_unary_result(a, "tensor_exp2");
    if(result == NULL){
        return NULL;
    }
    for(int i = 0; i < a->capacity; i++){
        result->data[i] = exp2f(a->data[i]);
    }
    const float ln2 = logf(2.0f);
    result->_backward = [=](){
        for(int i = 0; i < a->capacity; i++){
            a->grad[i] += result->grad[i] * result->data[i] * ln2;
        }
    };
    return result;
}

Tensor* tensor_sqrt(Tensor* a){
    Tensor* result = tensor_unary_result(a, "tensor_sqrt");
    if(result == NULL){
        return NULL;
    }
    for(int i = 0; i < a->capacity; i++){
        float val = a->data[i];
        if(val < 0.0f){
            fprintf(stderr, "Warning (tensor_sqrt): sqrt undefined for negative values, got %f\n", val);
            result->data[i] = NAN;
        } else {
            result->data[i] = sqrtf(val);
        }
    }
    result->_backward = [=](){
        for(int i = 0; i < a->capacity; i++){
            float val = a->data[i];
            if(val <= 0.0f || result->data[i] == 0.0f){
                continue;
            }
            float local_grad = 0.5f / result->data[i];
            a->grad[i] += result->grad[i] * local_grad;
        }
    };
    return result;
}

Tensor* tensor_sin(Tensor* a){
    Tensor* result = tensor_unary_result(a, "tensor_sin");
    if(result == NULL){
        return NULL;
    }
    for(int i = 0; i < a->capacity; i++){
        result->data[i] = sinf(a->data[i]);
    }
    result->_backward = [=](){
        for(int i = 0; i < a->capacity; i++){
            a->grad[i] += result->grad[i] * cosf(a->data[i]);
        }
    };
    return result;
}

Tensor* tensor_cos(Tensor* a){
    Tensor* result = tensor_unary_result(a, "tensor_cos");
    if(result == NULL){
        return NULL;
    }
    for(int i = 0; i < a->capacity; i++){
        result->data[i] = cosf(a->data[i]);
    }
    result->_backward = [=](){
        for(int i = 0; i < a->capacity; i++){
            a->grad[i] -= result->grad[i] * sinf(a->data[i]);
        }
    };
    return result;
}

Tensor* tensor_tan(Tensor* a){
    Tensor* result = tensor_unary_result(a, "tensor_tan");
    if(result == NULL){
        return NULL;
    }
    for(int i = 0; i < a->capacity; i++){
        result->data[i] = tanf(a->data[i]);
    }
    result->_backward = [=](){
        for(int i = 0; i < a->capacity; i++){
            float tangent = result->data[i];
            a->grad[i] += result->grad[i] * (1.0f + tangent * tangent);
        }
    };
    return result;
}

Tensor* tensor_trunc(Tensor* a){
    Tensor* result = tensor_unary_result(a, "tensor_trunc");
    if(result == NULL){
        return NULL;
    }
    for(int i = 0; i < a->capacity; i++){
        result->data[i] = truncf(a->data[i]);
    }
    result->_backward = [](){};
    return result;
}

Tensor* tensor_ceil(Tensor* a){
    Tensor* result = tensor_unary_result(a, "tensor_ceil");
    if(result == NULL){
        return NULL;
    }
    for(int i = 0; i < a->capacity; i++){
        result->data[i] = ceilf(a->data[i]);
    }
    result->_backward = [](){};
    return result;
}

Tensor* tensor_floor(Tensor* a){
    Tensor* result = tensor_unary_result(a, "tensor_floor");
    if(result == NULL){
        return NULL;
    }
    for(int i = 0; i < a->capacity; i++){
        result->data[i] = floorf(a->data[i]);
    }
    result->_backward = [](){};
    return result;
}

Tensor* tensor_round(Tensor* a){
    Tensor* result = tensor_unary_result(a, "tensor_round");
    if(result == NULL){
        return NULL;
    }
    for(int i = 0; i < a->capacity; i++){
        result->data[i] = roundf(a->data[i]);
    }
    result->_backward = [](){};
    return result;
}

Tensor* tensor_square(Tensor* a){
    Tensor* result = tensor_unary_result(a, "tensor_square");
    if(result == NULL){
        return NULL;
    }
    for(int i = 0; i < a->capacity; i++){
        float val = a->data[i];
        result->data[i] = val * val;
    }
    result->_backward = [=](){
        for(int i = 0; i < a->capacity; i++){
            a->grad[i] += result->grad[i] * 2.0f * a->data[i];
        }
    };
    return result;
}

Tensor* tensor_sign(Tensor* a){
    Tensor* result = tensor_unary_result(a, "tensor_sign");
    if(result == NULL){
        return NULL;
    }
    for(int i = 0; i < a->capacity; i++){
        float val = a->data[i];
        result->data[i] = (val > 0.0f) ? 1.0f : ((val < 0.0f) ? -1.0f : 0.0f);
    }
    result->_backward = [](){};
    return result;
}

Tensor* tensor_abs(Tensor* a){
    Tensor* result = tensor_unary_result(a, "tensor_abs");
    if(result == NULL){
        return NULL;
    }
    for(int i = 0; i < a->capacity; i++){
        result->data[i] = fabsf(a->data[i]);
    }
    result->_backward = [=](){
        for(int i = 0; i < a->capacity; i++){
            float val = a->data[i];
            float sign = 0.0f;
            if(val > 0.0f){
                sign = 1.0f;
            }else if(val < 0.0f){
                sign = -1.0f;
            }
            a->grad[i] += result->grad[i] * sign;
        }
    };
    return result;
}

Tensor* tensor_reciprocal(Tensor* a){
    Tensor* result = tensor_unary_result(a, "tensor_reciprocal");
    if(result == NULL){
        return NULL;
    }
    for(int i = 0; i < a->capacity; i++){
        float val = a->data[i];
        if(val == 0.0f){
            fprintf(stderr, "Warning (tensor_reciprocal): division by zero\n");
            result->data[i] = INFINITY;
        }else{
            result->data[i] = 1.0f / val;
        }
    }
    result->_backward = [=](){
        for(int i = 0; i < a->capacity; i++){
            float val = a->data[i];
            if(val == 0.0f){
                continue;
            }
            float local_grad = -1.0f / (val * val);
            a->grad[i] += result->grad[i] * local_grad;
        }
    };
    return result;
}

Tensor* tensor_pow(Tensor* a, float exponent){
    Tensor* result = tensor_unary_result(a, "tensor_pow");
    if(result == NULL){
        return NULL;
    }
    for(int i = 0; i < a->capacity; i++){
        result->data[i] = powf(a->data[i], exponent);
    }
    const float exponent_copy = exponent;
    result->_backward = [=](){
        for(int i = 0; i < a->capacity; i++){
            float base = a->data[i];
            float local_grad = 0.0f;
            if(base == 0.0f){
                if(exponent_copy > 1.0f){
                    local_grad = 0.0f;
                }else if(exponent_copy == 1.0f){
                    local_grad = 1.0f;
                }else{
                    continue;
                }
            }else{
                local_grad = exponent_copy * powf(base, exponent_copy - 1.0f);
            }
            a->grad[i] += result->grad[i] * local_grad;
        }
    };
    return result;
}

Tensor* tensor_exp(Tensor* a){
    Tensor* result = tensor_unary_result(a, "tensor_exp");
    if(result == NULL){
        return NULL;
    }
    for(int i = 0; i < a->capacity; i++){
        result->data[i] = expf(a->data[i]);
    }
    result->_backward = [=](){
        for(int i = 0; i < a->capacity; i++){
            a->grad[i] += result->grad[i] * result->data[i];
        }
    };
    return result;
}

Tensor* tensor_log(Tensor* a){
    Tensor* result = tensor_unary_result(a, "tensor_log");
    if(result == NULL){
        return NULL;
    }
    for(int i = 0; i < a->capacity; i++){
        float val = a->data[i];
        if(val <= 0.0f){
            fprintf(stderr, "Warning (tensor_log): log undefined for non-positive values, got %f\n", val);
            result->data[i] = -INFINITY;
        }else{
            result->data[i] = logf(val);
        }
    }
    result->_backward = [=](){
        for(int i = 0; i < a->capacity; i++){
            float val = a->data[i];
            if(val <= 0.0f){
                continue;
            }
            a->grad[i] += result->grad[i] / val;
        }
    };
    return result;
}

Tensor* tensor_aggregate(Tensor* a){
    if(a == NULL){
        fprintf(stderr, "Error: Tensor is NULL\n");
        return NULL;
    }

    if(a->data == NULL || a->grad == NULL){
        fprintf(stderr, "Error: Tensor data or grad arrays are NULL\n");
        return NULL;
    }
    
    float sum = 0.0f;
    for(int i = 0; i < a->capacity; i++){
        sum += a->data[i];
    }
    
    int result_shape[] = {1};
    Tensor* result = tensor_create(1, result_shape);
    if(result == NULL){
        return NULL;
    }
    result->data[0] = sum;
    
    result->_parents[0] = (Tensor*)a;
    result->_parents[1] = NULL;
    
    result->_backward = [=](){
        float grad_val = result->grad[0];
        for (int i=0; i < a->capacity; i++){
            a->grad[i] += grad_val;
        }
    };
    return result;
}