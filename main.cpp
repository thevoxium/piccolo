#include "src/tensor.hpp"
#include "src/tensor_ops.hpp"
#include "src/engine.hpp"
#include <iostream>
#include <cstdio>

int main() {
    // Create two tensors with shape [2, 2] filled with random values
    int shape[] = {2, 2};
    Tensor* a = tensor_random(2, shape);
    Tensor* b = tensor_random(2, shape);
    
    if (a == NULL || b == NULL) {
        std::cerr << "Error: Failed to create tensors" << std::endl;
        if (a != NULL) tensor_free(a);
        if (b != NULL) tensor_free(b);
        return 1;
    }
    
    // Add the tensors
    Tensor* result = tensor_sub(a, b);
    
    if (result == NULL) {
        std::cerr << "Error: tensor_sub failed" << std::endl;
        tensor_free(a);
        tensor_free(b);
        return 1;
    }
    
    std::cout << "Tensor a: " << *a << std::endl;
    std::cout << "Tensor b: " << *b << std::endl;
    std::cout << "Result (a - b): " << *result << std::endl;
    
    // Compute gradients
    backward(result);
    
    // Print gradients
    std::cout << "\nGradients:" << std::endl;
    if (result->grad != NULL) {
        std::cout << "Result grad: [";
        for (int i = 0; i < result->capacity; i++) {
            std::cout << result->grad[i];
            if (i < result->capacity - 1) std::cout << ", ";
        }
        std::cout << "]" << std::endl;
    }
    
    if (a->grad != NULL) {
        std::cout << "Tensor a grad: [";
        for (int i = 0; i < a->capacity; i++) {
            std::cout << a->grad[i];
            if (i < a->capacity - 1) std::cout << ", ";
        }
        std::cout << "]" << std::endl;
    }
    
    if (b->grad != NULL) {
        std::cout << "Tensor b grad: [";
        for (int i = 0; i < b->capacity; i++) {
            std::cout << b->grad[i];
            if (i < b->capacity - 1) std::cout << ", ";
        }
        std::cout << "]" << std::endl;
    }
    
    // Clean up
    tensor_free(result);
    tensor_free(a);
    tensor_free(b);
    
    return 0;
}

