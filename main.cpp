#include "src/tensor.hpp"
#include "src/tensor_ops.hpp"
#include "src/engine.hpp"
#include <iostream>
#include <cstdio>

int main() {
    // Multiply two compatible matrices using BLAS-backed tensor_mm
    int shape_a[] = {2, 3};
    int shape_b[] = {3, 2};
    Tensor* a = tensor_random(2, shape_a);
    Tensor* b = tensor_random(2, shape_b);
    
    if (a == NULL || b == NULL) {
        std::cerr << "Error: Failed to create tensors" << std::endl;
        if (a != NULL) tensor_free(a);
        if (b != NULL) tensor_free(b);
        return 1;
    }
    
    Tensor* result = tensor_mm(a, b);
    
    if (result == NULL) {
        std::cerr << "Error: tensor_mm failed" << std::endl;
        tensor_free(a);
        tensor_free(b);
        return 1;
    }
    
    std::cout << "Tensor a: " << *a << std::endl;
    std::cout << "Tensor b: " << *b << std::endl;
    std::cout << "Result (a mm b): " << *result << std::endl;
    
    // Compute gradients through the matmul
    backward(result);
    
    auto print_grad = [](const char* label, Tensor* t) {
        if (t == NULL || t->grad == NULL) return;
        std::cout << label << ": [";
        for (int i = 0; i < t->capacity; i++) {
            std::cout << t->grad[i];
            if (i < t->capacity - 1) std::cout << ", ";
        }
        std::cout << "]" << std::endl;
    };
    
    std::cout << "\nGradients:" << std::endl;
    print_grad("Result grad", result);
    print_grad("Tensor a grad", a);
    print_grad("Tensor b grad", b);
    
    // Clean up
    tensor_free(result);
    tensor_free(a);
    tensor_free(b);
    
    return 0;
}

