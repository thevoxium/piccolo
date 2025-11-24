#include "src/tensor.h"


int main(){
    int shape[] = {2, 3};
    Tensor* t = tensor_create(2, shape);
    if(t == NULL) {
        fprintf(stderr, "Error: Failed to create tensor\n");
        return -1;
    }
    std::cout << *t << std::endl;
    tensor_free(t);
    return 0;
}