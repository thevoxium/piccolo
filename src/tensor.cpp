#include "tensor.hpp"
#include <climits>
#include <functional>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

Tensor *tensor_create(int ndim, int *shape, Device device) {
  if (ndim <= 0 || shape == NULL) {
    ERROR_RETURN_NULL("Error: Check for ndim or shape of the tensor \n");
  }

  for (int i = 0; i < ndim; i++) {
    if (shape[i] <= 0) {
      ERROR_RETURN_NULL("Error: Shape can't be negative\n");
    }
  }

  int capacity = 1;
  for (int i = 0; i < ndim; i++) {
    if (shape[i] > 0 && capacity > INT_MAX / shape[i]) {
      ERROR_RETURN_NULL("Error: Capcity overflow for tensor allocation\n");
    }
    capacity *= shape[i];
  }

  // Initially, I was using malloc here, it was working fine on mac locally
  // but when run on colab, it was seg fault all over the place
  // now i did not know the issue, what was causing it
  // LLM suggested to use new and delete here because of the backward
  // std::function it says since malloc/free does not call constructor and
  // destructor, it is allocating garbage to the backward() new and delete
  // correctly calls constructor and destructor correctly need to check this
  // more
  Tensor *t = new Tensor();
  CHECK_NULL_T(t);

  t->data = (float *)malloc(capacity * sizeof(float));
  if (t->data == NULL) {
    delete t;
    return NULL;
  }

  t->shape = (int *)malloc(ndim * sizeof(int));
  if (t->shape == NULL) {
    free(t->data);
    delete t;
    return NULL;
  }
  memcpy(t->shape, shape, ndim * sizeof(int));

  t->strides = (int *)malloc(ndim * sizeof(int));
  if (t->strides == NULL) {
    free(t->shape);
    free(t->data);
    delete t;
    return NULL;
  }

  t->strides[ndim - 1] = 1;
  for (int i = ndim - 2; i >= 0; i--) {
    t->strides[i] = t->strides[i + 1] * shape[i + 1];
  }

  t->ndim = ndim;
  t->capacity = capacity;

  t->_parents = (Tensor **)malloc(2 * sizeof(Tensor *));
  if (t->_parents == NULL) {
    free(t->strides);
    free(t->shape);
    free(t->data);
    delete t;
    return NULL;
  }
  t->_parents[0] = NULL;
  t->_parents[1] = NULL;
  t->_backward = NULL;
  t->_forward = NULL;
  t->_realized = false;

  t->grad = (float *)malloc(capacity * sizeof(float));
  if (t->grad == NULL) {
    free(t->_parents);
    free(t->strides);
    free(t->shape);
    free(t->data);
    delete t;
    return NULL;
  }
  memset(t->grad, 0, capacity * sizeof(float));

  t->device = device;

  if (device == DEVICE_GPU) {
#ifdef USE_CUDA
    CUDA_CHECK(cudaMalloc(&t->d_data, capacity * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&t->d_grad, capacity * sizeof(float)));
    CUDA_CHECK(cudaMemset(t->d_data, 0, capacity * sizeof(float)));
    CUDA_CHECK(cudaMemset(t->d_grad, 0, capacity * sizeof(float)));
#else
    ERROR_MSG("Error: GPU as Device, but build without CUDA Flag\n");
    t->device = DEVICE_CPU;
#endif
  }

  return t;
}

Tensor *tensor_ones(int ndim, int *shape, Device device) {
  Tensor *t = tensor_create(ndim, shape, device);
  CHECK_NULL_T(t);

  for (int i = 0; i < t->capacity; i++) {
    t->data[i] = 1.0f;
  }

  if (t->device == DEVICE_GPU) {
#ifdef USE_CUDA
    CUDA_CHECK(cudaMemcpy(t->d_data, t->data, t->capacity * sizeof(float),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(t->d_grad, t->grad, t->capacity * sizeof(float),
                          cudaMemcpyHostToDevice));
#else
    ERROR_MSG("Error: GPU as Device, but build without CUDA Flag\n");
    t->device = DEVICE_CPU;
#endif
  }

  return t;
}

Tensor *tensor_zeroes(int ndim, int *shape, Device device) {
  Tensor *t = tensor_create(ndim, shape, device);
  CHECK_NULL_T(t);

  memset(t->data, 0, t->capacity * sizeof(float));

  if (t->device == DEVICE_GPU) {
#ifdef USE_CUDA
    CUDA_CHECK(cudaMemcpy(t->d_data, t->data, t->capacity * sizeof(float),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(t->d_grad, t->grad, t->capacity * sizeof(float),
                          cudaMemcpyHostToDevice));
#else
    ERROR_MSG("Error: GPU as Device, but build without CUDA Flag\n");
    t->device = DEVICE_CPU;
#endif
  }

  return t;
}

Tensor *tensor_random(int ndim, int *shape, Device device) {
  Tensor *t = tensor_create(ndim, shape, device);
  CHECK_NULL_T(t);

  for (int i = 0; i < t->capacity; i++) {
    t->data[i] = (float)rand() / (float)RAND_MAX;
  }

  if (t->device == DEVICE_GPU) {
#ifdef USE_CUDA
    CUDA_CHECK(cudaMemcpy(t->d_data, t->data, t->capacity * sizeof(float),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(t->d_grad, t->grad, t->capacity * sizeof(float),
                          cudaMemcpyHostToDevice));
#else
    ERROR_MSG("Error: GPU as Device, but build without CUDA Flag\n");
    t->device = DEVICE_CPU;
#endif
  }

  return t;
}

Tensor *tensor_from_data(int ndim, int *shape, float *data, Device device) {
  if (data == NULL) {
    ERROR_RETURN_NULL("Error: data array is NULL\n");
  }

  Tensor *t = tensor_create(ndim, shape, device);
  CHECK_NULL_T(t);

  memcpy(t->data, data, t->capacity * sizeof(float));

  if (t->device == DEVICE_GPU) {
#ifdef USE_CUDA
    CUDA_CHECK(cudaMemcpy(t->d_data, t->data, t->capacity * sizeof(float),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(t->d_grad, t->grad, t->capacity * sizeof(float),
                          cudaMemcpyHostToDevice));
#else
    ERROR_MSG("Error: GPU as Device, but build without CUDA Flag\n");
    t->device = DEVICE_CPU;
#endif
  }

  return t;
}

void realize(Tensor *t) {
  if (t == NULL) {
    ERROR_MSG("Error: Realizing a NULL Tensor\n");
    return;
  }

  if (t->_realized == true) {
    return;
  }
  if (t->_forward != NULL) {
    t->_forward();
  }
  t->_realized = true;
}

void tensor_free(Tensor *t) {
  if (t == NULL) {
    return;
  }
  if (t->data != NULL) {
    free(t->data);
  }
  if (t->shape != NULL) {
    free(t->shape);
  }
  if (t->strides != NULL) {
    free(t->strides);
  }
  if (t->_parents != NULL) {
    free(t->_parents);
  }
  if (t->grad != NULL) {
    free(t->grad);
  }

  if (t->device == DEVICE_GPU) {
#ifdef USE_CUDA
    CUDA_CHECK(cudaFree(t->d_data));
    CUDA_CHECK(cudaFree(t->d_grad));
#else
    ERROR_MSG("Error: Free, device is GPU but not build using CUDA Flag\n");
#endif
  }

  delete t;
}
