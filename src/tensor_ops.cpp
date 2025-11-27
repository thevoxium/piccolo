#include "tensor_ops.hpp"
#include "tensor.hpp"
#include <math.h>
#include <stdio.h>
#ifdef USE_CUDA
#include "cuda/cuda_ops.hpp"
#endif
#if __has_include(<cblas.h>)
extern "C" {
#include <cblas.h>
}
#elif __has_include(<Accelerate/Accelerate.h>)
#include <Accelerate/Accelerate.h>
#else
#error "cblas interface not available"
#endif

static Tensor *tensor_unary_result(Tensor *a, const char *op_name) {
  if (a == NULL) {
    fprintf(stderr, "Error (%s): Tensor is NULL\n", op_name);
    return NULL;
  }
  if (a->data == NULL || a->grad == NULL) {
    fprintf(stderr, "Error (%s): Tensor data or grad arrays are NULL\n",
            op_name);
    return NULL;
  }
  Tensor *result = tensor_create(a->ndim, a->shape, a->device);
  if (result == NULL) {
    fprintf(stderr, "Error (%s): Failed to create result tensor\n", op_name);
    return NULL;
  }
  result->_parents[0] = (Tensor *)a;
  result->_parents[1] = NULL;
  return result;
}

Tensor *tensor_add(Tensor *a, Tensor *b) {
  TENSOR_OPS_COMPATIBLE_CHECK(a, b);

  Tensor *result = tensor_create(a->ndim, a->shape, a->device);
  CHECK_NULL_T(result);

  result->_parents[0] = (Tensor *)a;
  result->_parents[1] = (Tensor *)b;

  if (a->device == DEVICE_GPU) {
#ifdef USE_CUDA
    result->_forward = [=]() {
      cu_tensor_add((const float *)a->d_data, (const float *)b->d_data,
                    (float *)result->d_data, a->capacity);
    };
    result->_backward = [=]() {
      cu_tensor_add_backward((float *)a->d_grad, (float *)b->d_grad,
                             (const float *)result->d_grad, a->capacity);
    };
#else
    ERROR_RETURN_NULL(
        "Error: Device is GPU but Not compiled using CUDA Flag\n");
#endif
  } else {
    result->_forward = [=]() {
      for (int i = 0; i < a->capacity; i++) {
        result->data[i] = a->data[i] + b->data[i];
      }
    };
    result->_backward = [=]() {
      for (int i = 0; i < a->capacity; i++) {
        a->grad[i] += result->grad[i];
        b->grad[i] += result->grad[i];
      }
    };
  }
  return result;
}

Tensor *tensor_sub(Tensor *a, Tensor *b) {
  TENSOR_OPS_COMPATIBLE_CHECK(a, b);

  Tensor *result = tensor_create(a->ndim, a->shape, a->device);
  CHECK_NULL_T(result);

  result->_parents[0] = (Tensor *)a;
  result->_parents[1] = (Tensor *)b;

  if (a->device == DEVICE_GPU) {
#ifdef USE_CUDA
    result->_forward = [=]() {
      cu_tensor_sub((const float *)a->d_data, (const float *)b->d_data,
                    (float *)result->d_data, a->capacity);
    };
    result->_backward = [=]() {
      cu_tensor_sub_backward((float *)a->d_grad, (float *)b->d_grad,
                             (const float *)result->d_grad, a->capacity);
    };
#else
    ERROR_RETURN_NULL(
        "Error: Device is GPU but Not compiled using CUDA Flag\n");
#endif
  } else {
    result->_forward = [=]() {
      for (int i = 0; i < a->capacity; i++) {
        result->data[i] = a->data[i] - b->data[i];
      }
    };
    result->_backward = [=]() {
      for (int i = 0; i < a->capacity; i++) {
        a->grad[i] += result->grad[i];
        b->grad[i] -= result->grad[i];
      }
    };
  }
  return result;
}

Tensor *tensor_scale(Tensor *a, float k) {
  if (a == NULL) {
    fprintf(stderr, "Error: Tensor is NULL\n");
    return NULL;
  }

  if (a->data == NULL || a->grad == NULL) {
    fprintf(stderr, "Error: Tensor data or grad arrays are NULL\n");
    return NULL;
  }
  Tensor *result = tensor_create(a->ndim, a->shape, a->device);
  if (result == NULL) {
    return NULL;
  }

  result->_parents[0] = (Tensor *)a;
  result->_parents[1] = NULL;

  if (a->device == DEVICE_GPU) {
#ifdef USE_CUDA
    result->_forward = [=]() {
      cu_tensor_scale((const float *)a->d_data, (float *)result->d_data, k,
                      a->capacity);
    };
    result->_backward = [=]() {
      cu_tensor_scale_backward((float *)a->d_grad,
                               (const float *)result->d_grad, k, a->capacity);
    };
#else
    ERROR_RETURN_NULL(
        "Error: Device is GPU but Not compiled using CUDA Flag\n");
#endif
  } else {
    result->_forward = [=]() {
      for (int i = 0; i < a->capacity; i++) {
        result->data[i] = k * a->data[i];
      }
    };
    result->_backward = [=]() {
      for (int i = 0; i < a->capacity; i++) {
        a->grad[i] += k * result->grad[i];
      }
    };
  }
  return result;
}

Tensor *tensor_dot(Tensor *a, Tensor *b) {
  if (a == NULL || b == NULL) {
    fprintf(stderr, "Error: Tensor is NULL\n");
    return NULL;
  }

  // Check if both tensors are 1D
  if (a->ndim != 1 || b->ndim != 1) {
    fprintf(stderr, "Error: tensor_dot only supports 1D vectors\n");
    return NULL;
  }

  if (a->shape[0] != b->shape[0]) {
    fprintf(stderr, "Error: Vector lengths must match for dot product\n");
    return NULL;
  }

  if (a->data == NULL || b->data == NULL || a->grad == NULL ||
      b->grad == NULL) {
    fprintf(stderr, "Error: Tensor data or grad arrays are NULL\n");
    return NULL;
  }

  if (a->device != b->device) {
    fprintf(stderr, "Error: Tensor devices do not match\n");
    return NULL;
  }

  int result_shape[] = {1};
  Tensor *result = tensor_create(1, result_shape, a->device);
  if (result == NULL) {
    return NULL;
  }

  result->_parents[0] = (Tensor *)a;
  result->_parents[1] = (Tensor *)b;

  if (a->device == DEVICE_GPU) {
#ifdef USE_CUDA
    result->_forward = [=]() {
      cu_tensor_dot((const float *)a->d_data, (const float *)b->d_data,
                    (float *)result->d_data, a->capacity);
    };
    result->_backward = [=]() {
      float grad_val;
      cudaMemcpy(&grad_val, result->d_grad, sizeof(float),
                 cudaMemcpyDeviceToHost);
      cu_tensor_dot_backward((float *)a->d_grad, (float *)b->d_grad,
                             (const float *)a->d_data, (const float *)b->d_data,
                             grad_val, a->capacity);
    };
#else
    ERROR_RETURN_NULL(
        "Error: Device is GPU but Not compiled using CUDA Flag\n");
#endif
  } else {
    result->_forward = [=]() {
      float dot_result = 0.0f;
      for (int i = 0; i < a->capacity; i++) {
        dot_result += a->data[i] * b->data[i];
      }
      result->data[0] = dot_result;
    };
    result->_backward = [=]() {
      float grad_val = result->grad[0];
      for (int i = 0; i < a->capacity; i++) {
        a->grad[i] += b->data[i] * grad_val;
        b->grad[i] += a->data[i] * grad_val;
      }
    };
  }
  return result;
}

Tensor *tensor_mm(Tensor *a, Tensor *b) {
  if (a == NULL || b == NULL) {
    fprintf(stderr, "Error (tensor_mm): Tensor is NULL\n");
    return NULL;
  }
  if (a->ndim != 2 || b->ndim != 2) {
    fprintf(stderr, "Error (tensor_mm): Both tensors must be 2D\n");
    return NULL;
  }
  if (a->shape[1] != b->shape[0]) {
    fprintf(stderr,
            "Error (tensor_mm): Incompatible shapes (%d, %d) x (%d, %d)\n",
            a->shape[0], a->shape[1], b->shape[0], b->shape[1]);
    return NULL;
  }
  if (a->data == NULL || b->data == NULL || a->grad == NULL ||
      b->grad == NULL) {
    fprintf(stderr, "Error (tensor_mm): Tensor data or grad arrays are NULL\n");
    return NULL;
  }
  if (a->device != b->device) {
    fprintf(stderr, "Error (tensor_mm): Tensor devices do not match\n");
    return NULL;
  }

  const int m = a->shape[0];
  const int k = a->shape[1];
  const int n = b->shape[1];

  int result_shape[2] = {m, n};
  Tensor *result = tensor_create(2, result_shape, a->device);
  if (result == NULL) {
    fprintf(stderr, "Error (tensor_mm): Failed to create result tensor\n");
    return NULL;
  }

  result->_parents[0] = (Tensor *)a;
  result->_parents[1] = (Tensor *)b;

  if (a->device == DEVICE_GPU) {
#ifdef USE_CUDA
    result->_forward = [=]() {
      cu_tensor_mm((const float *)a->d_data, (const float *)b->d_data,
                   (float *)result->d_data, m, k, n);
    };
    result->_backward = [=]() {
      cu_tensor_mm_backward((float *)a->d_grad, (float *)b->d_grad,
                            (const float *)a->d_data, (const float *)b->d_data,
                            (const float *)result->d_grad, m, k, n);
    };
#else
    ERROR_RETURN_NULL(
        "Error (tensor_mm): Device is GPU but Not compiled using CUDA Flag\n");
#endif
  } else {
    result->_forward = [=]() {
      cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, n, k, 1.0f,
                  a->data, k, b->data, n, 0.0f, result->data, n);
    };

    result->_backward = [=]() {
      if (a->grad != NULL) {
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, m, k, n, 1.0f,
                    result->grad, n, b->data, n, 1.0f, a->grad, k);
      }
      if (b->grad != NULL) {
        cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans, k, n, m, 1.0f,
                    a->data, k, result->grad, n, 1.0f, b->grad, n);
      }
    };
  }

  return result;
}

Tensor *tensor_neg(Tensor *a) {
  Tensor *result = tensor_unary_result(a, "tensor_neg");
  if (result == NULL) {
    return NULL;
  }

  if (a->device == DEVICE_GPU) {
#ifdef USE_CUDA
    result->_forward = [=]() {
      cu_tensor_neg((const float *)a->d_data, (float *)result->d_data,
                    a->capacity);
    };
    result->_backward = [=]() {
      cu_tensor_neg_backward((float *)a->d_grad, (const float *)result->d_grad,
                             a->capacity);
    };
#else
    fprintf(stderr, "Error: Device is GPU but Not compiled using CUDA Flag\n");
    tensor_free(result);
    return NULL;
#endif
  } else {
    result->_forward = [=]() {
      for (int i = 0; i < a->capacity; i++) {
        result->data[i] = -a->data[i];
      }
    };
    result->_backward = [=]() {
      for (int i = 0; i < a->capacity; i++) {
        a->grad[i] -= result->grad[i];
      }
    };
  }
  return result;
}

Tensor *tensor_log2(Tensor *a) {
  Tensor *result = tensor_unary_result(a, "tensor_log2");
  if (result == NULL) {
    return NULL;
  }

  if (a->device == DEVICE_GPU) {
#ifdef USE_CUDA
    result->_forward = [=]() {
      cu_tensor_log2((const float *)a->d_data, (float *)result->d_data,
                     a->capacity);
    };
    result->_backward = [=]() {
      cu_tensor_log2_backward((float *)a->d_grad, (const float *)a->d_data,
                              (const float *)result->d_grad, a->capacity);
    };
#else
    fprintf(stderr, "Error: Device is GPU but Not compiled using CUDA Flag\n");
    tensor_free(result);
    return NULL;
#endif
  } else {
    result->_forward = [=]() {
      for (int i = 0; i < a->capacity; i++) {
        float val = a->data[i];
        if (val <= 0.0f) {
          fprintf(stderr,
                  "Warning (tensor_log2): log2 undefined for non-positive "
                  "values, got %f\n",
                  val);
          result->data[i] = -INFINITY;
        } else {
          result->data[i] = log2f(val);
        }
      }
    };
    const float ln2 = logf(2.0f);
    result->_backward = [=]() {
      for (int i = 0; i < a->capacity; i++) {
        float val = a->data[i];
        if (val <= 0.0f) {
          continue;
        }
        a->grad[i] += result->grad[i] / (val * ln2);
      }
    };
  }
  return result;
}

Tensor *tensor_exp2(Tensor *a) {
  Tensor *result = tensor_unary_result(a, "tensor_exp2");
  if (result == NULL) {
    return NULL;
  }

  if (a->device == DEVICE_GPU) {
#ifdef USE_CUDA
    result->_forward = [=]() {
      cu_tensor_exp2((const float *)a->d_data, (float *)result->d_data,
                     a->capacity);
    };
    result->_backward = [=]() {
      cu_tensor_exp2_backward((float *)a->d_grad,
                              (const float *)result->d_data,
                              (const float *)result->d_grad, a->capacity);
    };
#else
    fprintf(stderr, "Error: Device is GPU but Not compiled using CUDA Flag\n");
    tensor_free(result);
    return NULL;
#endif
  } else {
    result->_forward = [=]() {
      for (int i = 0; i < a->capacity; i++) {
        result->data[i] = exp2f(a->data[i]);
      }
    };
    const float ln2 = logf(2.0f);
    result->_backward = [=]() {
      for (int i = 0; i < a->capacity; i++) {
        a->grad[i] += result->grad[i] * result->data[i] * ln2;
      }
    };
  }
  return result;
}

Tensor *tensor_sqrt(Tensor *a) {
  Tensor *result = tensor_unary_result(a, "tensor_sqrt");
  if (result == NULL) {
    return NULL;
  }

  if (a->device == DEVICE_GPU) {
#ifdef USE_CUDA
    result->_forward = [=]() {
      cu_tensor_sqrt((const float *)a->d_data, (float *)result->d_data,
                     a->capacity);
    };
    result->_backward = [=]() {
      cu_tensor_sqrt_backward((float *)a->d_grad, (const float *)a->d_data,
                              (const float *)result->d_data,
                              (const float *)result->d_grad, a->capacity);
    };
#else
    fprintf(stderr, "Error: Device is GPU but Not compiled using CUDA Flag\n");
    tensor_free(result);
    return NULL;
#endif
  } else {
    result->_forward = [=]() {
      for (int i = 0; i < a->capacity; i++) {
        float val = a->data[i];
        if (val < 0.0f) {
          fprintf(stderr,
                  "Warning (tensor_sqrt): sqrt undefined for negative values, "
                  "got %f\n",
                  val);
          result->data[i] = NAN;
        } else {
          result->data[i] = sqrtf(val);
        }
      }
    };
    result->_backward = [=]() {
      for (int i = 0; i < a->capacity; i++) {
        float val = a->data[i];
        if (val <= 0.0f || result->data[i] == 0.0f) {
          continue;
        }
        float local_grad = 0.5f / result->data[i];
        a->grad[i] += result->grad[i] * local_grad;
      }
    };
  }
  return result;
}

Tensor *tensor_sin(Tensor *a) {
  Tensor *result = tensor_unary_result(a, "tensor_sin");
  if (result == NULL) {
    return NULL;
  }

  if (a->device == DEVICE_GPU) {
#ifdef USE_CUDA
    result->_forward = [=]() {
      cu_tensor_sin((const float *)a->d_data, (float *)result->d_data,
                    a->capacity);
    };
    result->_backward = [=]() {
      cu_tensor_sin_backward((float *)a->d_grad, (const float *)a->d_data,
                             (const float *)result->d_grad, a->capacity);
    };
#else
    fprintf(stderr, "Error: Device is GPU but Not compiled using CUDA Flag\n");
    tensor_free(result);
    return NULL;
#endif
  } else {
    result->_forward = [=]() {
      for (int i = 0; i < a->capacity; i++) {
        result->data[i] = sinf(a->data[i]);
      }
    };
    result->_backward = [=]() {
      for (int i = 0; i < a->capacity; i++) {
        a->grad[i] += result->grad[i] * cosf(a->data[i]);
      }
    };
  }
  return result;
}

Tensor *tensor_cos(Tensor *a) {
  Tensor *result = tensor_unary_result(a, "tensor_cos");
  if (result == NULL) {
    return NULL;
  }

  if (a->device == DEVICE_GPU) {
#ifdef USE_CUDA
    result->_forward = [=]() {
      cu_tensor_cos((const float *)a->d_data, (float *)result->d_data,
                    a->capacity);
    };
    result->_backward = [=]() {
      cu_tensor_cos_backward((float *)a->d_grad, (const float *)a->d_data,
                             (const float *)result->d_grad, a->capacity);
    };
#else
    fprintf(stderr, "Error: Device is GPU but Not compiled using CUDA Flag\n");
    tensor_free(result);
    return NULL;
#endif
  } else {
    result->_forward = [=]() {
      for (int i = 0; i < a->capacity; i++) {
        result->data[i] = cosf(a->data[i]);
      }
    };
    result->_backward = [=]() {
      for (int i = 0; i < a->capacity; i++) {
        a->grad[i] -= result->grad[i] * sinf(a->data[i]);
      }
    };
  }
  return result;
}

Tensor *tensor_tan(Tensor *a) {
  Tensor *result = tensor_unary_result(a, "tensor_tan");
  if (result == NULL) {
    return NULL;
  }

  if (a->device == DEVICE_GPU) {
#ifdef USE_CUDA
    result->_forward = [=]() {
      cu_tensor_tan((const float *)a->d_data, (float *)result->d_data,
                    a->capacity);
    };
    result->_backward = [=]() {
      cu_tensor_tan_backward((float *)a->d_grad,
                             (const float *)result->d_data,
                             (const float *)result->d_grad, a->capacity);
    };
#else
    fprintf(stderr, "Error: Device is GPU but Not compiled using CUDA Flag\n");
    tensor_free(result);
    return NULL;
#endif
  } else {
    result->_forward = [=]() {
      for (int i = 0; i < a->capacity; i++) {
        result->data[i] = tanf(a->data[i]);
      }
    };
    result->_backward = [=]() {
      for (int i = 0; i < a->capacity; i++) {
        float tangent = result->data[i];
        a->grad[i] += result->grad[i] * (1.0f + tangent * tangent);
      }
    };
  }
  return result;
}

Tensor *tensor_trunc(Tensor *a) {
  Tensor *result = tensor_unary_result(a, "tensor_trunc");
  if (result == NULL) {
    return NULL;
  }

  if (a->device == DEVICE_GPU) {
#ifdef USE_CUDA
    result->_forward = [=]() {
      cu_tensor_trunc((const float *)a->d_data, (float *)result->d_data,
                      a->capacity);
    };
    result->_backward = []() {};
#else
    fprintf(stderr, "Error: Device is GPU but Not compiled using CUDA Flag\n");
    tensor_free(result);
    return NULL;
#endif
  } else {
    result->_forward = [=]() {
      for (int i = 0; i < a->capacity; i++) {
        result->data[i] = truncf(a->data[i]);
      }
    };
    result->_backward = []() {};
  }
  return result;
}

Tensor *tensor_ceil(Tensor *a) {
  Tensor *result = tensor_unary_result(a, "tensor_ceil");
  if (result == NULL) {
    return NULL;
  }

  if (a->device == DEVICE_GPU) {
#ifdef USE_CUDA
    result->_forward = [=]() {
      cu_tensor_ceil((const float *)a->d_data, (float *)result->d_data,
                     a->capacity);
    };
    result->_backward = []() {};
#else
    fprintf(stderr, "Error: Device is GPU but Not compiled using CUDA Flag\n");
    tensor_free(result);
    return NULL;
#endif
  } else {
    result->_forward = [=]() {
      for (int i = 0; i < a->capacity; i++) {
        result->data[i] = ceilf(a->data[i]);
      }
    };
    result->_backward = []() {};
  }
  return result;
}

Tensor *tensor_floor(Tensor *a) {
  Tensor *result = tensor_unary_result(a, "tensor_floor");
  if (result == NULL) {
    return NULL;
  }

  if (a->device == DEVICE_GPU) {
#ifdef USE_CUDA
    result->_forward = [=]() {
      cu_tensor_floor((const float *)a->d_data, (float *)result->d_data,
                      a->capacity);
    };
    result->_backward = []() {};
#else
    fprintf(stderr, "Error: Device is GPU but Not compiled using CUDA Flag\n");
    tensor_free(result);
    return NULL;
#endif
  } else {
    result->_forward = [=]() {
      for (int i = 0; i < a->capacity; i++) {
        result->data[i] = floorf(a->data[i]);
      }
    };
    result->_backward = []() {};
  }
  return result;
}

Tensor *tensor_round(Tensor *a) {
  Tensor *result = tensor_unary_result(a, "tensor_round");
  if (result == NULL) {
    return NULL;
  }

  if (a->device == DEVICE_GPU) {
#ifdef USE_CUDA
    result->_forward = [=]() {
      cu_tensor_round((const float *)a->d_data, (float *)result->d_data,
                      a->capacity);
    };
    result->_backward = []() {};
#else
    fprintf(stderr, "Error: Device is GPU but Not compiled using CUDA Flag\n");
    tensor_free(result);
    return NULL;
#endif
  } else {
    result->_forward = [=]() {
      for (int i = 0; i < a->capacity; i++) {
        result->data[i] = roundf(a->data[i]);
      }
    };
    result->_backward = []() {};
  }
  return result;
}

Tensor *tensor_square(Tensor *a) {
  Tensor *result = tensor_unary_result(a, "tensor_square");
  if (result == NULL) {
    return NULL;
  }

  if (a->device == DEVICE_GPU) {
#ifdef USE_CUDA
    result->_forward = [=]() {
      cu_tensor_square((const float *)a->d_data, (float *)result->d_data,
                       a->capacity);
    };
    result->_backward = [=]() {
      cu_tensor_square_backward((float *)a->d_grad, (const float *)a->d_data,
                                (const float *)result->d_grad, a->capacity);
    };
#else
    fprintf(stderr, "Error: Device is GPU but Not compiled using CUDA Flag\n");
    tensor_free(result);
    return NULL;
#endif
  } else {
    result->_forward = [=]() {
      for (int i = 0; i < a->capacity; i++) {
        float val = a->data[i];
        result->data[i] = val * val;
      }
    };
    result->_backward = [=]() {
      for (int i = 0; i < a->capacity; i++) {
        a->grad[i] += result->grad[i] * 2.0f * a->data[i];
      }
    };
  }
  return result;
}

Tensor *tensor_sign(Tensor *a) {
  Tensor *result = tensor_unary_result(a, "tensor_sign");
  if (result == NULL) {
    return NULL;
  }

  if (a->device == DEVICE_GPU) {
#ifdef USE_CUDA
    result->_forward = [=]() {
      cu_tensor_sign((const float *)a->d_data, (float *)result->d_data,
                     a->capacity);
    };
    result->_backward = []() {};
#else
    fprintf(stderr, "Error: Device is GPU but Not compiled using CUDA Flag\n");
    tensor_free(result);
    return NULL;
#endif
  } else {
    result->_forward = [=]() {
      for (int i = 0; i < a->capacity; i++) {
        float val = a->data[i];
        result->data[i] = (val > 0.0f) ? 1.0f : ((val < 0.0f) ? -1.0f : 0.0f);
      }
    };
    result->_backward = []() {};
  }
  return result;
}

Tensor *tensor_abs(Tensor *a) {
  Tensor *result = tensor_unary_result(a, "tensor_abs");
  if (result == NULL) {
    return NULL;
  }

  if (a->device == DEVICE_GPU) {
#ifdef USE_CUDA
    result->_forward = [=]() {
      cu_tensor_abs((const float *)a->d_data, (float *)result->d_data,
                    a->capacity);
    };
    result->_backward = [=]() {
      cu_tensor_abs_backward((float *)a->d_grad, (const float *)a->d_data,
                             (const float *)result->d_grad, a->capacity);
    };
#else
    fprintf(stderr, "Error: Device is GPU but Not compiled using CUDA Flag\n");
    tensor_free(result);
    return NULL;
#endif
  } else {
    result->_forward = [=]() {
      for (int i = 0; i < a->capacity; i++) {
        result->data[i] = fabsf(a->data[i]);
      }
    };
    result->_backward = [=]() {
      for (int i = 0; i < a->capacity; i++) {
        float val = a->data[i];
        float sign = 0.0f;
        if (val > 0.0f) {
          sign = 1.0f;
        } else if (val < 0.0f) {
          sign = -1.0f;
        }
        a->grad[i] += result->grad[i] * sign;
      }
    };
  }
  return result;
}

Tensor *tensor_reciprocal(Tensor *a) {
  Tensor *result = tensor_unary_result(a, "tensor_reciprocal");
  if (result == NULL) {
    return NULL;
  }

  if (a->device == DEVICE_GPU) {
#ifdef USE_CUDA
    result->_forward = [=]() {
      cu_tensor_reciprocal((const float *)a->d_data, (float *)result->d_data,
                           a->capacity);
    };
    result->_backward = [=]() {
      cu_tensor_reciprocal_backward((float *)a->d_grad,
                                    (const float *)a->d_data,
                                    (const float *)result->d_grad, a->capacity);
    };
#else
    fprintf(stderr, "Error: Device is GPU but Not compiled using CUDA Flag\n");
    tensor_free(result);
    return NULL;
#endif
  } else {
    result->_forward = [=]() {
      for (int i = 0; i < a->capacity; i++) {
        float val = a->data[i];
        if (val == 0.0f) {
          fprintf(stderr, "Warning (tensor_reciprocal): division by zero\n");
          result->data[i] = INFINITY;
        } else {
          result->data[i] = 1.0f / val;
        }
      }
    };
    result->_backward = [=]() {
      for (int i = 0; i < a->capacity; i++) {
        float val = a->data[i];
        if (val == 0.0f) {
          continue;
        }
        float local_grad = -1.0f / (val * val);
        a->grad[i] += result->grad[i] * local_grad;
      }
    };
  }
  return result;
}

Tensor *tensor_pow(Tensor *a, float exponent) {
  Tensor *result = tensor_unary_result(a, "tensor_pow");
  if (result == NULL) {
    return NULL;
  }

  if (a->device == DEVICE_GPU) {
#ifdef USE_CUDA
    result->_forward = [=]() {
      cu_tensor_pow((const float *)a->d_data, (float *)result->d_data, exponent,
                    a->capacity);
    };
    result->_backward = [=]() {
      cu_tensor_pow_backward((float *)a->d_grad, (const float *)a->d_data,
                             (const float *)result->d_grad, exponent,
                             a->capacity);
    };
#else
    fprintf(stderr, "Error: Device is GPU but Not compiled using CUDA Flag\n");
    tensor_free(result);
    return NULL;
#endif
  } else {
    result->_forward = [=]() {
      for (int i = 0; i < a->capacity; i++) {
        result->data[i] = powf(a->data[i], exponent);
      }
    };
    result->_backward = [=]() {
      for (int i = 0; i < a->capacity; i++) {
        float base = a->data[i];
        float local_grad = 0.0f;
        if (base == 0.0f) {
          if (exponent > 1.0f) {
            local_grad = 0.0f;
          } else if (exponent == 1.0f) {
            local_grad = 1.0f;
          } else {
            continue;
          }
        } else {
          local_grad = exponent * powf(base, exponent - 1.0f);
        }
        a->grad[i] += result->grad[i] * local_grad;
      }
    };
  }
  return result;
}

Tensor *tensor_exp(Tensor *a) {
  Tensor *result = tensor_unary_result(a, "tensor_exp");
  if (result == NULL) {
    return NULL;
  }

  if (a->device == DEVICE_GPU) {
#ifdef USE_CUDA
    result->_forward = [=]() {
      cu_tensor_exp((const float *)a->d_data, (float *)result->d_data,
                    a->capacity);
    };
    result->_backward = [=]() {
      cu_tensor_exp_backward((float *)a->d_grad,
                             (const float *)result->d_data,
                             (const float *)result->d_grad, a->capacity);
    };
#else
    fprintf(stderr, "Error: Device is GPU but Not compiled using CUDA Flag\n");
    tensor_free(result);
    return NULL;
#endif
  } else {
    result->_forward = [=]() {
      for (int i = 0; i < a->capacity; i++) {
        result->data[i] = expf(a->data[i]);
      }
    };
    result->_backward = [=]() {
      for (int i = 0; i < a->capacity; i++) {
        a->grad[i] += result->grad[i] * result->data[i];
      }
    };
  }
  return result;
}

Tensor *tensor_log(Tensor *a) {
  Tensor *result = tensor_unary_result(a, "tensor_log");
  if (result == NULL) {
    return NULL;
  }

  if (a->device == DEVICE_GPU) {
#ifdef USE_CUDA
    result->_forward = [=]() {
      cu_tensor_log((const float *)a->d_data, (float *)result->d_data,
                    a->capacity);
    };
    result->_backward = [=]() {
      cu_tensor_log_backward((float *)a->d_grad, (const float *)a->d_data,
                             (const float *)result->d_grad, a->capacity);
    };
#else
    fprintf(stderr, "Error: Device is GPU but Not compiled using CUDA Flag\n");
    tensor_free(result);
    return NULL;
#endif
  } else {
    result->_forward = [=]() {
      for (int i = 0; i < a->capacity; i++) {
        float val = a->data[i];
        if (val <= 0.0f) {
          fprintf(stderr,
                  "Warning (tensor_log): log undefined for non-positive "
                  "values, got %f\n",
                  val);
          result->data[i] = -INFINITY;
        } else {
          result->data[i] = logf(val);
        }
      }
    };
    result->_backward = [=]() {
      for (int i = 0; i < a->capacity; i++) {
        float val = a->data[i];
        if (val <= 0.0f) {
          continue;
        }
        a->grad[i] += result->grad[i] / val;
      }
    };
  }
  return result;
}

Tensor *tensor_aggregate(Tensor *a) {
  if (a == NULL) {
    fprintf(stderr, "Error: Tensor is NULL\n");
    return NULL;
  }

  if (a->data == NULL || a->grad == NULL) {
    fprintf(stderr, "Error: Tensor data or grad arrays are NULL\n");
    return NULL;
  }

  int result_shape[] = {1};
  Tensor *result = tensor_create(1, result_shape, a->device);
  if (result == NULL) {
    return NULL;
  }

  result->_parents[0] = (Tensor *)a;
  result->_parents[1] = NULL;

  if (a->device == DEVICE_GPU) {
#ifdef USE_CUDA
    result->_forward = [=]() {
      cu_tensor_aggregate((const float *)a->d_data, (float *)result->d_data,
                          a->capacity);
    };
    result->_backward = [=]() {
      float grad_val;
      cudaMemcpy(&grad_val, result->d_grad, sizeof(float),
                 cudaMemcpyDeviceToHost);
      cu_tensor_aggregate_backward((float *)a->d_grad, grad_val, a->capacity);
    };
#else
    fprintf(stderr, "Error: Device is GPU but Not compiled using CUDA Flag\n");
    tensor_free(result);
    return NULL;
#endif
  } else {
    result->_forward = [=]() {
      float sum = 0.0f;
      for (int i = 0; i < a->capacity; i++) {
        sum += a->data[i];
      }
      result->data[0] = sum;
    };
    result->_backward = [=]() {
      float grad_val = result->grad[0];
      for (int i = 0; i < a->capacity; i++) {
        a->grad[i] += grad_val;
      }
    };
  }
  return result;
}
