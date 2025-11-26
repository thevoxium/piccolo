#include "utils.hpp"
#include <string>

// Helper function to get CPU data pointer (handles GPU->CPU copy if needed)
static float *get_cpu_data(const Tensor &t, float **temp_buffer) {
  if (t.device == DEVICE_GPU) {
#ifdef USE_CUDA
    if (t.d_data == NULL)
      return NULL;
    *temp_buffer = (float *)malloc(t.capacity * sizeof(float));
    if (*temp_buffer == NULL)
      return NULL;
    CUDA_CHECK(cudaMemcpy(*temp_buffer, t.d_data, t.capacity * sizeof(float),
                          cudaMemcpyDeviceToHost));
    return *temp_buffer;
#else
    return NULL;
#endif
  }
  return t.data;
}

// Helper function to recursively print tensor data
static void print_tensor_data(std::ostream &os, const float *data,
                               const Tensor &t, int dim, int offset) {
  if (dim >= t.ndim)
    return;

  os << "[";
  int size = t.shape[dim];

  if (dim == t.ndim - 1) {
    // Last dimension: print values directly
    for (int i = 0; i < size; ++i) {
      if (i > 0)
        os << ", ";
      if (offset + i < t.capacity)
        os << data[offset + i];
    }
  } else {
    // Recursive case: print nested arrays
    for (int i = 0; i < size; ++i) {
      if (i > 0)
        os << ",\n" << std::string(dim + 1, ' ');
      int new_offset = offset + i * t.strides[dim];
      print_tensor_data(os, data, t, dim + 1, new_offset);
    }
  }
  os << "]";
}

std::ostream &operator<<(std::ostream &os, const Tensor &t) {
  // Validate tensor
  if (t.shape == NULL || t.strides == NULL) {
    os << "INVALID_TENSOR";
    return os;
  }

  // Get CPU data pointer (handles GPU copy if needed)
  float *temp_buffer = nullptr;
  const float *data = get_cpu_data(t, &temp_buffer);

  if (data == NULL) {
    os << "INVALID_TENSOR";
    if (temp_buffer != nullptr)
      free(temp_buffer);
    return os;
  }

  // Print tensor
  os << "Tensor(";
  print_tensor_data(os, data, t, 0, 0);
  os << ", shape: (";
  for (int i = 0; i < t.ndim; ++i) {
    if (i > 0)
      os << ", ";
    os << t.shape[i];
  }
  os << "), device: " << t.device << ")";

  // Cleanup
  if (temp_buffer != nullptr)
    free(temp_buffer);

  return os;
}

