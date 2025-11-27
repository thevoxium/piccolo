---
title: Tensor Management
order: 1
---

# Tensor Management

The `Tensor` struct is the core data structure in Tensory. It manages memory for n-dimensional arrays on both Host (CPU) and Device (GPU).

## The Tensor Struct

The `Tensor` struct contains data pointers, gradient pointers, shape information, and the computation graph state.

```cpp
typedef struct Tensor {
  float *data;       // Host data pointer
  float *grad;       // Host gradient pointer
  int ndim;          // Number of dimensions
  int *shape;        // Array of dimension sizes
  int *strides;      // Array of strides
  int capacity;      // Total number of elements
  Device device;     // DEVICE_CPU or DEVICE_GPU

  // Graph state
  Tensor **_parents; // Parent tensors in the graph
  bool _realized;    // Whether the tensor data is computed
  bool _host_dirty;  // If true, device data is newer than host
  bool _device_dirty;// If true, host data is newer than device
  
  // CUDA specific
  float *d_data;     // Device data pointer
  float *d_grad;     // Device gradient pointer
} Tensor;
```

## Creation & Initialization

### `tensor_create`
Allocates a new tensor with undefined data.

```cpp
Tensor* tensor_create(int ndim, int* shape, Device device);
```

### `tensor_from_data`
Creates a new tensor initialized with data from a host array.

```cpp
Tensor *tensor_from_data(int ndim, int *shape, float *data, Device device);
```

### Factory Functions
Helper functions to create initialized tensors.

```cpp
// Create tensor filled with ones
Tensor *tensor_ones(int ndim, int *shape, Device device);

// Create tensor filled with zeros
Tensor *tensor_zeroes(int ndim, int *shape, Device device);

// Create tensor with random uniform values [0, 1]
Tensor *tensor_random(int ndim, int *shape, Device device);
```

## Memory Management

Since Tensory uses C-style manual memory management, you are responsible for freeing tensors when they are no longer needed.

```cpp
void tensor_free(Tensor *t);
```

> **Note:** `tensor_free` releases the memory for the tensor's data and shape. It does **not** recursively free the computation graph. Use `free_graph` (from engine) for that.

## Device Synchronization

Tensory allows explicit control over data movement between CPU and GPU.

### `sync_to_host`
Copies data from Device to Host if the device data is newer (dirty).

```cpp
void sync_to_host(Tensor *t);
```

### `sync_to_device`
Copies data from Host to Device if the host data is newer (dirty).

```cpp
void sync_to_device(Tensor *t);
```

### Graph Synchronization
These functions recursively synchronize an entire graph rooted at `root`.

```cpp
void sync_graph_to_host(Tensor *root);
void sync_graph_to_device(Tensor *root);
```

## Device Management

You can move an entire computation graph to a different device.

```cpp
// Recursively move the graph rooted at 'root' to the specified device
void set_graph_device(Tensor *root, Device device);
```
