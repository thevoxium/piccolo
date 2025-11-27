---
title: Engine & Autograd
order: 2
---

# Engine & Autograd

Tensory implements a **define-by-run** (lazy) execution engine. Operations on tensors build a computation graph, which is evaluated only when explicitly requested.

## Execution Model

### `realize`
The `realize` function triggers the forward pass. It topologically sorts the computation graph ending at the target tensor and executes the kernel for each node that hasn't been realized yet.

```cpp
void realize(Tensor *t);
```

### `ensure_realized`
A lightweight check that calls `realize` only if the tensor hasn't been realized.

```cpp
void ensure_realized(Tensor *t);
```

## Automatic Differentiation

Tensory supports reverse-mode automatic differentiation (backpropagation).

### `backward`
Triggers the backward pass to compute gradients.

```cpp
void backward(Tensor *root);
```

**Steps performed by `backward`:**
1.  Calls `realize(root)` to ensure the forward pass is complete.
2.  Sets the gradient of the `root` tensor to 1.0 (assuming it is a scalar loss).
3.  Traverses the graph in reverse topological order.
4.  Executes the backward function for each node to propagate gradients to its parents.

### `zero_grad`
Resets gradients to zero. This is typically done before the backward pass in a training loop to prevent gradient accumulation from previous iterations.

```cpp
// Zero gradient for a single tensor
void zero_grad(Tensor *t);

// Zero gradients for the entire graph
void zero_graph_grad(Tensor *root);
```

## Graph Management

### `free_graph`
Recursively frees the tensor and all its ancestors in the graph. This is useful for cleaning up after an iteration.

```cpp
void free_graph(Tensor *root);
```

