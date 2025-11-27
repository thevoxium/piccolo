---
title: Operations
order: 3
---

# Operations

Tensory provides a suite of mathematical operations that automatically build the computation graph. All operations support both CPU and GPU execution, determined by the device of the input tensors.

## Binary Operations

These operations take two tensors as input. Shapes must be broadcastable or match (currently strict matching or scalar broadcasting is supported depending on implementation).

| Function | Description |
| :--- | :--- |
| `tensor_add(a, b)` | Element-wise addition ($a + b$) |
| `tensor_sub(a, b)` | Element-wise subtraction ($a - b$) |
| `tensor_mul(a, b)` | Element-wise multiplication ($a * b$) |
| `tensor_mm(a, b)` | Matrix multiplication ($A \times B$) |
| `tensor_dot(a, b)` | Dot product |

## Unary Operations

These operations apply a function element-wise to a single tensor.

| Function | Description |
| :--- | :--- |
| `tensor_neg(t)` | Negation ($-t$) |
| `tensor_scale(t, k)` | Scale by a scalar float constant ($k * t$) |
| `tensor_pow(t, exp)` | Power ($t^{exp}$) |
| `tensor_exp(t)` | Exponential ($e^t$) |
| `tensor_log(t)` | Natural logarithm ($\ln(t)$) |
| `tensor_log2(t)` | Base-2 logarithm ($\log_2(t)$) |
| `tensor_sqrt(t)` | Square root ($\sqrt{t}$) |
| `tensor_square(t)` | Square ($t^2$) |
| `tensor_sin(t)` | Sine ($\sin(t)$) |
| `tensor_cos(t)` | Cosine ($\cos(t)$) |
| `tensor_tan(t)` | Tangent ($\tan(t)$) |
| `tensor_abs(t)` | Absolute value ($|t|$) |
| `tensor_sign(t)` | Sign function |
| `tensor_reciprocal(t)` | Reciprocal ($1/t$) |
| `tensor_trunc(t)` | Truncate |
| `tensor_floor(t)` | Floor |
| `tensor_ceil(t)` | Ceiling |
| `tensor_round(t)` | Round |

## Reduction Operations

| Function | Description |
| :--- | :--- |
| `tensor_aggregate(t)` | Sum all elements (reduces to scalar) |

## Loss Functions

Loss functions are typically the root of the computation graph in training.

```cpp
// Mean Squared Error Loss
Tensor* loss_mse(Tensor* pred, Tensor* target);
```

## Checks & Validation

All operations perform compatibility checks (e.g., matching devices, compatible shapes) before execution. If a check fails, they print an error to `stderr` and return `NULL`.

