---
title: Home
order: 0
---

# Tensory

**Bare-metal Autograd Engine for C++17**

Tensory is a lightweight, header-heavy deep learning framework built from scratch. It prioritizes manual control over memory and execution, making it ideal for learning how deep learning frameworks work under the hood or for embedded systems where overhead must be minimized.

## Features

-   **Lazy Evaluation**: Define your computation graph first, execute only when needed.
-   **Manual Memory Management**: No hidden allocations. You control the lifetime of every tensor.
-   **Explicit Device Control**: Manually sync data between CPU and GPU.
-   **CUDA Acceleration**: Drop-in GPU support for matrix operations.

## Quick Start

```bash
# Clone the repository
git clone https://github.com/yourusername/tensory.git
cd tensory

# Build the example
make

# Run
./build/piccolo
```

