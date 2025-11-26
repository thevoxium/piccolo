# Compiler and flags
CXX = g++
CXXFLAGS = -std=c++17 -Wall -Wextra -I. -Isrc
LDFLAGS =
LDLIBS =

# BLAS/CBLAS support
# 1. Check for Mac Homebrew OpenBLAS
OPENBLAS_DIR ?= /opt/homebrew/opt/openblas
ifneq ($(wildcard $(OPENBLAS_DIR)/include),)
  CXXFLAGS += -I$(OPENBLAS_DIR)/include
  LDFLAGS += -L$(OPENBLAS_DIR)/lib
  LDLIBS += -lopenblas
else
  # 2. For Linux/Colab: try system OpenBLAS or CBLAS
  # OpenBLAS is typically installed on Colab and provides CBLAS interface
  # We'll link against it - the linker will find it in system paths
  LDLIBS += -lopenblas
endif

# CUDA support (set via CUDA=1 or use run-cuda target)
ifeq ($(CUDA),1)
  CXXFLAGS += -DUSE_CUDA
  
  # Try to find CUDA installation - check multiple sources
  # 1. Check CUDA_HOME environment variable (common on Colab)
  ifneq ($(CUDA_HOME),)
    CUDA_PATH := $(CUDA_HOME)
  else
    # 2. Check CUDA_PATH environment variable
    ifneq ($(CUDA_PATH),)
      # CUDA_PATH is already set
    else
      # 3. Try to find via nvcc
      NVCC := $(shell which nvcc 2>/dev/null)
      ifneq ($(NVCC),)
        # Extract CUDA path from nvcc location (nvcc is typically in /path/to/cuda/bin/nvcc)
        NVCC_DIR := $(shell dirname $(NVCC))
        CUDA_PATH := $(shell dirname $(NVCC_DIR))
      else
        # 4. Check common CUDA installation paths
        ifneq ($(wildcard /usr/local/cuda),)
          CUDA_PATH := /usr/local/cuda
        else
          ifneq ($(wildcard /usr/cuda),)
            CUDA_PATH := /usr/cuda
          else
            # Default fallback (will try system paths if this doesn't exist)
            CUDA_PATH := /usr/local/cuda
          endif
        endif
      endif
    endif
  endif
  
  # Set NVCC compiler if not already set
  ifeq ($(NVCC),)
    NVCC := $(CUDA_PATH)/bin/nvcc
    ifeq ($(wildcard $(NVCC)),)
      NVCC := nvcc
    endif
  endif
  
  # NVCC flags
  # Add GPU architecture support for common Colab GPUs:
  # - sm_70: V100
  # - sm_75: T4
  # - sm_80: A100
  # Using -gencode for multiple architectures ensures compatibility
  NVCCFLAGS = -std=c++17 -I. -Isrc -DUSE_CUDA
  NVCCFLAGS += -I$(CUDA_PATH)/include -I/usr/local/cuda/include
  NVCCFLAGS += -gencode arch=compute_70,code=sm_70
  NVCCFLAGS += -gencode arch=compute_75,code=sm_75
  NVCCFLAGS += -gencode arch=compute_80,code=sm_80
  
  # Add CUDA include directories
  # Always add the detected CUDA path
  CXXFLAGS += -I$(CUDA_PATH)/include
  # Also add standard location (for Colab - /usr/local/cuda is standard)
  CXXFLAGS += -I/usr/local/cuda/include
  
  # Add CUDA library directories (check both lib64 and lib)
  # Always add the detected path
  LDFLAGS += -L$(CUDA_PATH)/lib64 -L$(CUDA_PATH)/lib
  # Also add standard location (for Colab)
  LDFLAGS += -L/usr/local/cuda/lib64 -L/usr/local/cuda/lib
  
  # Always link CUDA runtime library
  LDLIBS += -lcudart
endif

# Source directories
SRC_DIR = src
SRC_FILES = $(wildcard $(SRC_DIR)/*.cpp)
CUDA_DIR = src/cuda
CUDA_FILES = $(wildcard $(CUDA_DIR)/*.cu)
MAIN_FILE = main.cpp

# Object files
OBJ_DIR = build
OBJ_FILES = $(SRC_FILES:$(SRC_DIR)/%.cpp=$(OBJ_DIR)/%.o)
CUDA_OBJ_FILES = $(CUDA_FILES:$(CUDA_DIR)/%.cu=$(OBJ_DIR)/cuda_%.o)
MAIN_OBJ = $(OBJ_DIR)/main.o
TARGET = $(OBJ_DIR)/piccolo

# All object files (conditionally include CUDA objects)
ifeq ($(CUDA),1)
  ALL_OBJ = $(MAIN_OBJ) $(OBJ_FILES) $(CUDA_OBJ_FILES)
else
  ALL_OBJ = $(MAIN_OBJ) $(OBJ_FILES)
endif

# Default target
all: $(TARGET)

# Create build directory if it doesn't exist
$(OBJ_DIR):
	mkdir -p $(OBJ_DIR)

# Compile source files from src/
$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cpp | $(OBJ_DIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Compile CUDA source files from src/cuda/
$(OBJ_DIR)/cuda_%.o: $(CUDA_DIR)/%.cu | $(OBJ_DIR)
ifeq ($(CUDA),1)
	$(NVCC) $(NVCCFLAGS) -c $< -o $@
else
	@echo "Error: CUDA file $< requires CUDA=1"
	@exit 1
endif

# Compile main.cpp
$(MAIN_OBJ): $(MAIN_FILE) | $(OBJ_DIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Link all object files into executable
$(TARGET): $(ALL_OBJ)
	@echo "Linking with LDFLAGS=$(LDFLAGS) LDLIBS=$(LDLIBS)"
	$(CXX) $(ALL_OBJ) $(LDFLAGS) -o $(TARGET) $(LDLIBS)

# Run target - cleans and rebuilds, then runs (defaults to CPU)
run: clean $(TARGET)
	./$(TARGET)
	rm -rf $(OBJ_DIR)

# Run with CPU (no CUDA flags)
run-cpu:
	$(MAKE) clean CUDA=0
	$(MAKE) $(TARGET) CUDA=0
	./$(TARGET)
	rm -rf $(OBJ_DIR)

# Run with CUDA (USE_CUDA flag enabled)
run-cuda:
	$(MAKE) clean CUDA=1
	$(MAKE) $(TARGET) CUDA=1
	./$(TARGET)
	rm -rf $(OBJ_DIR)

# Debug target to show CUDA configuration
debug-cuda:
	@echo "CUDA variable: $(CUDA)"
	@echo "CXXFLAGS: $(CXXFLAGS)"
	@echo "LDFLAGS: $(LDFLAGS)"
	@echo "LDLIBS: $(LDLIBS)"

clean:
	rm -rf $(OBJ_DIR) 

# Phony targets
.PHONY: all clean run run-cpu run-cuda debug-cuda 

