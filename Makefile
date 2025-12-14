# Piccolo - CPU Tensor Library
# Simple Makefile for building the project

CXX = g++
CXXFLAGS = -std=c++17 -Wall -Wextra -O2 -I.

# BLAS support detection
# 1. Check for Mac Accelerate framework
# 2. Check for OpenBLAS (Homebrew on Mac or system on Linux)
OPENBLAS_DIR ?= /opt/homebrew/opt/openblas
ifneq ($(wildcard $(OPENBLAS_DIR)/include),)
  CXXFLAGS += -I$(OPENBLAS_DIR)/include
  LDFLAGS += -L$(OPENBLAS_DIR)/lib
  LDLIBS += -lopenblas
else
  # Fall back to system OpenBLAS (common on Linux/Colab)
  LDLIBS += -lopenblas
endif

# Source files
SRC_DIR = src
SRC_FILES = $(wildcard $(SRC_DIR)/*.cpp)
MAIN_FILE = main.cpp

# Build directory
BUILD_DIR = build

# Object files
OBJ_FILES = $(SRC_FILES:$(SRC_DIR)/%.cpp=$(BUILD_DIR)/%.o)
MAIN_OBJ = $(BUILD_DIR)/main.o
ALL_OBJ = $(MAIN_OBJ) $(OBJ_FILES)

# Target executable
TARGET = $(BUILD_DIR)/piccolo

# Default target
all: $(TARGET)

# Create build directory
$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

# Compile source files
$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cpp | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Compile main.cpp
$(MAIN_OBJ): $(MAIN_FILE) | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Link executable
$(TARGET): $(ALL_OBJ)
	$(CXX) $(ALL_OBJ) $(LDFLAGS) -o $(TARGET) $(LDLIBS)

# Run the program
run: $(TARGET)
	./$(TARGET)

# Clean build artifacts
clean:
	rm -rf $(BUILD_DIR)

# Rebuild from scratch
rebuild: clean all

.PHONY: all clean run rebuild
