# Compiler and flags
CXX = g++
CXXFLAGS = -std=c++17 -Wall -Wextra -I.
OPENBLAS_DIR ?= /opt/homebrew/opt/openblas
CXXFLAGS += -I$(OPENBLAS_DIR)/include
LDFLAGS += -L$(OPENBLAS_DIR)/lib
LDLIBS += -lopenblas

# CUDA support (set via CUDA=1 or use run-cuda target)
ifeq ($(CUDA),1)
  CXXFLAGS += -DUSE_CUDA
endif

# Source directories
SRC_DIR = src
SRC_FILES = $(wildcard $(SRC_DIR)/*.cpp)
MAIN_FILE = main.cpp

# Object files
OBJ_DIR = build
OBJ_FILES = $(SRC_FILES:$(SRC_DIR)/%.cpp=$(OBJ_DIR)/%.o)
MAIN_OBJ = $(OBJ_DIR)/main.o
TARGET = $(OBJ_DIR)/piccolo

# All object files
ALL_OBJ = $(MAIN_OBJ) $(OBJ_FILES)

# Default target
all: $(TARGET)

# Create build directory if it doesn't exist
$(OBJ_DIR):
	mkdir -p $(OBJ_DIR)

# Compile source files from src/
$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cpp | $(OBJ_DIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Compile main.cpp
$(MAIN_OBJ): $(MAIN_FILE) | $(OBJ_DIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Link all object files into executable
$(TARGET): $(ALL_OBJ)
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

clean:
	rm -rf $(OBJ_DIR) 

# Phony targets
.PHONY: all clean run run-cpu run-cuda 

