# Compiler and flags
CXX = g++
CXXFLAGS = -std=c++17 -Wall -Wextra -I.
OPENBLAS_DIR ?= /opt/homebrew/opt/openblas
CXXFLAGS += -I$(OPENBLAS_DIR)/include
LDFLAGS += -L$(OPENBLAS_DIR)/lib
LDLIBS += -lopenblas
TARGET = piccolo

# Source directories
SRC_DIR = src
SRC_FILES = $(wildcard $(SRC_DIR)/*.cpp)
MAIN_FILE = main.cpp

# Object files
OBJ_DIR = build
OBJ_FILES = $(SRC_FILES:$(SRC_DIR)/%.cpp=$(OBJ_DIR)/%.o)
MAIN_OBJ = $(OBJ_DIR)/main.o

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

# Run target - cleans and rebuilds, then runs
run: clean $(TARGET)
	./$(TARGET)

# Test files
TEST_DIR = tests
TEST_TARGET = test_runner
TEST_FILE = $(TEST_DIR)/test_all.cpp
TEST_OBJ = $(OBJ_DIR)/test_all.o

# Test executable
$(TEST_TARGET): $(TEST_OBJ) $(OBJ_FILES)
	$(CXX) $(TEST_OBJ) $(OBJ_FILES) $(LDFLAGS) -o $(TEST_TARGET) $(LDLIBS)

# Compile test file
$(TEST_OBJ): $(TEST_FILE) | $(OBJ_DIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Run tests
test: $(TEST_TARGET)
	./$(TEST_TARGET)

# Clean target - removes binaries and build directory
clean:
	rm -rf $(OBJ_DIR) $(TARGET) $(TEST_TARGET)

# Phony targets
.PHONY: all clean run test

