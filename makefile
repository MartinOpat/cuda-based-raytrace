# Compiler and flags
NVCC = nvcc
CXXFLAGS = -I./src -I./linalg -I./img -I./objs -std=c++17

# Directories
SRC_DIR = src
BUILD_DIR = build

# Files
TARGET = $(BUILD_DIR)/main
SRC_FILES = $(wildcard $(SRC_DIR)/*.cu)
OBJ_FILES = $(patsubst $(SRC_DIR)/%.cu,$(BUILD_DIR)/%.o,$(SRC_FILES))

# Default target
all: $(TARGET)

# Build the main target
$(TARGET): $(OBJ_FILES) | $(BUILD_DIR)
	$(NVCC) $(CXXFLAGS) -o $@ $^

# Compile object files
$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cu
	$(NVCC) $(CXXFLAGS) -c $< -o $@

# Debug build
debug: CXXFLAGS += -g
debug: clean all

# Clean build directory
clean:
	rm -rf $(BUILD_DIR)/*

# Create build directory if it doesn't exist
$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

.PHONY: all clean debug
