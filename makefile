# Compiler and flags
NVCC = nvcc
CXXFLAGS = -I./src -std=c++17 $(shell ncxx4-config --cflags) $(shell ncxx4-config --libs) -g -G
COMPILE_OBJ_FLAGS = --device-c

# Directories
SRC_DIR = src
BUILD_DIR = build

# Files
TARGET = $(BUILD_DIR)/main
SRC_FILES := $(shell find $(SRC_DIR) -type f \( -name '*.cu' -o -name '*.cpp' \))
OBJ_FILES = $(patsubst $(SRC_DIR)/%.cu,$(BUILD_DIR)/%.o,$(SRC_FILES))

# Default target
all: $(TARGET)

# Build the main target
$(TARGET): $(OBJ_FILES) | $(BUILD_DIR)
	$(NVCC) $(CXXFLAGS) $^ -o $@

# Compile object files
$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cu
	@mkdir -p $(dir $@)
	$(NVCC) $(CXXFLAGS) $(COMPILE_OBJ_FLAGS) -c $< -o $@

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
