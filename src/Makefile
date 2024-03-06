# Makefile for VGG16 project using oneDNN

# Compiler settings
CXX := /home/samjons/thesis/llvm-EPI-development-toolchain-cross/bin/clang++
CXXFLAGS := -g -I/home/samjons/thesis/oneDNN/src
LDFLAGS := -L/usr/local/lib -ldnnl -L/home/samjons/thesis/llvm-EPI-toolchain-cross/sysroot/lib -static -lomp

# Target executable and source files
TARGET := ../build/VGG16
SOURCE := VGG16.cpp

# Default target
all: $(TARGET)

$(TARGET): $(SOURCE)
	$(CXX) $(CXXFLAGS) -o $@ $< $(LDFLAGS)

# Clean target for cleaning up build artifacts
clean:
	rm -f $(TARGET)

.PHONY: all clean
