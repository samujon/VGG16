# Makefile for VGG16 project using oneDNN

# Compiler settings
CXX := /home/samjons/thesis/llvm-EPI-development-toolchain-cross/bin/clang++
CXXFLAGS := -g -I/home/samjons/thesis/oneDNN/src -march=rv64gcv
LDFLAGS := -L/usr/local/lib -ldnnl -L/home/samjons/thesis/llvm-EPI-development-toolchain-cross/sysroot/lib  /home/samjons/thesis/llvm-EPI-development-toolchain-cross/riscv64-unknown-linux-gnu/sysroot/lib/libomp.a -static
# Target executable and source  
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
