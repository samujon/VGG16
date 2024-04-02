# Makefile for VGG16 project using oneDNN

# Compiler settings
CXX := /home/samjons/thesis/RISCV/bin/riscv64-unknown-linux-gnu-g++
CXXFLAGS := -g -I/home/samjons/thesis/oneDNN/src
LDFLAGS := -L/usr/local/lib -ldnnl -L/home/samjons/thesis/RISCV/sysroot/lib -lgomp -static

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
