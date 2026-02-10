# Trinity Engine Makefile

CXX=g++
NVCC=nvcc
CXXFLAGS=-O3 -fopenmp -std=c++17
NVCCFLAGS=-O3 -arch=sm_61 -std=c++17 --compiler-options -fopenmp

SRC_DIR=.
BUILD_DIR=build

all: cpu

# CPU Build
cpu: $(SRC_DIR)/main.cpp $(SRC_DIR)/cpu_kernel.cpp
	$(CXX) $(CXXFLAGS) -o trinity_cpu $(SRC_DIR)/main.cpp $(SRC_DIR)/cpu_kernel.cpp

# GPU Build
gpu: $(SRC_DIR)/main.cpp $(SRC_DIR)/cpu_kernel.cpp $(SRC_DIR)/gpu_kernel.cu
	$(NVCC) $(NVCCFLAGS) -o trinity_gpu $(SRC_DIR)/main.cpp $(SRC_DIR)/cpu_kernel.cpp $(SRC_DIR)/gpu_kernel.cu

clean:
	rm -f trinity_cpu trinity_gpu *.o
