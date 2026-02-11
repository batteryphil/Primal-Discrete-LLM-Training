@echo off
if not exist "bin" mkdir bin
echo Building CPU...
cl /EHsc /O2 /openmp main.cpp cpu_kernel.cpp gpu_dummy.cpp /Fe:bin/primal_cpu.exe
echo Building GPU...
nvcc -O3 -o bin/primal_gpu.exe main.cpp cpu_kernel.cpp gpu_kernel.cu
echo Done.
