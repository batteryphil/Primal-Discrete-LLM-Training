@echo off
setlocal

echo [Trinity] Building CPU Engine (MSVC + OpenMP)...
cl /std:c++17 /EHsc /O2 /openmp main.cpp cpu_kernel.cpp gpu_dummy.cpp /Fe:trinity_cpu.exe
if %ERRORLEVEL% EQU 0 (
    echo [Trinity] CPU Build Success: trinity_cpu.exe
) else (
    echo [Trinity] CPU Build FAILED. Ensure you are in a VS Developer Command Prompt.
)

echo.
echo [Trinity] Building GPU Engine (NVCC + CUDA)...
nvcc -O3 -arch=sm_61 -std=c++17 -allow-unsupported-compiler -D_ALLOW_COMPILER_AND_STL_VERSION_MISMATCH -o trinity_gpu.exe main.cpp cpu_kernel.cpp gpu_kernel.cu
if %ERRORLEVEL% EQU 0 (
    echo [Trinity] GPU Build Success: trinity_gpu.exe
) else (
    echo [Trinity] GPU Build FAILED. Check CUDA installation and cl.exe path.
)

endlocal
