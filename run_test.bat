@echo off
echo [Test] Initializing Visual Studio Environment...
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"

echo [Test] Building Engine...
cd src\engine
call build.bat

if exist trinity_cpu.exe (
    echo.
    echo =========================================
    echo [Test] Running CPU Benchmark (OpenMP/AVX)
    echo =========================================
    trinity_cpu.exe ..\..\models\trinity_1.58bit_packed.bin
)

if exist trinity_gpu.exe (
    echo.
    echo =========================================
    echo [Test] Running GPU Benchmark (CUDA)
    echo =========================================
    trinity_gpu.exe ..\..\models\trinity_1.58bit_packed.bin gpu
)
