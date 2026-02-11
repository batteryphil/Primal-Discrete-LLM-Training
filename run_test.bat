@echo off
cd src/engine
call build.bat
if exist bin\primal_cpu.exe (
    echo Running CPU Benchmark...
    bin\primal_cpu.exe
)
if exist bin\primal_gpu.exe (
    echo Running GPU Benchmark...
    bin\primal_gpu.exe gpu
)
pause
