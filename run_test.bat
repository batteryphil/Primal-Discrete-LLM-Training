@echo off
cd src/engine
call build.bat

echo.
echo === TESTING CPU ===
if exist bin\primal_cpu.exe bin\primal_cpu.exe
if not exist bin\primal_cpu.exe echo [FAIL] CPU Binary Missing.

echo.
echo === TESTING GPU ===
if exist bin\primal_gpu.exe bin\primal_gpu.exe gpu
if not exist bin\primal_gpu.exe echo [FAIL] GPU Binary Missing.

echo.
pause
