@echo off
if not exist "src\engine\bin" mkdir "src\engine\bin" 2>nul
echo Building Primal DLL...

:: Build from project root context or adjust paths
:: Correcting paths assuming script is run from root or src/engine
:: Actually, let's just force the compiler to look in src/engine if files are not found, 
:: OR we assume this script is run from src/engine and we fix the caller.
:: The previous run_test.bat did `cd src/engine` then `call build.bat`.
:: The user's command was `src\engine\build.bat` from root.
:: So %~dp0 is the script's directory.

pushd %~dp0
mkdir bin 2>nul

nvcc -O3 -shared -Xcompiler "/openmp /EHsc /D_ALLOW_COMPILER_AND_STL_VERSION_MISMATCH /D_ALLOW_ITERATOR_DEBUG_LEVEL_MISMATCH" ^
    -allow-unsupported-compiler ^
    -lcudart ^
    -o bin/primal_engine.dll ^
    primal.cpp cpu_kernel.cpp gpu_kernel.cu primal_lib.cpp

if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] DLL Build Failed.
    popd
    exit /b 1
)

echo [SUCCESS] bin/primal_engine.dll created.
popd
