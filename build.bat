@echo off
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"
set DISTUTILS_USE_SDK=1
python setup_primal_cuda.py build_ext --inplace
