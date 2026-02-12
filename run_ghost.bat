@echo off
set "PYTHONPATH=%cd%"
call vcvars64.bat
python primal_train_ghost.py
pause
