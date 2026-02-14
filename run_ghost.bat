@echo off
set "PYTHONPATH=%cd%"
call vcvars64.bat
echo [*] Launching Training + Monitor...
start "PRIMAL MONITOR" cmd /k "python monitor_primal.py"
python primal_train_ghost.py > training.log 2>&1
pause
