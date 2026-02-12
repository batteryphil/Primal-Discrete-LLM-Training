@echo off
set "PYTHONPATH=%cd%"
call vcvars64.bat
echo [*] Logging to training.log. Run 'python monitor_primal.py' to view output.
python primal_train_ghost.py > training.log 2>&1
pause
