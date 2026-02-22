@echo off
set "NGROK_EXE=%~dp0ngrok.exe"

echo Starting ngrok tunnel for dashboard (Port 8000)...
echo.
echo [IMPORTANT] 
echo 1. Ensure 'ngrok.exe' is in this folder: %NGROK_EXE%
echo 2. If this is your first time, run: ngrok config add-authtoken <YOUR_TOKEN>
echo.
echo Launching...

if not exist "%NGROK_EXE%" (
    echo [ERROR] ngrok.exe not found!
    pause
    exit /b
)

"%NGROK_EXE%" http 8000 --log=stdout > ngrok.log 2>&1

echo.
echo Tunnel process ended. Checking logs for errors...
type ngrok.log
echo.
echo If you see 'authentication failed', you need to add your token.
pause
