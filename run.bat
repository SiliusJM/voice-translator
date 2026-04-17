@echo off
setlocal

if not exist .venv (
    echo ERROR: El entorno virtual no existe. Ejecuta install.bat primero.
    pause
    exit /b 1
)

echo Iniciando Voice-to-Voice Translator...
echo (El primer arranque descarga ~3.5 GB de modelos, puede tardar varios minutos)
echo.
.venv\Scripts\python.exe app.py
pause
