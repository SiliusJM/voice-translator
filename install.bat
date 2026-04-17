@echo off
setlocal
cd /d "%~dp0"

echo ============================================================
echo  Local Voice-to-Voice Translator -- Instalacion
echo  Requiere: Python 3.11, GPU NVIDIA (RTX 3060+ recomendada)
echo  Compatible: RTX 3060/3070/3080/4060/4070/4080/5060/5070+
echo ============================================================
echo.

:: --- Verificar Python 3.11 ---
echo Verificando Python 3.11...
py -3.11 --version >nul 2>&1
if errorlevel 1 (
    echo.
    echo ============================================================
    echo  ERROR: Python 3.11 NO encontrado.
    echo ============================================================
    echo.
    echo  Descarga e instala Python 3.11 desde:
    echo    https://www.python.org/ftp/python/3.11.9/python-3.11.9-amd64.exe
    echo.
    echo  IMPORTANTE durante la instalacion:
    echo    [x] Marca "Add Python to PATH"
    echo    [x] Marca "Install py launcher for all users"
    echo.
    pause
    exit /b 1
)
for /f "tokens=*" %%v in ('py -3.11 --version') do echo   Encontrado: %%v

:: --- Verificar GPU NVIDIA ---
echo.
echo Verificando GPU NVIDIA...
nvidia-smi >nul 2>&1
if errorlevel 1 (
    echo.
    echo ============================================================
    echo  ERROR: No se detecto GPU NVIDIA o drivers no instalados.
    echo ============================================================
    echo.
    echo  Este programa requiere una GPU NVIDIA con CUDA.
    echo  GPUs compatibles: RTX 3060, 3070, 3080, 4060, 4070, 4080, 5060, 5070, etc.
    echo.
    echo  Descarga los drivers mas recientes desde:
    echo    https://www.nvidia.com/Download/index.aspx
    echo.
    echo  Si tienes una GPU AMD o Intel, este programa no es compatible.
    echo.
    pause
    exit /b 1
)
for /f "tokens=4 delims= " %%v in ('nvidia-smi --query-gpu=name --format=csv,noheader') do echo   GPU detectada: %%v
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader

:: --- [1/4] Crear entorno virtual limpio ---
echo.
echo [1/4] Creando entorno virtual limpio...
if exist .venv (
    echo   Borrando .venv anterior...
    rmdir /s /q .venv
)
py -3.11 -m venv .venv
if errorlevel 1 ( echo ERROR al crear venv. & pause & exit /b 1 )
echo   Entorno .venv creado.

set PY=.venv\Scripts\python.exe
set PIP=.venv\Scripts\pip.exe
%PY% -m pip install --upgrade pip --quiet

:: --- [2/4] Instalar dependencias de la app PRIMERO ---
echo.
echo [2/4] Instalando dependencias de la app...
%PIP% install -r requirements.txt
if errorlevel 1 ( echo. & echo ERROR en dependencias. & pause & exit /b 1 )

:: --- [3/4] PyTorch CUDA AL FINAL para reemplazar versiones CPU ---
:: chatterbox-tts y transformers descargan torchvision CPU de PyPI.
:: Instalamos PyTorch CUDA despues para pisarlos con la version correcta.
:: Se auto-detecta la version CUDA del driver para compatibilidad.
echo.
echo [3/4] Detectando version CUDA e instalando PyTorch...
echo.

:: Detectar version CUDA del driver
set CUDA_URL=https://download.pytorch.org/whl/cu128
for /f "tokens=*" %%i in ('%PY% -c "import subprocess,re; o=subprocess.check_output(['nvidia-smi'],text=True); m=re.search(r'CUDA Version: (\d+)\.(\d+)',o); major=int(m.group(1)) if m else 0; minor=int(m.group(2)) if m else 0; url='cu118' if major<12 else ('cu121' if minor<4 else ('cu124' if minor<8 else 'cu128')); print(url)"') do set CUDA_TAG=%%i

if "%CUDA_TAG%"=="" (
    echo   No se detecto CUDA, usando cu128 por defecto...
    set CUDA_TAG=cu128
)
set CUDA_URL=https://download.pytorch.org/whl/%CUDA_TAG%
echo   Detectado: %CUDA_TAG% -- Instalando desde %CUDA_URL%
echo.
%PIP% install torch torchvision torchaudio --index-url %CUDA_URL% --force-reinstall --no-deps
if errorlevel 1 ( echo. & echo ERROR en PyTorch CUDA. & pause & exit /b 1 )

:: --- [4/4] Verificar GPU ---
echo.
echo [4/4] Verificando GPU...
%PY% -c "import torch; c=torch.cuda.is_available(); print('CUDA:', c); print('GPU:', torch.cuda.get_device_name(0) if c else 'NO ENCONTRADA')"

echo.
echo ============================================================
echo  Instalacion completada!  Usa run.bat para iniciar la app.
echo ============================================================
pause
exit /b 0
