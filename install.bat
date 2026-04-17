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
for /f "tokens=*" %%v in ('nvidia-smi --query-gpu=name --format^=csv^,noheader') do echo   GPU detectada: %%v
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
:: GPUs Blackwell (RTX 50xx) requieren PyTorch NIGHTLY cu128 (sm_120).
echo.
echo [3/4] Detectando version CUDA e instalando PyTorch...
echo.

:: Detectar version CUDA del driver
set CUDA_TAG=
set IS_BLACKWELL=0
for /f "tokens=*" %%i in ('%PY% -c "import subprocess,re; o=subprocess.check_output(['nvidia-smi'],text=True); m=re.search(r'CUDA Version: (\d+)\.(\d+)',o); major=int(m.group(1)) if m else 0; minor=int(m.group(2)) if m else 0; url='cu118' if major<12 else ('cu121' if minor<4 else ('cu124' if minor<8 else 'cu128')); print(url)"') do set CUDA_TAG=%%i

:: Detectar si es GPU Blackwell (RTX 50xx) — requiere nightly cu128
for /f "tokens=*" %%g in ('nvidia-smi --query-gpu=name --format=csv,noheader') do set GPU_NAME=%%g
echo %GPU_NAME% | findstr /i "5060 5070 5080 5090 5050" >nul 2>&1
if not errorlevel 1 (
    echo   GPU Blackwell detectada: %GPU_NAME%
    echo   Requiere PyTorch nightly con cu128 (arquitectura sm_120)
    set CUDA_TAG=cu128
    set IS_BLACKWELL=1
)

if "%CUDA_TAG%"=="" (
    echo   No se detecto CUDA, usando cu128 por defecto...
    set CUDA_TAG=cu128
)

:: Blackwell usa nightly (unica build con kernels sm_120)
:: Otras GPUs usan stable (mas estable y probado)
if "%IS_BLACKWELL%"=="1" (
    set CUDA_URL=https://download.pytorch.org/whl/nightly/cu128
    echo   Instalando PyTorch NIGHTLY cu128 (requerido para RTX 50xx)
    echo   (La descarga es ~2.7 GB, puede tardar varios minutos)
    echo.
    %PIP% install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128 --force-reinstall --no-deps
) else (
    set CUDA_URL=https://download.pytorch.org/whl/%CUDA_TAG%
    echo   Detectado: %CUDA_TAG% -- Instalando desde %CUDA_URL%
    echo   (La descarga es ~2.7 GB, puede tardar varios minutos)
    echo.
    %PIP% install torch torchvision torchaudio --index-url %CUDA_URL% --force-reinstall --no-deps
)
if errorlevel 1 (
    echo.
    echo   ERROR: No se pudo instalar PyTorch CUDA.
    echo   Intentando de nuevo sin cache...
    echo.
    if "%IS_BLACKWELL%"=="1" (
        %PIP% install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128 --force-reinstall --no-deps --no-cache-dir
    ) else (
        %PIP% install torch torchvision torchaudio --index-url %CUDA_URL% --force-reinstall --no-deps --no-cache-dir
    )
    if errorlevel 1 (
        echo.
        echo   ERROR CRITICO: PyTorch CUDA no se pudo instalar.
        echo   La app funcionara en modo CPU (mucho mas lento).
        if "%IS_BLACKWELL%"=="1" (
            echo   Para instalar manualmente despues:
            echo     .venv\Scripts\pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128 --force-reinstall --no-deps
        ) else (
            echo   Para instalar manualmente despues:
            echo     .venv\Scripts\pip install torch torchvision torchaudio --index-url %CUDA_URL% --force-reinstall --no-deps
        )
        echo.
        pause
    )
)

:: --- [4/4] Verificar GPU ---
:: Se escribe un script .py temporal para evitar problemas de comillas en CMD.
:: Se ejecuta en subproceso para que un crash/segfault no cierre el instalador.
echo.
echo [4/4] Verificando instalacion...
(
echo import torch
echo import warnings
echo warnings.filterwarnings^('default'^)
echo v = torch.__version__
echo c = torch.cuda.is_available^(^)
echo g = torch.cuda.get_device_name^(0^) if c else 'N/A'
echo print^('  PyTorch: ' + v^)
echo print^('  CUDA en wheel: SI' if 'cu' in v else '  CUDA en wheel: NO -- version CPU instalada!'^)
echo print^('  CUDA disponible: SI' if c else '  CUDA disponible: NO'^)
echo print^('  GPU: ' + g^)
) > _verify_gpu.py
%PY% _verify_gpu.py
del _verify_gpu.py 2>nul

echo.
echo ============================================================
echo  Instalacion completada!  Usa run.bat para iniciar la app.
echo ============================================================
echo.
pause
exit /b 0
