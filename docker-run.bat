@echo off
REM ═══════════════════════════════════════════════════════════
REM EmotiScan - Windows Docker Launcher
REM Usage:  docker-run.bat           (CPU mode for school PCs)
REM         docker-run.bat --gpu     (with NVIDIA GPU)
REM ═══════════════════════════════════════════════════════════

echo.
echo  ╔═══════════════════════════════════════╗
echo  ║     EmotiScan Docker Launcher         ║
echo  ╚═══════════════════════════════════════╝
echo.

REM Check if Docker is running
docker info >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo [ERROR] Docker is not running!
    echo Please start Docker Desktop first.
    echo.
    pause
    exit /b 1
)

REM Check for --gpu flag
set PROFILES=--profile cpu
set SERVICE=emotiscan
if "%1"=="--gpu" (
    set PROFILES=--profile gpu
    set SERVICE=emotiscan-gpu
    echo   GPU mode enabled
)

echo [1/3] Building EmotiScan image (first run takes ~5-10 min)...
docker compose %PROFILES% build %SERVICE%

echo.
echo [2/3] Starting EmotiScan...
docker compose %PROFILES% up -d

echo.
echo [3/3] Waiting for app to start...
timeout /t 10 /nobreak >nul

echo.
echo ═══════════════════════════════════════════
echo   EmotiScan is running!
echo   Open: http://localhost:80
echo.
echo   To stop:  docker compose %PROFILES% down
echo   Logs:     docker compose %PROFILES% logs -f
echo ═══════════════════════════════════════════
echo.

REM Open browser
start http://localhost:80

pause
