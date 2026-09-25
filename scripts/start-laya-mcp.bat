@echo off
title Laya MCP Server :8765

echo ==========================================
echo   Laya MCP Server Launcher
echo   Stop it: close its window or Ctrl+C there
echo ==========================================
echo.

:: --- CONFIG ------------------------------------------------------------------
set LAYA_DIR=D:\Laya_Sandbox
set LAYA_MCP_PORT=8765
set LAYA_MCP_HOST=0.0.0.0
set LAYA_MCP_THREADS=6
set LAYA_MCP_PRELOAD=english
set HF_HUB_DISABLE_SYMLINKS_WARNING=1
:: Uncomment once scripts\prefetch_models.py has downloaded all three checkpoints,
:: so the server never reaches out to Hugging Face:
:: set HF_HUB_OFFLINE=1
:: Other settings: see %LAYA_DIR%\.env.example (a .env file in %LAYA_DIR% is also read).

:: -----------------------------------------------------------------------------
echo [1/3] Checking for a running instance...
curl -s http://localhost:%LAYA_MCP_PORT%/health >NUL 2>&1
if %ERRORLEVEL%==0 (
    echo    Laya MCP is already running on port %LAYA_MCP_PORT%.
    goto STATUS
)
echo    None found.
echo.

echo [2/3] Starting server (loads the english model, ~30-60 s)...
start "Laya MCP Server :%LAYA_MCP_PORT%" cmd /k "cd /d "%LAYA_DIR%" && venv\Scripts\laya-mcp.exe"
set /a WAITED=0
:WAIT_LOOP
timeout /t 3 /nobreak >NUL
set /a WAITED+=3
curl -s http://localhost:%LAYA_MCP_PORT%/health >NUL 2>&1
if %ERRORLEVEL%==0 goto READY
if %WAITED% geq 180 (
    echo    [!!] Not up after 180 s - check the "Laya MCP Server" window for errors.
    goto END
)
goto WAIT_LOOP
:READY
echo    Server ready after ~%WAITED% s.
echo.

:STATUS
echo [3/3] Status
curl -s http://localhost:%LAYA_MCP_PORT%/health
echo.
echo.
echo ==========================================
echo   Local:     http://localhost:%LAYA_MCP_PORT%/mcp
for /f "usebackq delims=" %%i in (`powershell -NoProfile -Command "(Get-NetIPAddress -AddressFamily IPv4 | Where-Object { $_.InterfaceAlias -notmatch 'Loopback|vEthernet' -and $_.IPAddress -notlike '169.254*' -and $_.IPAddress -notlike '192.168.56.*' } | Select-Object -First 1).IPAddress"`) do set LAYA_IP=%%i
echo   Intranet:  http://%LAYA_IP%:%LAYA_MCP_PORT%/mcp
echo.
echo   Add to Claude Code on a team machine:
echo   claude mcp add --transport http laya http://%LAYA_IP%:%LAYA_MCP_PORT%/mcp --header "X-Laya-Team: dev"
echo ==========================================

:END
pause
