@echo off
setlocal

set "ROOT=%~dp0"
set "COMMAND=%~1"

if "%COMMAND%"=="" goto :usage

shift

node "%ROOT%scripts\%COMMAND%.js" %*
goto :eof

:usage
echo Usage:
echo   rwa command [options]
echo.
echo Commands:
for %%f in (scripts\*.js) do echo - %%~nf

exit /b 1
