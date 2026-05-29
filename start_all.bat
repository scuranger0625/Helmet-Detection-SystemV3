@echo off

echo ========================================
echo Starting Helmet Detection System...
echo ========================================

:: ------------------------------------------------
:: Backend
:: ------------------------------------------------
start cmd /k "cd /d C:\Users\Leon\Desktop\project\backend && C:\Users\Leon\Desktop\project\venv\Scripts\activate.bat && python app.py"

:: ------------------------------------------------
:: Wait backend boot
:: ------------------------------------------------
timeout /t 5 > nul

:: ------------------------------------------------
:: Frontend
:: ------------------------------------------------
start cmd /k "cd /d C:\Users\Leon\Desktop\project\frontend && npm run dev"

:: ------------------------------------------------
:: Wait frontend boot
:: ------------------------------------------------
timeout /t 5 > nul

:: ------------------------------------------------
:: Open Browser
:: ------------------------------------------------
start http://localhost:5173

echo.
echo System Started Successfully 😎
echo.

pause