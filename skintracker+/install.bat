@echo off
echo 🧴 SkinTrack+ Installation Helper
echo =================================

echo.
echo Checking for Python installation...

python --version >nul 2>&1
if %errorlevel% equ 0 (
    echo ✅ Python found!
    goto :install_packages
)

python3 --version >nul 2>&1
if %errorlevel% equ 0 (
    echo ✅ Python3 found!
    goto :install_packages
)

echo ❌ Python not found!
echo.
echo Please install Python 3.8 or higher:
echo.
echo Option 1: Download from python.org
echo   - Go to https://www.python.org/downloads/
echo   - Download Python 3.8 or higher
echo   - During installation, CHECK "Add Python to PATH"
echo.
echo Option 2: Install from Microsoft Store
echo   - Open Microsoft Store
echo   - Search for "Python 3.11" or higher
echo   - Install the official Python app
echo.
echo After installing Python, run this script again.
echo.
pause
exit /b 1

:install_packages
echo.
echo Installing required packages...
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

if %errorlevel% equ 0 (
    echo.
    echo ✅ Installation completed successfully!
    echo.
    echo To run SkinTrack+:
    echo   streamlit run skintrack_app.py
    echo.
    echo For help, see README.md
) else (
    echo.
    echo ❌ Installation failed. Please check the error messages above.
)

echo.
pause
