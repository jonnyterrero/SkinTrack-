#!/usr/bin/env pwsh
<#
.SYNOPSIS
    SkinTrack+ Installation Helper
    
.DESCRIPTION
    This script helps install Python dependencies for SkinTrack+
#>

Write-Host "🧴 SkinTrack+ Installation Helper" -ForegroundColor Cyan
Write-Host "=================================" -ForegroundColor Cyan
Write-Host ""

# Check for Python installation
Write-Host "Checking for Python installation..." -ForegroundColor Yellow

$pythonCmd = $null
$pythonVersion = $null

# Try different Python commands
$pythonCommands = @("python", "python3", "py")

foreach ($cmd in $pythonCommands) {
    try {
        $result = & $cmd --version 2>$null
        if ($LASTEXITCODE -eq 0) {
            $pythonCmd = $cmd
            $pythonVersion = $result
            break
        }
    }
    catch {
        continue
    }
}

if ($pythonCmd) {
    Write-Host "✅ Python found: $pythonVersion" -ForegroundColor Green
} else {
    Write-Host "❌ Python not found!" -ForegroundColor Red
    Write-Host ""
    Write-Host "Please install Python 3.8 or higher:" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "Option 1: Download from python.org" -ForegroundColor White
    Write-Host "  - Go to https://www.python.org/downloads/" -ForegroundColor Gray
    Write-Host "  - Download Python 3.8 or higher" -ForegroundColor Gray
    Write-Host "  - During installation, CHECK 'Add Python to PATH'" -ForegroundColor Gray
    Write-Host ""
    Write-Host "Option 2: Install from Microsoft Store" -ForegroundColor White
    Write-Host "  - Open Microsoft Store" -ForegroundColor Gray
    Write-Host "  - Search for 'Python 3.11' or higher" -ForegroundColor Gray
    Write-Host "  - Install the official Python app" -ForegroundColor Gray
    Write-Host ""
    Write-Host "After installing Python, run this script again." -ForegroundColor Yellow
    Write-Host ""
    Read-Host "Press Enter to exit"
    exit 1
}

# Create directories
Write-Host ""
Write-Host "Creating data directories..." -ForegroundColor Yellow
$directories = @("skintrack_data", "skintrack_data/images", "skintrack_data/models")

foreach ($dir in $directories) {
    if (!(Test-Path $dir)) {
        New-Item -ItemType Directory -Path $dir -Force | Out-Null
        Write-Host "  Created: $dir" -ForegroundColor Gray
    } else {
        Write-Host "  Exists: $dir" -ForegroundColor Gray
    }
}

# Install packages
Write-Host ""
Write-Host "Installing required packages..." -ForegroundColor Yellow

try {
    # Upgrade pip
    Write-Host "  Upgrading pip..." -ForegroundColor Gray
    & $pythonCmd -m pip install --upgrade pip | Out-Null
    
    # Install requirements
    Write-Host "  Installing packages from requirements.txt..." -ForegroundColor Gray
    & $pythonCmd -m pip install -r requirements.txt
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host ""
        Write-Host "✅ Installation completed successfully!" -ForegroundColor Green
        Write-Host ""
        Write-Host "To run SkinTrack+:" -ForegroundColor Cyan
        Write-Host "  streamlit run skintrack_app.py" -ForegroundColor White
        Write-Host ""
        Write-Host "For help, see README.md" -ForegroundColor Gray
    } else {
        Write-Host ""
        Write-Host "❌ Installation failed. Please check the error messages above." -ForegroundColor Red
    }
}
catch {
    Write-Host ""
    Write-Host "❌ Error during installation: $($_.Exception.Message)" -ForegroundColor Red
}

Write-Host ""
Read-Host "Press Enter to exit"
