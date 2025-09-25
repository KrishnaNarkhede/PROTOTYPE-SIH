@echo off
echo BioMapper Prototype - AI-Enhanced Deep-Sea eDNA Analysis
echo ========================================================

echo.
echo Choose an option:
echo 1. Run Analysis (Command Line)
echo 2. Start Web Interface
echo 3. Install Dependencies
echo.

set /p choice="Enter choice (1-3): "

if "%choice%"=="1" (
    echo.
    echo Running analysis with sample data...
    python biomapper_prototype.py biodiversity_sample.fasta
    pause
) else if "%choice%"=="2" (
    echo.
    echo Starting web interface...
    echo Open http://localhost:5000 in your browser
    python app.py
) else if "%choice%"=="3" (
    echo.
    echo Installing dependencies...
    pip install -r requirements.txt
    echo Installation complete!
    pause
) else (
    echo Invalid choice!
    pause
)