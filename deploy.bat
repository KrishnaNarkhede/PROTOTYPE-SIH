@echo off
echo 🚀 BioMapper AI - Vercel Deployment Script
echo ==========================================

echo.
echo 📋 Step 1: Checking prerequisites...
where git >nul 2>nul
if %errorlevel% neq 0 (
    echo ❌ Git not found. Please install Git first.
    pause
    exit /b 1
)

where vercel >nul 2>nul
if %errorlevel% neq 0 (
    echo ❌ Vercel CLI not found. Installing...
    npm install -g vercel
    if %errorlevel% neq 0 (
        echo ❌ Failed to install Vercel CLI. Please install manually: npm install -g vercel
        pause
        exit /b 1
    )
)

echo ✅ Prerequisites check complete

echo.
echo 📋 Step 2: Preparing repository...
if not exist .git (
    echo Initializing Git repository...
    git init
)

echo Adding files to Git...
git add .
git status

echo.
set /p commit_msg="Enter commit message (or press Enter for default): "
if "%commit_msg%"=="" set commit_msg=Deploy BioMapper AI to Vercel

echo Committing changes...
git commit -m "%commit_msg%"

echo.
echo 📋 Step 3: Vercel Login
echo Please login to Vercel when prompted...
vercel login

echo.
echo 📋 Step 4: Deploying to Vercel...
echo This will deploy your BioMapper AI application to Vercel.
echo.
pause

vercel

echo.
echo 📋 Step 5: Production Deployment
echo.
set /p deploy_prod="Deploy to production? (y/N): "
if /i "%deploy_prod%"=="y" (
    echo Deploying to production...
    vercel --prod
    echo.
    echo 🎉 Production deployment complete!
) else (
    echo Skipping production deployment.
)

echo.
echo ✅ Deployment process complete!
echo.
echo 🌐 Your BioMapper AI is now deployed on Vercel
echo Check your Vercel dashboard for the deployment URL
echo.
pause