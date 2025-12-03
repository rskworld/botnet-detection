@echo off
REM Push Botnet Detection Project to GitHub
REM Project: Botnet Detection with Machine Learning
REM Developer: RSK World

echo ========================================
echo Pushing to GitHub Repository
echo ========================================
echo.

echo Checking git status...
git status

echo.
echo Pushing to main branch...
git push -u origin main

echo.
echo Pushing tag v1.0.0...
git push origin v1.0.0

echo.
echo ========================================
echo Done! Check your GitHub repository:
echo https://github.com/rskworld/botnet-detection
echo ========================================
pause

