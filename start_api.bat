@echo off
echo ======================================================================
echo Starting FastAPI Server
echo ======================================================================
echo.
echo Command: uvicorn src.serving.api:app --port 8000
echo.
echo After server starts, open: http://localhost:8000/docs
echo Press Ctrl+C to stop the server
echo ======================================================================
echo.

cd /d C:\Users\benam\Downloads\cyber-anomaly-detection-mlops
uvicorn src.serving.api:app --port 8000
