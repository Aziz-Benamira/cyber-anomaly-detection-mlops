@echo off
echo ======================================================================
echo 🛡️ Starting MoE Cybersecurity Detection Dashboard
echo ======================================================================
echo.
echo Make sure the API is running:
echo   docker-compose -f docker/docker-compose.yml up api
echo.
echo Dashboard will open in your browser...
echo Press Ctrl+C to stop the dashboard
echo ======================================================================
echo.

cd /d C:\Users\benam\Downloads\cyber-anomaly-detection-mlops
streamlit run src/serving/streamlit_app.py --server.port 8501
