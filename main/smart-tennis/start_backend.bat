@echo off
echo 🎾 啟動 Smart Tennis 後端服務...
cd /d "%~dp0"
call tennis_env\Scripts\activate.bat
cd backend
python app.py
pause