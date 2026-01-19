@echo off
echo 🎾 啟動 Smart Tennis 後端服務...
cd /d "%~dp0"
call tennis_env\Scripts\activate.bat
set "YOLO_MODEL_PATH=D:\work\Tennis\main\model\last.pt"
cd backend
python app.py
pause