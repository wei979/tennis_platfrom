@echo off
echo 🎾 Smart Tennis 專案啟動腳本
echo ================================

echo 正在啟動後端服務...
start "Smart Tennis Backend" cmd /k "cd /d %~dp0 && call tennis_env\Scripts\activate.bat && cd backend && python app.py"

echo 等待後端啟動...
timeout /t 3 /nobreak > nul

echo 正在啟動前端服務...
start "Smart Tennis Frontend" cmd /k "cd /d %~dp0\frontend && npm start"

echo.
echo 🎉 專案啟動完成！
echo.
echo 後端服務: http://localhost:5000
echo 前端服務: http://localhost:3000
echo.
echo 請在瀏覽器中訪問 http://localhost:3000
pause