#!/bin/bash

echo "🚀 啟動 Smart Tennis 應用"
echo "========================="

# 檢查是否在正確的目錄
if [ ! -f "README.md" ]; then
    echo "❌ 請在 smart-tennis 根目錄執行此腳本"
    exit 1
fi

# 啟動後端（背景執行）
echo "🔧 啟動後端服務..."
cd backend
python3 app.py &
BACKEND_PID=$!
cd ..

# 等待後端啟動
sleep 3

# 啟動前端
echo "🎨 啟動前端服務..."
cd frontend
npm start &
FRONTEND_PID=$!
cd ..

echo "✅ 服務已啟動！"
echo "📝 後端: http://localhost:5000"
echo "🌐 前端: http://localhost:3000"
echo ""
echo "按 Ctrl+C 停止所有服務"

# 等待中斷信號
trap "echo '🛑 正在停止服務...'; kill $BACKEND_PID $FRONTEND_PID; exit" INT

wait
