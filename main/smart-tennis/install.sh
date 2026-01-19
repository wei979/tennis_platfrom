#!/bin/bash

echo "🎾 Smart Tennis 項目安裝腳本"
echo "================================"

# 檢查 Python
if ! command -v python3 &> /dev/null; then
    echo "❌ 錯誤: 需要安裝 Python 3"
    exit 1
fi

# 檢查 Node.js
if ! command -v node &> /dev/null; then
    echo "❌ 錯誤: 需要安裝 Node.js"
    exit 1
fi

echo "✅ Python 和 Node.js 已安裝"

# 安裝後端依賴
echo "📦 安裝後端依賴..."
cd backend
python3 -m pip install -r requirements.txt
cd ..

# 安裝前端依賴
echo "📦 安裝前端依賴..."
cd frontend
npm install
cd ..

echo "🎉 安裝完成！"
echo ""
echo "啟動方式："
echo "1. 後端: cd backend && python app.py"
echo "2. 前端: cd frontend && npm start"
echo ""
echo "然後訪問 http://localhost:3000"
