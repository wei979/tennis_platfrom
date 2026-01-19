# 🚀 Smart Tennis 執行指南

## 前置條件檢查

請確保您的系統已安裝以下軟體：

### 1. Python 3 ✅ (已安裝 - Python 3.13.2)
```bash
python3 --version  # 應該顯示 Python 3.x.x
```

### 2. Node.js ❌ (需要安裝)
```bash
# 檢查是否已安裝
node --version
npm --version

# 如果未安裝，請執行：
brew install node

# 或從官網下載：https://nodejs.org/
```

## 執行步驟

### 步驟 1: 安裝依賴

#### 安裝後端依賴 (Python)
```bash
cd backend
pip3 install -r requirements.txt
```

#### 安裝前端依賴 (Node.js) - 需要先安裝 Node.js
```bash
cd frontend
npm install
```

### 步驟 2: 啟動服務

#### 方法 A: 使用自動腳本（推薦）
```bash
# 在項目根目錄執行
./start.sh
```

#### 方法 B: 手動啟動

**終端 1 - 啟動後端:**
```bash
cd backend
python3 app.py
```
後端將在 http://localhost:5000 運行

**終端 2 - 啟動前端:**
```bash
cd frontend
npm start
```
前端將在 http://localhost:3000 運行

### 步驟 3: 使用應用

1. 在瀏覽器中訪問 http://localhost:3000
2. 點擊「上傳影片」
3. 選擇或拖拽網球影片檔案
4. 等待 AI 分析完成
5. 查看詳細的分析結果

## 支援的影片格式

- MP4
- AVI  
- MOV
- MKV
- 檔案大小限制：100MB

## 疑難排解

### 如果遇到 "command not found: node"
1. 確保 Node.js 已正確安裝
2. 重新啟動終端
3. 檢查 PATH 環境變數

### 如果 Python 模組安裝失敗
```bash
# 嘗試升級 pip
pip3 install --upgrade pip

# 使用虛擬環境（推薦）
python3 -m venv venv
source venv/bin/activate  # macOS/Linux
pip install -r requirements.txt
```

### 如果前端啟動失敗
```bash
# 清除 npm 快取
npm cache clean --force

# 刪除 node_modules 並重新安裝
rm -rf node_modules
npm install
```

## 下一步

安裝完成後，您就可以：
- 上傳網球影片進行 AI 分析
- 查看網球追蹤結果
- 分析正反手擊球統計
- 檢視速度分析報告
