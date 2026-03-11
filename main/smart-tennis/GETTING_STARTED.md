# Smart Tennis 環境配置與安裝指南

## 目錄
- [系統需求](#系統需求)
- [環境安裝](#環境安裝)
  - [Windows](#windows)
  - [macOS / Linux](#macos--linux)
- [模型檔案](#模型檔案)
- [環境變數](#環境變數)
- [啟動專案](#啟動專案)
  - [Windows](#windows-啟動)
  - [macOS / Linux](#macos--linux-啟動)
- [驗證安裝](#驗證安裝)
- [GPU 加速（可選）](#gpu-加速可選)
- [常見問題](#常見問題)

---

## 系統需求

| 項目 | 最低需求 | 建議 |
|------|----------|------|
| **Python** | 3.10+ | 3.11 / 3.12 / 3.13 |
| **Node.js** | 16+ | 18 LTS 或 20 LTS |
| **npm** | 8+ | 隨 Node.js 安裝 |
| **RAM** | 4GB | 8GB+ |
| **磁碟空間** | 2GB | 5GB+（含模型與虛擬環境） |
| **GPU（可選）** | - | NVIDIA CUDA 相容 GPU |
| **作業系統** | Windows 10 / macOS 12 / Ubuntu 20.04 | Windows 11 / macOS 14 / Ubuntu 22.04 |

### 檢查版本
```bash
python --version    # 或 python3 --version
node --version
npm --version
```

---

## 環境安裝

### Windows

#### 1. 建立 Python 虛擬環境
```powershell
cd main\smart-tennis
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

> 如果遇到 PowerShell 執行政策錯誤，先執行：
> ```powershell
> Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
> ```

#### 2. 安裝後端依賴
```powershell
pip install --upgrade pip
pip install -r backend/requirements.txt
```

#### 3. 安裝前端依賴
```powershell
cd frontend
npm install
cd ..
```

### macOS / Linux

#### 1. 建立 Python 虛擬環境
```bash
cd main/smart-tennis
python3 -m venv .venv
source .venv/bin/activate
```

#### 2. 安裝後端依賴
```bash
pip install --upgrade pip
pip install -r backend/requirements.txt
```

#### 3. 安裝前端依賴
```bash
cd frontend
npm install
cd ..
```

#### 4. 一鍵安裝（macOS / Linux）
```bash
chmod +x install.sh
./install.sh
```

---

## 模型檔案

### 已包含於 Repo（無需額外操作）

| 模型 | 路徑 | 大小 | 用途 |
|------|------|------|------|
| Tennis Ball (best) | `models/tennis_ball/best.pt` | 5.2MB | 網球偵測（預設使用） |
| YOLOv8 Nano | `models/yolov8n.pt` | 6.3MB | 通用物件偵測（fallback） |
| 3D Pose 權重 | `../../checkpoint/` | ~130MB | VideoPose3D / GCN / MLP / ST-GCN |

### 自動下載（首次執行時）

| 模型 | 說明 |
|------|------|
| YOLO11-Pose Large | Ultralytics 會在首次使用時自動下載 `yolo11l-pose.pt`（~51MB） |
| MediaPipe Hand | `hand_landmarker.task` 由 MediaPipe 自動下載 |

### Checkpoint 權重目錄結構

```
checkpoint/
├── poseaug/                         # PoseAug 訓練後權重
│   ├── gcn/gt/poseaug/ckpt_best_dhp_p1.pth.tar       (1.1MB)
│   ├── mlp/gt/poseaug/ckpt_best_dhp_p1.pth.tar       (17MB)
│   ├── stgcn/gt/poseaug/ckpt_best_dhp_p1.pth.tar     (14MB)
│   └── videopose/gt/poseaug/ckpt_best_dhp_p1.pth.tar  (33MB)  ← 預設載入
└── pretrain_baseline/               # 預訓練基線權重（fallback）
    ├── gcn/gt/pretrain/ckpt_best.pth.tar              (1.1MB)
    ├── mlp/gt/pretrain/ckpt_best.pth.tar              (17MB)
    ├── stgcn/gt/pretrain/ckpt_best.pth.tar            (14MB)
    └── videopose/gt/pretrain/ckpt_best.pth.tar        (33MB)
```

後端啟動時會依序嘗試載入：
1. `checkpoint/poseaug/videopose/gt/poseaug/ckpt_best_dhp_p1.pth.tar`
2. `checkpoint/pretrain_baseline/videopose/gt/pretrain/ckpt_best.pth.tar`
3. 如果都找不到，使用啟發式深度估計作為 fallback

---

## 環境變數

複製 `.env.example` 為 `.env` 來自訂配置（可選，所有項目都有預設值）：

```bash
cp .env.example .env
```

| 變數名稱 | 預設值 | 說明 |
|----------|--------|------|
| `FLASK_ENV` | `development` | Flask 環境模式 |
| `FLASK_DEBUG` | `True` | 除錯模式 |
| `API_HOST` | `0.0.0.0` | API 監聽位址 |
| `API_PORT` | `5000` | API 監聽端口 |
| `UPLOAD_FOLDER` | `../uploads` | 上傳影片暫存路徑 |
| `OUTPUT_FOLDER` | `../output` | 分析輸出路徑 |
| `MAX_CONTENT_LENGTH` | `104857600` | 上傳檔案大小限制（100MB） |
| `CONFIDENCE_THRESHOLD` | `0.3` | YOLO 偵測信心閾值 |
| `USE_POSE` | `true` | 是否啟用姿態偵測 |
| `POSE_MODEL_SIZE` | `l` | 姿態模型大小（n/s/m/l/x） |
| `POSE_MIN_DETECTION_CONFIDENCE` | `0.5` | MediaPipe 偵測信心閾值 |
| `POSE_MIN_TRACKING_CONFIDENCE` | `0.5` | MediaPipe 追蹤信心閾值 |
| `COURT_LENGTH` | `23.77` | 球場長度（公尺） |
| `COURT_WIDTH` | `10.97` | 球場寬度（公尺） |

---

## 啟動專案

### Windows 啟動

#### 方法 A：批次檔（推薦）
```powershell
# 一鍵啟動前後端
main\smart-tennis\start_project.bat

# 或僅啟動後端
main\smart-tennis\start_backend.bat
```

#### 方法 B：手動啟動
開啟兩個終端：

**終端 1 — 後端**
```powershell
cd main\smart-tennis
.\.venv\Scripts\Activate.ps1
cd backend
python app.py
```

**終端 2 — 前端**
```powershell
cd main\smart-tennis\frontend
npm start
```

### macOS / Linux 啟動

#### 方法 A：一鍵腳本
```bash
cd main/smart-tennis
./start.sh
```

#### 方法 B：手動啟動
開啟兩個終端：

**終端 1 — 後端**
```bash
cd main/smart-tennis
source .venv/bin/activate
python backend/app.py
```

**終端 2 — 前端**
```bash
cd main/smart-tennis/frontend
npm start
```

### 服務端口

| 服務 | URL | 說明 |
|------|-----|------|
| 後端 API | http://localhost:5000 | Flask 服務 |
| 前端 UI | http://localhost:3000 | React 開發伺服器 |

> 前端透過 `package.json` 中 `proxy: "http://localhost:5000"` 自動代理 API 請求。

---

## 驗證安裝

### 1. 檢查後端健康狀態
```bash
curl http://localhost:5000/api/health
```
預期回應：
```json
{"status": "ok"}
```

### 2. 檢查前端
瀏覽器開啟 http://localhost:3000，應看到 Smart Tennis 首頁。

### 3. 檢查模型載入
後端啟動時，終端應顯示：
```
使用自訓練網球模型: .../models/tennis_ball/best.pt
已載入 YOLO 模型: .../models/tennis_ball/best.pt
✅ PoseAug 姿態增強模組已載入
✅ PoseAug GAN 模組已載入
[OK] VideoPose3D weights loaded: .../checkpoint/poseaug/videopose/gt/poseaug/ckpt_best_dhp_p1.pth.tar
```

---

## GPU 加速（可選）

預設安裝的 PyTorch 為 CPU 版本。如需 GPU 加速：

### NVIDIA CUDA GPU
1. 確認已安裝 [NVIDIA 驅動](https://www.nvidia.com/drivers)
2. 解除安裝 CPU 版 PyTorch：
   ```bash
   pip uninstall torch torchvision
   ```
3. 安裝 CUDA 版（以 CUDA 12.4 為例）：
   ```bash
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
   ```
4. 驗證：
   ```python
   python -c "import torch; print(torch.cuda.is_available())"
   # 應輸出 True
   ```

> 請至 [PyTorch 官網](https://pytorch.org/get-started/locally/) 選擇對應你 CUDA 版本的安裝指令。

---

## 常見問題

### 後端啟動失敗

**症狀：** `ModuleNotFoundError`
```bash
# 確認虛擬環境已啟用
# Windows:
.\.venv\Scripts\Activate.ps1
# macOS/Linux:
source .venv/bin/activate

# 重新安裝依賴
pip install -r backend/requirements.txt
```

**症狀：** 模型載入失敗
- 確認 `models/tennis_ball/best.pt` 存在
- 確認 `checkpoint/` 目錄下有權重檔

### 前端啟動失敗

**症狀：** `npm ERR!` 或 依賴錯誤
```bash
cd frontend
rm -rf node_modules package-lock.json
npm install
```

### 前端無法連上後端

- 確認後端在 `http://localhost:5000` 運行
- 確認 `frontend/package.json` 中 `"proxy": "http://localhost:5000"`
- 檢查防火牆是否阻擋端口

### 分析耗時過長

- 較長影片或無 GPU 環境下，分析需數分鐘屬正常
- 可將 `POSE_MODEL_SIZE` 改為 `n`（nano）以加速，但精度會下降
- 前端已設定 `timeout: 0`（不逾時）

### 處理後影片無法播放

- 確認已安裝 `imageio-ffmpeg`（已包含在 requirements.txt）
- 檢查 `output/` 下是否有 `_processed_h264.mp4` 檔案
- 嘗試重新整理結果頁

### PowerShell 執行政策錯誤

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### 端口被佔用

```bash
# 查看佔用端口的進程
# Windows:
netstat -ano | findstr :5000
# macOS/Linux:
lsof -i :5000

# 結束佔用的進程
# Windows:
taskkill /F /PID <PID>
# macOS/Linux:
kill -9 <PID>
```
