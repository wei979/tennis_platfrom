# Smart Tennis

智慧網球分析系統（後端 Flask + 前端 React）。支援上傳網球影片，以 YOLOv8 檢測與追蹤球體、估算擊球與速度，並提供 3D 人體姿態估計與增強分析，輸出分析結果與處理後影片。

---

## 目錄
- [專案介紹](#專案介紹)
- [專案架構](#專案架構)
- [前置需求](#前置需求)
- [模型下載](#模型下載)
- [安裝與啟動](#安裝與啟動)
- [使用流程](#使用流程)
- [API 說明](#api-說明後端)
- [影片編碼與相容性](#影片編碼與相容性處理後影片)
- [疑難排解](#疑難排解)

---

## 專案介紹
- **後端**：Flask 提供 API，OpenCV + Ultralytics YOLO 進行影像偵測與追蹤
- **前端**：React + TypeScript（Create React App），Tailwind UI
- **分析內容**：
  - 網球追蹤（YOLOv8 + 自訂邏輯）
  - 擊球點及類型偵測
  - 速度分析（像素速度估算與粗略換算 km/h）
  - 3D 人體姿態估計（YOLO11-Pose + VideoPose3D）
  - 姿態增強（PoseAug GAN）
  - 手部 / 球拍偵測（MediaPipe + YOLO）
  - 多視角 3D 重建
- **產物**：
  - JSON 分析結果：`output/<file_id>_analysis.json`
  - 3D 分析結果：`output/<file_id>_3d_analysis.json`
  - 處理後影片（含標註）：`output/<file_id>_processed.mp4`

---

## 專案架構

```
Tennis_detect_project/
├── README.md                        # 本文件
├── .gitignore
├── .gitattributes
│
├── checkpoint/                      # 3D 姿態估計預訓練權重
│   ├── poseaug/                     # PoseAug 訓練後權重
│   │   ├── gcn/gt/poseaug/          #   GCN 模型
│   │   ├── mlp/gt/poseaug/          #   MLP 模型
│   │   ├── stgcn/gt/poseaug/        #   ST-GCN 模型
│   │   └── videopose/gt/poseaug/    #   VideoPose3D 模型
│   └── pretrain_baseline/           # 預訓練基線權重
│       ├── gcn/gt/pretrain/
│       ├── mlp/gt/pretrain/
│       ├── stgcn/gt/pretrain/
│       └── videopose/gt/pretrain/
│
├── main/
│   ├── model/                       # 自訂模型（需另外下載）
│   │   ├── last.pt                  #   自訂網球偵測模型
│   │   ├── keypoints.pth            #   球場關鍵點模型
│   │   └── readme.md                #   下載說明
│   │
│   ├── models/                      # 內建模型（已含於 repo）
│   │   ├── yolov8n.pt               #   YOLOv8 Nano
│   │   └── tennis_ball/best.pt      #   網球偵測微調模型
│   │
│   ├── training/                    # 訓練用 Notebook
│   │   ├── Detectron2Training.ipynb
│   │   ├── TennisCourtKeypointsTraining.ipynb
│   │   └── YOLO11Traning.ipynb
│   │
│   └── smart-tennis/                # 主應用程式
│       ├── backend/                 # 後端 Flask 服務
│       │   ├── app.py               #   API 入口（/api/*）
│       │   ├── tennis_tracker.py    #   YOLO 偵測 + 追蹤 + 標註
│       │   ├── shot_detector.py     #   擊球偵測
│       │   ├── speed_analyzer.py    #   速度分析與統計
│       │   ├── pose_detector.py     #   YOLO11-Pose 2D 姿態偵測
│       │   ├── pose_3d_reconstructor.py  # 2D→3D 姿態重建
│       │   ├── videopose3d_lifter.py     # VideoPose3D 模型
│       │   ├── poseaug_gan.py       #   PoseAug GAN 姿態增強
│       │   ├── pose_augmentor.py    #   姿態增強整合
│       │   ├── hand_detector.py     #   MediaPipe 手部偵測
│       │   ├── racket_detector.py   #   球拍偵測
│       │   ├── multiview_processor.py   # 多視角處理
│       │   ├── setup_models.py      #   模型初始化
│       │   ├── requirements.txt     #   後端依賴
│       │   ├── yolo11l-pose.pt      #   YOLO11 Pose Large（需另外放置）
│       │   └── hand_landmarker.task #   MediaPipe 手部模型
│       │
│       ├── frontend/                # 前端 React 應用
│       │   ├── package.json
│       │   ├── tailwind.config.js
│       │   ├── tsconfig.json
│       │   ├── postcss.config.js
│       │   ├── public/
│       │   │   ├── index.html
│       │   │   ├── favicon.ico
│       │   │   └── manifest.json
│       │   └── src/
│       │       ├── App.tsx           #   路由與主框架
│       │       ├── index.tsx
│       │       ├── App.css
│       │       ├── index.css
│       │       ├── services/
│       │       │   └── api.ts        #   後端 API 串接
│       │       ├── components/
│       │       │   ├── Navbar.tsx     #   導覽列
│       │       │   ├── Skeleton3DViewer.tsx    # 3D 骨架檢視器
│       │       │   ├── CameraAngleSelector.tsx # 攝影機角度選擇
│       │       │   ├── CameraConfigSelector.tsx # 攝影機設定
│       │       │   └── MultiResultsDisplay.tsx  # 多結果顯示
│       │       └── pages/
│       │           ├── HomePage.tsx           # 首頁
│       │           ├── UploadPage.tsx         # 上傳影片
│       │           ├── AnalysisPage.tsx       # 分析中
│       │           ├── ResultsPage.tsx        # 分析結果
│       │           ├── Analysis3DPage.tsx     # 3D 姿態分析
│       │           ├── PoseAugPage.tsx        # 姿態增強
│       │           ├── MultiUploadPage.tsx    # 多視角上傳
│       │           ├── MultiResultsPage.tsx   # 多視角結果
│       │           └── PlayerDashboardPage.tsx # 球員儀表板
│       │
│       ├── uploads/                 # 上傳影片暫存（gitignore）
│       ├── output/                  # 分析輸出（gitignore）
│       ├── models/                  # 模型副本
│       ├── .env.example             # 環境變數範例
│       ├── .gitignore
│       ├── install.sh               # macOS/Linux 安裝腳本
│       ├── start.sh                 # macOS/Linux 一鍵啟動
│       ├── start_backend.bat        # Windows 後端啟動
│       ├── start_project.bat        # Windows 一鍵啟動
│       └── GETTING_STARTED.md       # 快速入門指南
│
├── output/                          # 分析輸出（gitignore）
└── uploads/                         # 上傳暫存（gitignore）
```

---

## 前置需求
- Python 3.10+（建議 3.11/3.12 以上）
- Node.js 16+（建議 LTS）
- 建議安裝：ffmpeg（或安裝 `imageio-ffmpeg` 套件，後端會自動下載內建 ffmpeg）
  - 後端 requirements 已內建 `imageio-ffmpeg`，建議保留以確保處理後影片可在瀏覽器播放
- GPU（可選）：支援 CUDA 的 NVIDIA GPU 可大幅加速 YOLO 推論與 3D 姿態估計

---

## 模型下載

### 已包含於 Repo
| 模型 | 路徑 | 用途 |
|------|------|------|
| YOLOv8 Nano | `main/models/yolov8n.pt` | 通用物件偵測 |
| Tennis Ball | `main/models/tennis_ball/best.pt` | 網球偵測微調模型 |
| 3D Pose 權重 | `checkpoint/` | VideoPose3D / GCN / MLP / ST-GCN 預訓練權重 |

### 需另外下載
| 模型 | 放置路徑 | 下載方式 |
|------|----------|----------|
| 自訂網球偵測 | `main/model/last.pt` | [Google Drive](https://drive.google.com/file/d/1MTtgjH7V-WCOIt9S8zdjjNWtXHOIscv0/view?usp=drive_link) |
| 球場關鍵點 | `main/model/keypoints.pth` | 同上 |
| YOLO11 Pose Large | `main/smart-tennis/backend/yolo11l-pose.pt` | Ultralytics 自動下載或手動放置 |
| MediaPipe 手部 | `main/smart-tennis/backend/hand_landmarker.task` | MediaPipe 自動下載 |

設定方式（Windows PowerShell）：
- 使用提供的批次檔（已內建環境變數）：
  - `main\smart-tennis\start_backend.bat`
  - `main\smart-tennis\start_project.bat`
- 或手動設定環境變數後啟動後端：
  ```powershell
  $env:YOLO_MODEL_PATH = "main\model\last.pt"
  python main\smart-tennis\backend\app.py
  ```

---

## 安裝與啟動

### A. macOS/Linux
1) 建立虛擬環境與安裝後端
```bash
cd main/smart-tennis
python3 -m venv .venv
source .venv/bin/activate
pip install -r backend/requirements.txt
```

2) 安裝前端依賴
```bash
cd frontend
npm install
```

3) 啟動
- 後端（終端 1）：
```bash
cd main/smart-tennis
source .venv/bin/activate
python backend/app.py
```
- 前端（終端 2）：
```bash
cd main/smart-tennis/frontend
npm start
```
- 瀏覽器開啟 http://localhost:3000

### B. Windows
1) 建立虛擬環境與安裝後端
```powershell
cd main\smart-tennis
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r backend/requirements.txt
```

2) 安裝前端依賴
```powershell
cd frontend
npm install
```

3) 啟動
- 後端（終端 1）：
```powershell
cd main\smart-tennis
.\.venv\Scripts\Activate.ps1
# 可選：$env:YOLO_MODEL_PATH = "main\model\last.pt"
python backend\app.py
```
- 前端（終端 2）：
```powershell
cd main\smart-tennis\frontend
npm start
```
- 瀏覽器開啟 http://localhost:3000

### C. 一鍵腳本（macOS/Linux）
```bash
cd main/smart-tennis
chmod +x install.sh start.sh
./install.sh
./start.sh
```

---

## 使用流程
1. 開啟前端（http://localhost:3000）
2. 進入「上傳影片」頁，選擇 MP4/AVI/MOV/MKV 檔案（最大 100MB）
3. 上傳完成會導向「分析中」頁，等待分析結束
4. 轉跳到「結果」頁，可瀏覽：
   - 總結統計、正反手分佈、速度分析
   - 「影片」分頁可播放原始影片與處理後影片
5. 進入「3D 姿態分析」頁，可進行：
   - 2D→3D 人體姿態估計
   - 3D 骨架即時檢視
6. 進入「姿態增強」頁，可使用 PoseAug GAN 進行姿態數據增強
7. 「多視角上傳」支援同時上傳多角度影片進行 3D 重建

---

## API 說明（後端）
- `GET  /api/health` 健康檢查
- `POST /api/upload` 上傳影片（表單欄位：`video`）
  - 回應：`{ success, file_id, filename, video_info }`
- `POST /api/analyze/{file_id}` 執行分析
  - 回應：`{ success, results }`（包含追蹤、擊球、速度與 summary）
- `GET  /api/results/{file_id}` 取得分析結果（JSON）
- `GET  /api/video/{file_id}` 取得原始上傳影片
- `GET  /api/processed-video/{file_id}` 取得處理後影片（MP4）

前端預設會透過 CRA 的 proxy（`package.json` 裡 `proxy: http://localhost:5000`）連線後端。
如需跨域部署，可在前端設定 `REACT_APP_API_URL` 指向 API 主機。

---

## 影片編碼與相容性（處理後影片）
- OpenCV 直接輸出的 MP4 並不一定是瀏覽器可直接播放的 H.264/yuv420p/faststart 格式
- 本專案在分析結束後會嘗試使用 `imageio-ffmpeg` 轉檔：
  - 參數：`libx264 + yuv420p + +faststart`
  - 若 Windows 正在讀取導致檔案鎖定，會以非破壞方式輸出 `*_h264.mp4`，回傳時優先使用此檔案
- 建議：在後端環境安裝 `imageio-ffmpeg`（或系統 `ffmpeg`）以確保前端可順利播放處理後影片

---

## 疑難排解
- **後端啟動失敗**
  - 檢查虛擬環境已啟用、依賴安裝完成
  - 確認模型檔已放置到正確路徑
- **前端無法連上後端**
  - 確認後端在 `http://localhost:5000`、前端在 `http://localhost:3000`
  - 檢查 `frontend/package.json` 之 `proxy`
- **分析很久或逾時**
  - 前端 `analyzeVideo` 已設 `timeout: 0`（不逾時）
  - 影片較長或硬體效能有限，分析需數分鐘屬正常
- **處理後影片無法播放或只播放片段**
  - 請確認已安裝 `imageio-ffmpeg`
  - 重新進入結果頁（前端 URL 會添加時間戳避免快取）
  - 檢查 `output/` 下是否有 `_processed_h264.mp4`
- **3D 姿態分析無結果**
  - 確認 `checkpoint/` 目錄下有對應的權重檔
  - 確認 `yolo11l-pose.pt` 已放置於 `backend/` 目錄

---
