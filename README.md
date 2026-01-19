# Smart Tennis

智慧網球分析系統（後端 Flask + 前端 React）。支援上傳網球影片，並以 YOLOv8 檢測與追蹤球體、估算擊球與速度，輸出分析結果與處理後影片。

---

## 目錄
- 專案介紹
- 專案架構
- 前置需求
- 安裝與啟動
  - macOS/Linux（不用 .bat）
  - Windows（不用 .bat）
  - 一鍵腳本（macOS/Linux）
- 使用流程
- API 說明
- 影片編碼與相容性（處理後影片）
- 疑難排解

---

## 專案介紹
- 後端：Flask 提供 API，OpenCV + Ultralytics YOLOv8 進行影像偵測與追蹤
- 前端：React + TypeScript（Create React App），Tailwind UI
- 分析內容：
  - 網球追蹤（YOLOv8 + 自訂邏輯）
  - 擊球點及類型（簡化偵測）
  - 速度分析（像素速度估算與粗略換算 km/h）
- 產物：
  - JSON 分析結果：`output/<file_id>_analysis.json`
  - 處理後影片（含標註）：`output/<file_id>_processed.mp4`（或 `_processed_h264.mp4`）

---

## 專案架構

```
main/smart-tennis/
├── backend/                 # 後端 Flask 服務
│   ├── app.py               # 入口（/api/*）
│   ├── tennis_tracker.py    # YOLO 偵測 + 追蹤 + 標註輸出
│   ├── shot_detector.py     # 擊球偵測（簡化）
│   ├── speed_analyzer.py    # 速度分析與統計
│   ├── requirements.txt     # 後端依賴
│   └── ...
├── frontend/                # 前端 React 應用
│   ├── package.json         # NPM 腳本
│   └── src/
│       ├── pages/           # Upload / Analysis / Results
│       ├── services/api.ts  # 與後端串接
│       └── ...
├── models/                  # 模型檔（yolov8n.pt）
├── uploads/                 # 上傳影片暫存目錄
├── output/                  # 分析輸出與處理後影片
├── test_api.py              # 後端 API 快速測試
├── test_environment.py      # 後端環境快速檢查
├── install.sh               # macOS/Linux 安裝腳本
├── start.sh                 # macOS/Linux 一鍵啟動
└── README.md                # 你正在看的文件
```

---

## 模型下載

- 若要使用自訂網球偵測模型 `main\model\last.pt`，請先從 Google 雲端硬碟下載檔案並放到對應路徑：
  - 下載連結：https://drive.google.com/file/d/1MTtgjH7V-WCOIt9S8zdjjNWtXHOIscv0/view?usp=drive_link
  - 建議放置路徑：`D:\work\Tennis\main\model\last.pt`（Windows 示例）
  - 專案後端會讀取環境變數 `YOLO_MODEL_PATH` 以載入此模型；若未設定，預設使用 `../models/yolov8n.pt`

設定方式（Windows PowerShell）
- 使用提供的批次檔（已內建環境變數）：
  - `main\smart-tennis\start_backend.bat`
  - `main\smart-tennis\start_project.bat`
- 或手動設定環境變數後啟動後端：
  - `$env:YOLO_MODEL_PATH = "D:\work\Tennis\main\model\last.pt"`
  - `python main\smart-tennis\backend\app.py`


## 前置需求
- Python 3.10+（建議 3.11/3.12 以上）
- Node.js 16+（建議 LTS）
- 建議安裝：ffmpeg（或安裝 `imageio-ffmpeg` 套件，後端會自動下載內建 ffmpeg）
  - 後端 requirements 已內建 `imageio-ffmpeg`，建議保留以確保處理後影片可在瀏覽器播放

---

## 安裝與啟動

### A. macOS/Linux（不用 .bat）
1) 建立虛擬環境與安裝後端
```
cd main/smart-tennis
python3 -m venv .venv
source .venv/bin/activate
pip install -r backend/requirements.txt  
```

2) 安裝前端依賴
```
cd frontend
npm install
```

3) 啟動
- 後端（終端 1）：
```
cd main/smart-tennis
source .venv/bin/activate
python backend/app.py
```
- 前端（終端 2）：
```
cd main/smart-tennis/frontend
npm start
```
- 瀏覽器開啟 http://localhost:3000

### B. Windows（不用 .bat）
1) 建立虛擬環境與安裝後端
```
cd main\smart-tennis
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r backend/requirements.txt
```

2) 安裝前端依賴
```
cd frontend
npm install
```

3) 啟動
- 後端（終端 1）：
```
cd main\smart-tennis                                                                                           
..venv\Scripts\Activate.ps1                                                                                   
可選：$env:YOLO_MODEL_PATH = "D:\work\Tennis\main\model\last.pt"                                               
python backend\app.py
```
- 前端（終端 2）：
```
cd main\smart-tennis\frontend
npm start
```
- 瀏覽器開啟 http://localhost:3000

### C. 一鍵腳本（macOS/Linux）
```
cd main/smart-tennis
chmod +x install.sh start.sh
./install.sh
./start.sh
```
注意：`start.sh` 會同時啟動後端與前端，並顯示開啟網址。

---

## 使用流程
1. 開啟前端（http://localhost:3000）
2. 進入「上傳影片」頁，選擇 MP4/AVI/MOV/MKV 檔案（最大 100MB）
3. 上傳完成會導向「分析中」頁，等待分析結束
4. 轉跳到「結果」頁，可瀏覽：
   - 總結統計、正反手分佈、速度分析
   - 「影片」分頁可播放 原始影片 與 處理後影片

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
如需跨域部署，可在前端設定 `REACT_APP_API_URL` 指向 API 主機（`/src/services/api.ts` 會優先採用該環境變數）。

---

## 影片編碼與相容性（處理後影片）
- OpenCV 直接輸出的 MP4 並不一定是瀏覽器可直接播放的 H.264/yuv420p/faststart 格式
- 本專案在分析結束後會嘗試使用 `imageio-ffmpeg` 轉檔：
  - 參數：`libx264 + yuv420p + +faststart`
  - 若 Windows 正在讀取導致檔案鎖定，會以非破壞方式輸出 `*_h264.mp4`，回傳時優先使用此檔案
- 建議：在後端環境安裝 `imageio-ffmpeg`（或系統 `ffmpeg`）以確保前端可順利播放處理後影片

---

## 疑難排解
- 後端啟動失敗
  - 檢查虛擬環境已啟用、依賴安裝完成
  - 執行 `python test_environment.py` 檢查必要模組
- 前端無法連上後端
  - 確認後端在 `http://localhost:5000`、前端在 `http://localhost:3000`
  - 檢查 `frontend/package.json` 之 `proxy`
- 分析很久或逾時
  - 本專案已將分析請求的逾時設為不逾時（前端 `analyzeVideo` 設 `timeout: 0`）
  - 影片較長或硬體效能有限，分析需數分鐘屬正常
- 處理後影片無法播放或只播放片段
  - 請確認已安裝 `imageio-ffmpeg`
  - 重新進入結果頁（會重新請求影片，前端 URL 會添加時間戳避免快取）
  - 若有問題，檢查 `output/` 下是否有 `_processed_h264.mp4`，用 `ffprobe` 檢視 `codec_name=h264`、`pix_fmt=yuv420p`

---

## 測試
- 快速測 API 健康狀態：
```
cd main/smart-tennis
python test_api.py
```

---

