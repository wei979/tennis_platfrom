from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os
import cv2
import json
import numpy as np
from werkzeug.utils import secure_filename
# 假設這些類別在您的專案中存在，如果實際不存在，請確保它們有適當的 Mock
# ❗ 提醒: 確保您的 tennis_tracker, shot_detector, speed_analyzer 類別存在並可匯入
from tennis_tracker import TennisTracker
from shot_detector import ShotDetector
from speed_analyzer import SpeedAnalyzer

# 3D 重建模組 (可選)
try:
    from multiview_processor import MultiViewProcessor, create_default_camera_setup
    from pose_3d_reconstructor import Pose3DReconstructor, reconstruct_3d_from_multiview
    MULTIVIEW_3D_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ 3D 重建模組不可用: {e}")
    MULTIVIEW_3D_AVAILABLE = False

# PoseAug 姿態增強模組 (可選)
try:
    from pose_augmentor import (
        PoseAugmentor, PoseAugmentationGenerator, AugmentationConfig,
        create_sample_pose, validate_pose, KEYPOINT_NAMES, SKELETON_BONES
    )
    POSEAUG_AVAILABLE = True
    print("✅ PoseAug 姿態增強模組已載入")
except ImportError as e:
    print(f"⚠️ PoseAug 模組不可用: {e}")
    POSEAUG_AVAILABLE = False

# 預先定義路徑常數 (供後續模組載入使用)
CURRENT_DIR = os.path.abspath(os.path.dirname(__file__))

# PoseAug GAN 模組 (原始論文的 GAN 增強)
try:
    from poseaug_gan import PoseAugGAN
    poseaug_gan = PoseAugGAN()
    POSEAUG_GAN_AVAILABLE = True
    print("✅ PoseAug GAN 模組已載入")

    # 嘗試載入預訓練權重
    GAN_WEIGHTS_DIR = os.path.join(CURRENT_DIR, 'poseaug_weights')
    if os.path.exists(GAN_WEIGHTS_DIR):
        # 搜尋可用的權重檔案
        for f in os.listdir(GAN_WEIGHTS_DIR):
            if f.endswith(('.bin', '.pth', '.pt')):
                weight_path = os.path.join(GAN_WEIGHTS_DIR, f)
                poseaug_gan.load_weights(weight_path)
                break
    else:
        print(f"   預訓練權重目錄不存在: {GAN_WEIGHTS_DIR}")
        print(f"   使用隨機初始化 Generator (可下載權重到此目錄以提升品質)")
except Exception as e:
    print(f"⚠️ PoseAug GAN 模組不可用: {e}")
    POSEAUG_GAN_AVAILABLE = False
    poseaug_gan = None

# VideoPose3D 2D→3D 提升模組
try:
    from videopose3d_lifter import VideoPose3DLifter
    videopose3d_lifter = VideoPose3DLifter()
    VIDEOPOSE3D_AVAILABLE = True

    # 嘗試載入預訓練權重 (優先 poseaug 版本)
    CHECKPOINT_BASE = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(CURRENT_DIR))), 'checkpoint')
    VIDEOPOSE3D_WEIGHT_PATHS = [
        os.path.join(CHECKPOINT_BASE, 'poseaug', 'videopose', 'gt', 'poseaug', 'ckpt_best_dhp_p1.pth.tar'),
        os.path.join(CHECKPOINT_BASE, 'pretrain_baseline', 'videopose', 'gt', 'pretrain', 'ckpt_best.pth.tar'),
    ]
    vp3d_loaded = False
    for wp in VIDEOPOSE3D_WEIGHT_PATHS:
        if os.path.exists(wp):
            if videopose3d_lifter.load_weights(wp):
                vp3d_loaded = True
                break
    if not vp3d_loaded:
        print(f"   VideoPose3D: 未找到權重檔案，搜尋路徑: {CHECKPOINT_BASE}")
        print(f"   將使用舊版啟發式深度估計作為 fallback")

except Exception as e:
    print(f"⚠️ VideoPose3D 模組不可用: {e}")
    VIDEOPOSE3D_AVAILABLE = False
    videopose3d_lifter = None

# YOLO 網球拍偵測模組
try:
    from racket_detector import RacketDetector, compute_racket_3d_from_detection
    _racket_model = os.path.join(os.path.dirname(CURRENT_DIR), 'models', 'yolov8n.pt')
    if not os.path.exists(_racket_model):
        _racket_model = 'yolov8n.pt'
    racket_detector = RacketDetector(model_path=_racket_model)
    RACKET_DETECT_AVAILABLE = True
    print(f"YOLO Racket detector loaded: {_racket_model}")
except Exception as e:
    print(f"Racket detector not available: {e}")
    RACKET_DETECT_AVAILABLE = False
    racket_detector = None
import uuid
from datetime import datetime
import subprocess
import traceback

# =========================================================
# 應用程式初始化
# =========================================================
app = Flask(__name__)
CORS(app)

# 獲取 app.py 所在的目錄 (.../main/smart-tennis/backend) — 已在前面定義
BASE_DIR_MAIN = os.path.dirname(os.path.dirname(CURRENT_DIR))

# 設定路徑：使用專案根目錄下的 uploads 和 output
UPLOAD_FOLDER = os.path.join(BASE_DIR_MAIN, 'uploads') 
OUTPUT_FOLDER = os.path.join(BASE_DIR_MAIN, 'output') 
METADATA_FILE = os.path.join(BASE_DIR_MAIN, 'video_metadata.json') # 元數據檔案路徑

ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024 # 100MB 限制

# 確保資料夾存在
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# 姿態辨識設定
USE_POSE = os.getenv('USE_POSE', 'true').lower() == 'true'
POSE_MODEL_SIZE = os.getenv('POSE_MODEL_SIZE', 'l')  # 'n', 's', 'm', 'l', 'x'

# 初始化分析器
try:
    # ❗ 提醒: 檢查您的環境變數 YOLO_MODEL_PATH 是否已設定
    print(f"📊 姿態辨識設定: USE_POSE={USE_POSE}, POSE_MODEL_SIZE={POSE_MODEL_SIZE}")
    tennis_tracker = TennisTracker(
        model_path=os.getenv('YOLO_MODEL_PATH', None),
        use_pose=USE_POSE,
        pose_model_size=POSE_MODEL_SIZE
    )
    shot_detector = ShotDetector(
        use_pose=USE_POSE,
        pose_model_size=POSE_MODEL_SIZE
    )
    speed_analyzer = SpeedAnalyzer()
except Exception as e:
    print(f"警告: 分析器初始化失敗 - {e}。部分路由可能無法使用。")

# =========================================================
# 輔助函數
# =========================================================

def ensure_h264_mp4_safe(input_path: str):
    """嘗試轉碼影片為瀏覽器兼容的 H.264/MP4 格式。"""
    if not input_path.lower().endswith('.mp4'):
        return None
        
    try:
        import imageio_ffmpeg as ioff
        ffmpeg_exe = ioff.get_ffmpeg_exe()
    except Exception:
        print("提示: 建議安裝 imageio-ffmpeg 確保瀏覽器播放，目前跳過轉碼。")
        return None

    tmp_output = input_path.rsplit('.mp4', 1)[0] + '_h264_tmp.mp4'
    
    cmd = [
        ffmpeg_exe,
        '-y',
        '-i', input_path,
        '-c:v', 'libx264',
        '-pix_fmt', 'yuv420p',
        '-movflags', '+faststart',
        '-preset', 'veryfast',
        tmp_output
    ]
    
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=300) 
        os.replace(tmp_output, input_path)
        print(f"影片轉碼成功: {input_path}")
    except subprocess.CalledProcessError as e:
        print(f"轉碼失敗 (FFmpeg 錯誤碼): {e.returncode}")
        print(e.stderr.decode())
    except Exception as e:
        print(f"轉碼流程發生異常: {e}")
    finally:
        if os.path.exists(tmp_output):
            os.remove(tmp_output)
    
    return input_path

# ---------------------------------------------------------
# 元數據管理輔助函數
# ---------------------------------------------------------

def load_metadata():
    """從檔案載入所有影片元數據，如果檔案不存在則回傳空字典。"""
    if not os.path.exists(METADATA_FILE):
        return {}
    try:
        with open(METADATA_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        return {}

def save_metadata(metadata):
    """將所有影片元數據儲存到檔案。"""
    try:
        with open(METADATA_FILE, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=4)
        return True
    except Exception as e:
        print(f"儲存元數據失敗: {e}")
        return False
# ---------------------------------------------------------

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# =========================================================
# API 路由
# =========================================================

@app.route('/api/upload', methods=['POST'])
def upload_video():
    """上傳影片端點 (同步執行)"""
    try:
        if 'video' not in request.files:
            return jsonify({'success': False, 'error': '沒有選擇文件'}), 400
        
        file = request.files['video']
        if file.filename == '':
            return jsonify({'success': False, 'error': '沒有選擇文件'}), 400
        
        person_name = request.form.get('person_name', '未命名球員')
        
        if file and allowed_file(file.filename):
            file_id = str(uuid.uuid4())
            filename = secure_filename(file.filename)
            file_extension = filename.rsplit('.', 1)[1].lower()
            new_filename = f"{file_id}.{file_extension}" 
            
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], new_filename)
            file.save(file_path)
            
            # --- 儲存元數據的邏輯 ---
            metadata = load_metadata()
            metadata[file_id] = {
                'person_name': person_name,
                'original_filename': filename,
                'upload_time': datetime.now().isoformat(),
                'status': 'uploaded' 
            }
            save_metadata(metadata)
            # -----------------------------
            
            # 獲取影片資訊 (確保 cv2.VideoCapture 可用)
            cap = cv2.VideoCapture(file_path)
            fps = cap.get(cv2.CAP_PROP_FPS); frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = frame_count / fps if fps > 0 else 0
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cap.release()
            
            return jsonify({
                'success': True,
                'file_id': file_id,
                'filename': filename,
                'person_name': person_name, 
                'ext': file_extension,
                'video_info': {'duration': duration, 'fps': fps, 'width': width, 'height': height, 'frame_count': frame_count}
            })
            
        return jsonify({'success': False, 'error': '不支援的檔案格式'}), 400
        
    except Exception as e:
        print(f"上傳錯誤: {str(e)}")
        traceback.print_exc()
        return jsonify({'success': False, 'error': f'上傳失敗: {str(e)}'}), 500

@app.route('/api/analyze/<file_id>', methods=['POST'])
def analyze_video(file_id):
    """分析影片端點 (同步執行)"""
    try:
        # 尋找檔案 (遍歷查找原始影片)
        video_file = None
        for ext in ALLOWED_EXTENSIONS:
            potential_file = os.path.join(app.config['UPLOAD_FOLDER'], f"{file_id}.{ext}")
            if os.path.exists(potential_file):
                video_file = potential_file
                break
        
        if not video_file:
            return jsonify({'success': False, 'error': '找不到影片檔案'}), 404
        
        print(f"開始分析影片: {video_file}")
        
        # 1. 網球追蹤（同時輸出處理後影片）
        processed_video_path = os.path.join(app.config['OUTPUT_FOLDER'], f"{file_id}_processed.mp4")
        tracking_results = tennis_tracker.track_ball(video_file, output_path=processed_video_path)
        
        # 2. 3. 執行其他分析
        shot_results = shot_detector.detect_shots(video_file, tracking_results)
        speed_results = speed_analyzer.analyze_speed(tracking_results)

        # 轉碼處理後影片為瀏覽器兼容格式
        if os.path.exists(processed_video_path):
            ensure_h264_mp4_safe(processed_video_path)
        
        # 🎯 整合結果
        # 獲取姿態分析數據
        pose_analysis = shot_results.get('pose_analysis', {})
        pose_available = shot_results.get('pose_available', False)

        # 計算平均姿態評分 (如果有的話)
        shots_with_pose = [s for s in shot_results.get('shots', []) if s.get('pose_score')]
        avg_pose_score = round(sum(s['pose_score'] for s in shots_with_pose) / len(shots_with_pose), 1) if shots_with_pose else 0.0

        analysis_results = {
             'file_id': file_id,
             'timestamp': datetime.now().isoformat(),
             'tracking': tracking_results,
             'shots': shot_results,
             'speed': speed_results,
             'summary': {
                 'total_shots': len(shot_results.get('shots', [])),
                 'forehand_count': len([s for s in shot_results.get('shots', []) if s.get('type') == 'forehand']),
                 'backhand_count': len([s for s in shot_results.get('shots', []) if s.get('type') == 'backhand']),
                 'max_speed': speed_results.get('max_speed_kmh', 0.0),
                 'avg_speed': speed_results.get('avg_speed_kmh', 0.0),

                 'stabilityScore': tracking_results.get('stability_score', 0.0),
                 'hitHeight': shot_results.get('average_hit_height', 0.0),
                 'hitAngle': shot_results.get('average_hit_angle', 0.0),
                 'initial_speed': speed_results.get('avg_initial_speed_kmh', 0.0),

                 # 姿態分析數據
                 'pose_available': pose_available,
                 'pose_score': avg_pose_score,
                 'body_rotation': pose_analysis.get('avg_body_rotation', 0.0),
                 'arm_extension': pose_analysis.get('avg_arm_extension', 0.0),
                 'knee_bend': pose_analysis.get('avg_knee_bend', 0.0),
                 'balance_score': pose_analysis.get('avg_balance_score', 0.0),
                 'dominant_side': pose_analysis.get('dominant_side', None),
                 'pose_detected_frames': pose_analysis.get('pose_detected_frames', 0),
             }
          }
        
        # 保存結果
        result_file = os.path.join(app.config['OUTPUT_FOLDER'], f"{file_id}_analysis.json")
        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump(analysis_results, f, ensure_ascii=False, indent=2)
        
        # 💡 更新元數據狀態
        metadata = load_metadata()
        if file_id in metadata:
            metadata[file_id]['status'] = 'analyzed'
            metadata[file_id]['error'] = None 
            save_metadata(metadata)

        return jsonify({'success': True, 'message': '分析完成並儲存', 'results': analysis_results})
    
    except Exception as e:
        print(f"分析錯誤: {str(e)}")
        traceback.print_exc()
        
        # 💡 更新元數據狀態為 'failed'
        metadata = load_metadata()
        if file_id in metadata:
            metadata[file_id]['status'] = 'failed'
            metadata[file_id]['error'] = str(e)
            save_metadata(metadata)

        return jsonify({'success': False, 'error': f'分析失敗: {str(e)}'}), 500

@app.route('/api/results/<file_id>', methods=['GET'])
def get_results(file_id):
    """獲取分析結果 (用於結果頁面)"""
    try:
        result_file = os.path.join(app.config['OUTPUT_FOLDER'], f"{file_id}_analysis.json")
        
        if not os.path.exists(result_file):
            return jsonify({'error': '找不到分析結果或分析仍在進行中'}), 404
        
        with open(result_file, 'r', encoding='utf-8') as f:
            results = json.load(f)
        
        return jsonify(results)
    
    except Exception as e:
        print(f"讀取結果失敗: {str(e)}")
        return jsonify({'error': f'讀取結果失敗: {str(e)}'}), 500

@app.route('/api/files', methods=['GET'])
def list_files_for_dropdown():
    """回傳所有已上傳/分析影片的列表，結構符合前端 FetchedResult[] 介面要求。"""
    try:
        batch_id_filter = request.args.get('batch_id')
        metadata = load_metadata()
        file_list_for_frontend = []
        
        for file_id, info in metadata.items():
            
            analysis_file_path = os.path.join(app.config['OUTPUT_FOLDER'], f"{file_id}_analysis.json")
            
            current_status = info.get('status', 'uploaded')
            status_map = {'uploaded': '處理中', 'analyzing': '處理中', 'analyzed': '完成', 'failed': '失敗'}
            frontend_status = status_map.get(current_status, '處理中')
            
            data = None
            error = info.get('error') 
            
            if frontend_status == "完成" and os.path.exists(analysis_file_path):
                try:
                    with open(analysis_file_path, 'r', encoding='utf-8') as f:
                        analysis_data = json.load(f)
                    data = {
                        "summary": analysis_data.get('summary', {}),
                        "shots": analysis_data.get('shots', {}),
                        "speed": analysis_data.get('speed', {})
                    }
                    error = None
                except Exception as e:
                    frontend_status = "失敗" 
                    error = f"讀取結果失敗: {str(e)}"

            display_name = f"{info.get('person_name', '未知球員')} - {info.get('original_filename', file_id)}"
            
            file_list_for_frontend.append({
                'fileId': file_id,
                'name': display_name, # 供前端提取人名並生成連續編號
                'status': frontend_status, 
                'data': data, 
                'error': error, 
                'fileName': info.get('original_filename', file_id), 
            })
            
        return jsonify(file_list_for_frontend)
    
    except Exception as e:
        print(f"獲取檔案列表失敗: {str(e)}")
        traceback.print_exc()
        return jsonify([]), 500

# ------------------------------------------------------------
# 🎯 新增的球員數據儀表板 API 路由
# ------------------------------------------------------------

@app.route('/api/players', methods=['GET'])
def list_unique_players():
    """回傳所有已上傳影片的唯一球員名稱列表 (用於前端下拉選單)。"""
    try:
        metadata = load_metadata()
        unique_names = set()
        
        for info in metadata.values():
            name = info.get('person_name')
            if name:
                unique_names.add(name)
                
        # 由於前端的 Axios 已經處理 URL 編碼/解碼，這裡直接使用原始人名即可
        return jsonify(sorted(list(unique_names)))
    except Exception as e:
        print(f"獲取球員列表失敗: {str(e)}")
        return jsonify([]), 500


@app.route('/api/analysis/player/<person_name>', methods=['GET'])
def get_player_analysis_history(person_name):
    """
    根據 person_name (球員姓名) 獲取該球員所有已分析影片的結果列表。
    """
    try:
        # Flask 會自動處理 URL 解碼，但以防萬一，我們使用 request.path 輔助檢查
        decoded_name = person_name 
        
        metadata = load_metadata()
        player_files = []

        for file_id, info in metadata.items():
            
            # 1. 篩選：只選擇匹配人名的記錄
            if info.get('person_name') != decoded_name:
                continue

            analysis_file_path = os.path.join(app.config['OUTPUT_FOLDER'], f"{file_id}_analysis.json")
            current_status = info.get('status', 'uploaded')
            
            # 2. 獲取結果，只回傳已完成分析的數據
            if current_status == 'analyzed' and os.path.exists(analysis_file_path):
                try:
                    with open(analysis_file_path, 'r', encoding='utf-8') as f:
                        analysis_data = json.load(f)
                    
                    # 格式化為前端所需的單一分析結果結構 (FetchedResult)
                    player_files.append({
                        'fileId': file_id,
                        'name': f"{decoded_name} - {info.get('original_filename', file_id)}",
                        'status': '完成',
                        'data': {
                            "summary": analysis_data.get('summary', {}),
                            "shots": analysis_data.get('shots', {}),
                            "speed": analysis_data.get('speed', {})
                        },
                        'error': None,
                        'fileName': info.get('original_filename', file_id),
                        'uploadTime': info.get('upload_time'), # 新增上傳時間，方便排序
                    })
                except Exception:
                    # 如果分析檔案損壞，則跳過
                    continue

        # 根據上傳時間排序（最新在上）
        player_files.sort(key=lambda x: x.get('uploadTime', '0'), reverse=True)
        
        if not player_files:
            return jsonify({'message': f'找不到 {decoded_name} 的任何分析結果'}), 404

        return jsonify(player_files)
    
    except Exception as e:
        print(f"獲取球員歷史數據失敗: {str(e)}")
        traceback.print_exc()
        return jsonify({'error': f'後端服務器錯誤: {str(e)}'}), 500


# ------------------------------------------------------------
# 🎯 新增的刪除 API 路由 (保持不變)
# ------------------------------------------------------------

@app.route('/api/delete/<file_id>', methods=['DELETE'])
def delete_video(file_id):
    """
    根據 file_id 刪除所有相關檔案和元數據記錄。
    """
    
    # 1. 載入並刪除元數據記錄
    metadata = load_metadata()
    if file_id not in metadata:
        return jsonify({'success': False, 'error': '找不到該檔案的元數據記錄'}), 404
        
    original_info = metadata.pop(file_id)
    save_metadata(metadata)
    
    deleted_files = []
    
    try:
        # 2. 刪除原始影片檔案 (在 uploads)
        found_original = False
        for ext in ALLOWED_EXTENSIONS:
            original_filename = f"{file_id}.{ext}"
            original_path = os.path.join(app.config['UPLOAD_FOLDER'], original_filename)
            if os.path.exists(original_path):
                os.remove(original_path)
                deleted_files.append(f"uploads/{original_filename}")
                found_original = True
                break
        
        # 3. 刪除處理後檔案 (在 output)
        processed_video_filename = f"{file_id}_processed.mp4"
        processed_video_path = os.path.join(app.config['OUTPUT_FOLDER'], processed_video_filename)
        if os.path.exists(processed_video_path):
            os.remove(processed_video_path)
            deleted_files.append(f"output/{processed_video_filename}")
            
        analysis_json_filename = f"{file_id}_analysis.json"
        analysis_json_path = os.path.join(app.config['OUTPUT_FOLDER'], analysis_json_filename)
        if os.path.exists(analysis_json_path):
            os.remove(analysis_json_path)
            deleted_files.append(f"output/{analysis_json_filename}")
            
        print(f"成功刪除檔案 ID: {file_id}. 刪除的文件: {deleted_files}")
        
        return jsonify({
            'success': True, 
            'message': f'檔案 {file_id} 及其所有相關數據已成功刪除。',
            'deleted_files': deleted_files
        })

    except Exception as e:
        print(f"刪除檔案時發生錯誤: {e}")
        traceback.print_exc()
        return jsonify({'success': False, 'error': f'刪除部分文件時發生錯誤，但元數據已清除: {str(e)}'}), 500
        
# ------------------------------------------------------------
# 影片路由 (保持不變)
# ------------------------------------------------------------

@app.route('/api/video/<file_id>', methods=['GET'])
def get_original_video(file_id):
    """獲取原始影片"""
    try:
        filename_with_ext = None
        for ext in ALLOWED_EXTENSIONS:
            potential_filename = f"{file_id}.{ext}"
            potential_path = os.path.join(app.config['UPLOAD_FOLDER'], potential_filename)
            if os.path.exists(potential_path):
                filename_with_ext = potential_filename
                break

        if not filename_with_ext:
            print(f"找不到原始影片: {file_id}")
            return jsonify({'error': '找不到原始影片'}), 404
        
        return send_from_directory(
            app.config['UPLOAD_FOLDER'], 
            filename_with_ext, 
            mimetype='video/*', 
            conditional=True 
        )
        
    except Exception as e:
        print(f"讀取原始影片失敗 (500): {str(e)}")
        traceback.print_exc()
        return jsonify({'error': f'讀取原始影片失敗: {str(e)}'}), 500


@app.route('/api/processed-video/<file_id>', methods=['GET'])
def get_processed_video(file_id):
    """獲取處理後的影片"""
    try:
        processed_filename = f"{file_id}_processed.mp4"
        processed_file_path = os.path.join(app.config['OUTPUT_FOLDER'], processed_filename)
        
        if not os.path.exists(processed_file_path):
            print(f"找不到處理後影片: {processed_file_path}")
            return jsonify({'error': '找不到處理後的影片'}), 404

        return send_from_directory(
            app.config['OUTPUT_FOLDER'], 
            processed_filename, 
            mimetype='video/mp4', 
            conditional=True
        )
        
    except Exception as e:
        print(f"讀取處理後影片失敗 (500): {str(e)}")
        traceback.print_exc()
        return jsonify({'error': f'讀取處理後影片失敗: {str(e)}'}), 500


# ------------------------------------------------------------
# 🎯 3D 姿態重建 API 路由
# ------------------------------------------------------------

@app.route('/api/upload-multiview', methods=['POST'])
def upload_multiview():
    """
    上傳多視角影片用於 3D 重建
    需要上傳多個影片檔案，每個視角一個
    """
    if not MULTIVIEW_3D_AVAILABLE:
        return jsonify({'success': False, 'error': '3D 重建模組不可用'}), 503

    try:
        # 檢查是否有檔案上傳
        if not request.files:
            return jsonify({'success': False, 'error': '沒有上傳任何檔案'}), 400

        # 獲取球員名稱
        person_name = request.form.get('person_name', '未命名球員')

        # 獲取每個視角的相機配置 (角度、距離、高度)
        camera_configs_str = request.form.get('camera_configs', '{}')
        try:
            camera_configs = json.loads(camera_configs_str)
        except json.JSONDecodeError:
            camera_configs = {}

        # 預設配置
        default_configs = {
            'front': {'angle': 0, 'distance': 5.0, 'height': 1.5},
            'right': {'angle': 90, 'distance': 5.0, 'height': 1.5},
            'back': {'angle': 180, 'distance': 5.0, 'height': 1.5},
            'left': {'angle': 270, 'distance': 5.0, 'height': 1.5},
        }

        # 合併自定義配置與預設配置
        for view_name in default_configs:
            if view_name not in camera_configs:
                camera_configs[view_name] = default_configs[view_name]

        # 生成唯一 ID
        batch_id = str(uuid.uuid4())

        # 儲存影片並收集路徑
        video_paths = {}
        view_names = ['front', 'right', 'back', 'left']

        for view_name in view_names:
            file_key = f'video_{view_name}'
            if file_key in request.files:
                file = request.files[file_key]
                if file and file.filename and allowed_file(file.filename):
                    filename = secure_filename(file.filename)
                    ext = filename.rsplit('.', 1)[1].lower()
                    save_filename = f"{batch_id}_{view_name}.{ext}"
                    save_path = os.path.join(app.config['UPLOAD_FOLDER'], save_filename)
                    file.save(save_path)
                    video_paths[view_name] = save_path
                    print(f"📷 已儲存 {view_name} 視角: {save_filename}")

        if len(video_paths) < 2:
            return jsonify({
                'success': False,
                'error': '至少需要上傳 2 個視角的影片 (建議 4 個)'
            }), 400

        # 儲存元數據
        metadata = load_metadata()
        metadata[batch_id] = {
            'type': '3d_multiview',
            'person_name': person_name,
            'views': list(video_paths.keys()),
            'camera_configs': camera_configs,  # 每個視角的完整配置 (角度、距離、高度)
            'upload_time': datetime.now().isoformat(),
            'status': 'uploaded'
        }
        save_metadata(metadata)

        return jsonify({
            'success': True,
            'batch_id': batch_id,
            'views_uploaded': list(video_paths.keys()),
            'num_views': len(video_paths),
            'person_name': person_name
        })

    except Exception as e:
        print(f"多視角上傳錯誤: {str(e)}")
        traceback.print_exc()
        return jsonify({'success': False, 'error': f'上傳失敗: {str(e)}'}), 500


@app.route('/api/analyze-3d/<batch_id>', methods=['POST'])
def analyze_3d(batch_id):
    """
    執行 3D 姿態重建分析
    """
    if not MULTIVIEW_3D_AVAILABLE:
        return jsonify({'success': False, 'error': '3D 重建模組不可用'}), 503

    try:
        # 獲取元數據
        metadata = load_metadata()
        if batch_id not in metadata:
            return jsonify({'success': False, 'error': '找不到該批次的上傳記錄'}), 404

        batch_info = metadata[batch_id]
        if batch_info.get('type') != '3d_multiview':
            return jsonify({'success': False, 'error': '此 ID 不是多視角上傳'}), 400

        # 更新狀態
        metadata[batch_id]['status'] = 'analyzing'
        save_metadata(metadata)

        print(f"🎬 開始 3D 分析: {batch_id}")

        # 收集影片路徑
        video_paths = {}
        for view_name in batch_info.get('views', []):
            for ext in ALLOWED_EXTENSIONS:
                potential_path = os.path.join(
                    app.config['UPLOAD_FOLDER'],
                    f"{batch_id}_{view_name}.{ext}"
                )
                if os.path.exists(potential_path):
                    video_paths[view_name] = potential_path
                    break

        if len(video_paths) < 2:
            raise ValueError("找不到足夠的影片檔案")

        # 建立處理器並處理多視角
        processor = MultiViewProcessor(pose_model_size=POSE_MODEL_SIZE)

        # 獲取每個視角的相機配置
        camera_configs = batch_info.get('camera_configs', {})

        for view_name, video_path in video_paths.items():
            # 使用每個視角的自定義配置
            view_config = camera_configs.get(view_name, {})
            processor.add_camera(
                view_name=view_name,
                video_path=video_path,
                angle_degrees=view_config.get('angle'),  # 自定義角度
                distance=view_config.get('distance', 5.0),  # 自定義距離
                height=view_config.get('height', 1.5)  # 自定義高度
            )

        # 處理多視角
        multiview_result = processor.process_multiview()

        # 3D 重建
        result_3d = reconstruct_3d_from_multiview(multiview_result, normalize=True)

        # 整合結果
        analysis_results = {
            'batch_id': batch_id,
            'timestamp': datetime.now().isoformat(),
            'type': '3d_reconstruction',
            'video_info': result_3d.get('video_info', {}),
            'camera_configs': result_3d.get('camera_configs', []),
            'poses_3d': result_3d.get('poses_3d', []),
            'skeleton_connections': result_3d.get('skeleton_connections', []),
            'keypoint_names': result_3d.get('keypoint_names', []),
            'statistics': result_3d.get('statistics', {})
        }

        # 儲存結果
        result_file = os.path.join(
            app.config['OUTPUT_FOLDER'],
            f"{batch_id}_3d_analysis.json"
        )
        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump(analysis_results, f, ensure_ascii=False, indent=2)

        # 更新元數據
        metadata[batch_id]['status'] = 'analyzed'
        metadata[batch_id]['statistics'] = result_3d.get('statistics', {})
        save_metadata(metadata)

        print(f"✅ 3D 分析完成: {batch_id}")

        return jsonify({
            'success': True,
            'message': '3D 分析完成',
            'batch_id': batch_id,
            'statistics': result_3d.get('statistics', {})
        })

    except Exception as e:
        print(f"3D 分析錯誤: {str(e)}")
        traceback.print_exc()

        # 更新元數據狀態
        metadata = load_metadata()
        if batch_id in metadata:
            metadata[batch_id]['status'] = 'failed'
            metadata[batch_id]['error'] = str(e)
            save_metadata(metadata)

        return jsonify({'success': False, 'error': f'3D 分析失敗: {str(e)}'}), 500


@app.route('/api/results-3d/<batch_id>', methods=['GET'])
def get_3d_results(batch_id):
    """獲取 3D 分析結果"""
    try:
        result_file = os.path.join(
            app.config['OUTPUT_FOLDER'],
            f"{batch_id}_3d_analysis.json"
        )

        if not os.path.exists(result_file):
            return jsonify({'error': '找不到 3D 分析結果'}), 404

        with open(result_file, 'r', encoding='utf-8') as f:
            results = json.load(f)

        return jsonify(results)

    except Exception as e:
        print(f"讀取 3D 結果失敗: {str(e)}")
        return jsonify({'error': f'讀取結果失敗: {str(e)}'}), 500


@app.route('/api/3d-status', methods=['GET'])
def get_3d_status():
    """檢查 3D 重建功能狀態"""
    return jsonify({
        'available': MULTIVIEW_3D_AVAILABLE,
        'pose_model_size': POSE_MODEL_SIZE if MULTIVIEW_3D_AVAILABLE else None,
        'message': '3D 重建功能可用' if MULTIVIEW_3D_AVAILABLE else '3D 重建模組未載入'
    })


@app.route('/api/3d-analyses', methods=['GET'])
def list_3d_analyses():
    """列出所有 3D 分析記錄"""
    try:
        metadata = load_metadata()
        analyses = []

        for batch_id, info in metadata.items():
            # 只返回 3D 多視角類型的記錄
            if info.get('type') == '3d_multiview':
                analyses.append({
                    'batch_id': batch_id,
                    'person_name': info.get('person_name', '未知'),
                    'views': info.get('views', []),
                    'upload_time': info.get('upload_time', ''),
                    'status': info.get('status', 'unknown'),
                    'statistics': info.get('statistics', {})
                })

        # 按上傳時間排序（最新在前）
        analyses.sort(key=lambda x: x.get('upload_time', ''), reverse=True)

        return jsonify({
            'success': True,
            'analyses': analyses,
            'total': len(analyses)
        })

    except Exception as e:
        print(f"列出 3D 分析記錄失敗: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500


# ------------------------------------------------------------
# 🎯 PoseAug 姿態增強 API 路由
# ------------------------------------------------------------

@app.route('/api/poseaug-status', methods=['GET'])
def get_poseaug_status():
    """檢查 PoseAug 功能狀態"""
    gan_status = poseaug_gan.get_status() if POSEAUG_GAN_AVAILABLE and poseaug_gan else {'available': False}
    return jsonify({
        'available': POSEAUG_AVAILABLE,
        'message': 'PoseAug 姿態增強功能可用' if POSEAUG_AVAILABLE else 'PoseAug 模組未載入',
        'keypoint_names': KEYPOINT_NAMES if POSEAUG_AVAILABLE else [],
        'skeleton_bones': SKELETON_BONES if POSEAUG_AVAILABLE else [],
        'gan_available': POSEAUG_GAN_AVAILABLE,
        'gan_status': gan_status,
        'videopose3d_available': VIDEOPOSE3D_AVAILABLE and videopose3d_lifter is not None and videopose3d_lifter.weights_loaded,
        'videopose3d_status': videopose3d_lifter.get_status() if VIDEOPOSE3D_AVAILABLE and videopose3d_lifter else {'available': False},
    })


@app.route('/api/poseaug/sample-pose', methods=['GET'])
def get_sample_pose():
    """獲取範例姿態數據用於測試"""
    if not POSEAUG_AVAILABLE:
        return jsonify({'success': False, 'error': 'PoseAug 模組不可用'}), 503

    try:
        sample_pose = create_sample_pose()
        return jsonify({
            'success': True,
            'pose': sample_pose.tolist(),
            'keypoint_names': KEYPOINT_NAMES,
            'skeleton_bones': SKELETON_BONES
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/poseaug/augmentation-presets', methods=['GET'])
def get_augmentation_presets():
    """獲取增強預設配置"""
    if not POSEAUG_AVAILABLE:
        return jsonify({'success': False, 'error': 'PoseAug 模組不可用'}), 503

    presets = {
        'conservative': {
            'name': '保守 (Conservative)',
            'description': '輕微增強，保持姿態自然',
            'config': {
                'bl_scale_range': (0.95, 1.05),
                'ba_angle_range': (-5, 5),
                'rt_angle_range': (-10, 10),
                'pos_shift_range': (-0.02, 0.02)
            }
        },
        'moderate': {
            'name': '適中 (Moderate)',
            'description': '中等增強，適合一般訓練',
            'config': {
                'bl_scale_range': (0.9, 1.1),
                'ba_angle_range': (-15, 15),
                'rt_angle_range': (-30, 30),
                'pos_shift_range': (-0.05, 0.05)
            }
        },
        'aggressive': {
            'name': '激進 (Aggressive)',
            'description': '強力增強，增加數據多樣性',
            'config': {
                'bl_scale_range': (0.8, 1.2),
                'ba_angle_range': (-25, 25),
                'rt_angle_range': (-45, 45),
                'pos_shift_range': (-0.1, 0.1)
            }
        }
    }

    return jsonify({
        'success': True,
        'presets': presets,
        'augmentation_types': ['BL', 'BA', 'RT', 'POS'],
        'augmentation_descriptions': {
            'BL': '骨骼長度增強 (Bone Length)',
            'BA': '骨骼角度增強 (Bone Angle)',
            'RT': '旋轉增強 (Rotation)',
            'POS': '位置偏移增強 (Position)'
        }
    })


@app.route('/api/poseaug/apply', methods=['POST'])
def apply_augmentation():
    """
    應用姿態增強
    請求 body:
    {
        "pose": [[x, y, z], ...],  // 17 個關鍵點的 3D 座標
        "augmentation_types": ["BL", "BA", "RT", "POS"],  // 要應用的增強類型
        "config": {  // 可選，自定義配置
            "bl_scale_range": [0.9, 1.1],
            "ba_angle_range": [-15, 15],
            "rt_angle_range": [-30, 30],
            "pos_shift_range": [-0.05, 0.05]
        },
        "num_augmentations": 5  // 要生成的增強數量
    }
    """
    if not POSEAUG_AVAILABLE:
        return jsonify({'success': False, 'error': 'PoseAug 模組不可用'}), 503

    try:
        data = request.get_json()
        if not data:
            return jsonify({'success': False, 'error': '缺少請求數據'}), 400

        # 獲取原始姿態
        pose_data = data.get('pose')
        if pose_data is None:
            # 如果沒有提供姿態，使用範例姿態
            pose = create_sample_pose()
        else:
            pose = np.array(pose_data, dtype=np.float32)

        # 驗證姿態
        validation_result = validate_pose(pose)
        if not validation_result.get('valid', False):
            issues = validation_result.get('issues', ['未知錯誤'])
            return jsonify({'success': False, 'error': f'姿態驗證失敗: {", ".join(issues)}'}), 400

        # 獲取增強類型
        aug_types = data.get('augmentation_types', ['BL', 'BA', 'RT', 'POS'])
        num_augmentations = data.get('num_augmentations', 5)

        # 獲取配置
        config_data = data.get('config', {})

        # 根據選擇的增強類型設定配置
        config = AugmentationConfig(
            bone_length_enabled='BL' in aug_types,
            bone_angle_enabled='BA' in aug_types,
            rotation_enabled='RT' in aug_types,
            translation_enabled='POS' in aug_types,
            bone_length_scale_range=tuple(config_data.get('bl_scale_range', (0.9, 1.1))),
            bone_angle_range=abs(config_data.get('ba_angle_range', [-15, 15])[1]),
            rotation_range=(
                abs(config_data.get('rt_angle_range', [-30, 30])[1]),
                abs(config_data.get('rt_angle_range', [-30, 30])[1]),
                15.0
            ),
            translation_range=abs(config_data.get('pos_shift_range', [-0.05, 0.05])[1])
        )

        # 創建增強器
        augmentor = PoseAugmentor(config)

        # 生成增強姿態
        augmented_poses = []
        augmentation_details = []

        for i in range(num_augmentations):
            # 使用 augment_pose 方法進行增強
            aug_pose = augmentor.augment_pose(pose)

            augmented_poses.append(aug_pose.tolist())
            augmentation_details.append({
                'index': i,
                'applied_augmentations': aug_types.copy()
            })

        return jsonify({
            'success': True,
            'original_pose': pose.tolist(),
            'augmented_poses': augmented_poses,
            'augmentation_details': augmentation_details,
            'num_generated': len(augmented_poses),
            'keypoint_names': KEYPOINT_NAMES,
            'skeleton_bones': SKELETON_BONES
        })

    except Exception as e:
        print(f"姿態增強錯誤: {str(e)}")
        traceback.print_exc()
        return jsonify({'success': False, 'error': f'增強失敗: {str(e)}'}), 500


@app.route('/api/poseaug/extract-from-video', methods=['POST'])
def extract_pose_from_video():
    """
    從單一影片提取姿態並轉換為 3D 座標
    使用 2D 姿態估計 + 深度估計生成偽 3D 座標
    """
    if not POSEAUG_AVAILABLE:
        return jsonify({'success': False, 'error': 'PoseAug 模組不可用'}), 503

    try:
        # 檢查是否有影片上傳
        if 'video' not in request.files:
            return jsonify({'success': False, 'error': '沒有上傳影片'}), 400

        video_file = request.files['video']
        if not video_file or not video_file.filename:
            return jsonify({'success': False, 'error': '無效的影片檔案'}), 400

        # 獲取參數
        max_frames = int(request.form.get('max_frames', 30))  # 最多提取幾幀
        frame_skip = int(request.form.get('frame_skip', 5))   # 每隔幾幀提取一次

        # 儲存臨時檔案
        temp_id = str(uuid.uuid4())
        filename = secure_filename(video_file.filename)
        ext = filename.rsplit('.', 1)[1].lower() if '.' in filename else 'mp4'
        temp_path = os.path.join(app.config['UPLOAD_FOLDER'], f"temp_poseaug_{temp_id}.{ext}")
        video_file.save(temp_path)

        # 使用 pose_detector 提取 2D 姿態
        from pose_detector import PoseDetector
        detector = PoseDetector(model_size=POSE_MODEL_SIZE)

        cap = None
        extracted_poses = []
        fps = 30
        total_frames = 0

        try:
            cap = cv2.VideoCapture(temp_path)
            fps = cap.get(cv2.CAP_PROP_FPS) or 30
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            frame_count = 0
            processed_count = 0

            while cap.isOpened() and processed_count < max_frames:
                ret, frame = cap.read()
                if not ret:
                    break

                if frame_count % frame_skip == 0:
                    # 檢測姿態
                    poses = detector.detect_pose(frame)

                    if poses and len(poses) > 0:
                        # 取第一個檢測到的人
                        main_pose = poses[0]
                        keypoints_raw = main_pose.get('keypoints_raw')

                        if keypoints_raw is not None:
                            pose_2d = np.array(keypoints_raw)[:, :2]
                            confidence = np.array(keypoints_raw)[:, 2] if keypoints_raw.shape[1] > 2 else np.ones(17)

                            # 估計 3D 座標
                            h, w = frame.shape[:2]
                            if VIDEOPOSE3D_AVAILABLE and videopose3d_lifter and videopose3d_lifter.weights_loaded:
                                pose_3d = videopose3d_lifter.lift(pose_2d, frame_width=w, frame_height=h, confidence=confidence)
                                lifting_method = 'VideoPose3D'
                            else:
                                pose_3d = estimate_3d_from_2d(pose_2d, confidence, frame.shape)
                                lifting_method = 'heuristic'

                            # YOLO 網球拍偵測
                            hand_info = None
                            if RACKET_DETECT_AVAILABLE and racket_detector:
                                try:
                                    racket_data = racket_detector.detect(frame)
                                    if racket_data:
                                        # 判斷球拍最接近哪隻手
                                        rc = np.array(racket_data['center_2d'])
                                        dist_r = np.linalg.norm(rc - pose_2d[10])
                                        dist_l = np.linalg.norm(rc - pose_2d[9])
                                        matched = 'right' if dist_r < dist_l else 'left'

                                        if matched == 'right':
                                            w3d, e3d = pose_3d[10], pose_3d[8]
                                            w2d = pose_2d[10]
                                        else:
                                            w3d, e3d = pose_3d[9], pose_3d[7]
                                            w2d = pose_2d[9]

                                        vectors_3d = compute_racket_3d_from_detection(
                                            racket_data, w2d, w3d, e3d, w, h, 0.002
                                        )
                                        hand_info = {
                                            'matched_hand': matched,
                                            'racket_dir_3d': vectors_3d['racket_dir_3d'],
                                            'palm_normal_3d': vectors_3d['palm_normal_3d'],
                                            'racket_bbox': racket_data['bbox'],
                                            'racket_conf': racket_data['confidence'],
                                        }
                                except Exception as re_err:
                                    if processed_count < 3:
                                        print(f"  [Racket ERROR] frame={frame_count}: {re_err}")

                            # 網球球體偵測
                            ball_info = None
                            try:
                                ball_detections = tennis_tracker.detect_tennis_ball(frame)
                                if ball_detections:
                                    best_ball = max(ball_detections, key=lambda d: d['confidence'])
                                    ball_info = {
                                        'center_2d': list(best_ball['center']),
                                        'bbox': list(best_ball['bbox']),
                                        'confidence': best_ball['confidence'],
                                        'size': list(best_ball['size']),
                                    }
                            except Exception:
                                pass

                            extracted_poses.append({
                                'frame_number': frame_count,
                                'timestamp': frame_count / fps if fps > 0 else 0,
                                'pose_2d': pose_2d.tolist(),
                                'pose_3d': pose_3d.tolist(),
                                'confidence': confidence.tolist(),
                                'lifting_method': lifting_method,
                                'hand_info': hand_info,
                                'ball_info': ball_info,
                            })

                    processed_count += 1
                frame_count += 1

        finally:
            # 確保釋放視頻資源
            if cap is not None:
                cap.release()
            # 強制垃圾回收以釋放檔案句柄
            import gc
            gc.collect()

        # 清理臨時檔案 (在 finally 之外，確保 cap 已釋放)
        import time
        for attempt in range(5):
            try:
                if os.path.exists(temp_path):
                    os.remove(temp_path)
                break
            except PermissionError:
                time.sleep(0.5)  # 等待檔案釋放

        if not extracted_poses:
            return jsonify({
                'success': False,
                'error': '無法從影片中檢測到姿態，請確保影片中有清晰可見的人物'
            }), 400

        # ===== 計算全局軌跡 =====
        # 用 2D 骨盆中心的像素位移 + 3D/2D 軀幹比例因子推算每幀的全局位置
        if len(extracted_poses) > 0:
            # 1. 收集每幀的 2D 骨盆中心
            pelvis_2d_list = []
            for ep in extracted_poses:
                p2d = np.array(ep['pose_2d'])
                pelvis = (p2d[11] + p2d[12]) / 2.0  # (left_hip + right_hip) / 2
                pelvis_2d_list.append(pelvis)

            # 2. 計算比例因子: 3D 軀幹長度 / 2D 軀幹長度 (像素)
            #    用第一幀作為參考
            p2d_ref = np.array(extracted_poses[0]['pose_2d'])
            p3d_ref = np.array(extracted_poses[0]['pose_3d'])

            # 2D 軀幹: 肩膀中心到臀部中心的像素距離
            shoulder_2d = (p2d_ref[5] + p2d_ref[6]) / 2.0
            hip_2d = (p2d_ref[11] + p2d_ref[12]) / 2.0
            torso_2d_len = np.linalg.norm(shoulder_2d - hip_2d)

            # 3D 軀幹: 肩膀中心到臀部中心的 3D 距離
            shoulder_3d = (p3d_ref[5] + np.array(p3d_ref[6])) / 2.0
            hip_3d = (np.array(p3d_ref[11]) + np.array(p3d_ref[12])) / 2.0
            torso_3d_len = np.linalg.norm(np.array(shoulder_3d) - np.array(hip_3d))

            # 比例: 每像素對應多少 3D 單位
            if torso_2d_len > 10:  # 避免除以零
                pixel_to_3d_scale = torso_3d_len / torso_2d_len
            else:
                pixel_to_3d_scale = 0.002  # 預設值

            # 3. 地面校正: 每幀直接把最低點移到 Y=0
            #    VideoPose3D 輸出是 root-relative (骨盆≈0)，腿部 Y 為負值
            #    對每幀，將所有關鍵點上移，讓腳踝剛好在 Y=0
            for ep in extracted_poses:
                p3d = np.array(ep['pose_3d'])
                y_min = p3d[:, 1].min()  # 該幀最低點
                p3d[:, 1] -= y_min       # 整體上移，最低點=0
                ep['pose_3d'] = p3d.tolist()

            # 4. 以第一幀為原點，計算相對位移 + 球體 3D 位置
            ref_pelvis = pelvis_2d_list[0]
            for i, ep in enumerate(extracted_poses):
                dx_pixels = pelvis_2d_list[i] - ref_pelvis
                root_x = -dx_pixels[0] * pixel_to_3d_scale
                root_z = -dx_pixels[1] * pixel_to_3d_scale
                root_y = 0.0
                ep['root_position'] = [float(root_x), float(root_y), float(root_z)]

                # 球體 2D → 3D 位置 (相對於第一幀的骨盆)
                if ep.get('ball_info'):
                    bc = np.array(ep['ball_info']['center_2d'])
                    ball_dx = -(bc[0] - ref_pelvis[0]) * pixel_to_3d_scale
                    ball_dz = -(bc[1] - ref_pelvis[1]) * pixel_to_3d_scale
                    # 球的高度: 用 2D Y 座標相對於腳踝估算
                    # 腳踝是畫面最低的身體部位，球在腳踝上方的像素距離 → 3D 高度
                    p2d = np.array(ep['pose_2d'])
                    ankle_y = max(p2d[15][1], p2d[16][1])  # 取較低的腳踝
                    ball_height_px = ankle_y - bc[1]  # 像素中球在腳踝上方多少
                    ball_y = max(0.0, ball_height_px * pixel_to_3d_scale)
                    ep['ball_info']['position_3d'] = [float(ball_dx), float(ball_y), float(ball_dz)]

        # 判斷使用的 3D 提升方法
        methods_used = set(p.get('lifting_method', 'heuristic') for p in extracted_poses)

        return jsonify({
            'success': True,
            'video_info': {
                'fps': fps,
                'total_frames': total_frames,
                'extracted_frames': len(extracted_poses)
            },
            'lifting_method': list(methods_used)[0] if len(methods_used) == 1 else 'mixed',
            'poses': extracted_poses,
            'keypoint_names': KEYPOINT_NAMES,
            'skeleton_bones': SKELETON_BONES
        })

    except Exception as e:
        print(f"影片姿態提取錯誤: {str(e)}")
        traceback.print_exc()
        return jsonify({'success': False, 'error': f'處理失敗: {str(e)}'}), 500


def estimate_3d_from_2d(pose_2d: np.ndarray, confidence: np.ndarray, frame_shape: tuple) -> np.ndarray:
    """
    從 2D 姿態估計偽 3D 座標
    使用骨骼結構和比例關係估計深度

    Args:
        pose_2d: [17, 2] 的 2D 關鍵點座標 (像素)
        confidence: [17] 的置信度
        frame_shape: (height, width, channels)

    Returns:
        pose_3d: [17, 3] 的歸一化 3D 座標
    """
    height, width = frame_shape[:2]

    # 歸一化 2D 座標到 [-1, 1] 範圍
    pose_2d_norm = pose_2d.copy().astype(np.float32)
    pose_2d_norm[:, 0] = (pose_2d_norm[:, 0] / width) * 2 - 1   # x: [-1, 1]
    pose_2d_norm[:, 1] = -((pose_2d_norm[:, 1] / height) * 2 - 1)  # y: [-1, 1], 翻轉 y 軸

    # 計算骨骼長度用於深度估計
    # 軀幹參考: 肩膀到臀部的距離
    left_shoulder = pose_2d_norm[5]
    right_shoulder = pose_2d_norm[6]
    left_hip = pose_2d_norm[11]
    right_hip = pose_2d_norm[12]

    shoulder_center = (left_shoulder + right_shoulder) / 2
    hip_center = (left_hip + right_hip) / 2
    torso_length = np.linalg.norm(shoulder_center - hip_center)

    # 參考軀幹長度 (假設標準軀幹長度為 0.5 單位)
    reference_torso = 0.5
    scale = reference_torso / (torso_length + 1e-6)

    # 初始化 3D 座標
    pose_3d = np.zeros((17, 3), dtype=np.float32)
    pose_3d[:, :2] = pose_2d_norm * scale

    # 估計深度 (z 座標)
    # 基於身體部位的相對深度關係
    # 軀幹中心為 z=0
    body_center = (shoulder_center + hip_center) / 2

    # 深度估計規則:
    # - 肩膀和臀部在同一平面 (z ≈ 0)
    # - 手肘和膝蓋稍微向前 (z > 0) 或向後 (z < 0) 根據角度
    # - 手腕和腳踝可能更向前或向後

    # 簡化的深度分配
    depth_map = {
        0: 0.1,    # nose (slightly forward)
        1: 0.05,   # left_eye
        2: 0.05,   # right_eye
        3: 0.0,    # left_ear
        4: 0.0,    # right_ear
        5: 0.0,    # left_shoulder
        6: 0.0,    # right_shoulder
        7: 0.05,   # left_elbow
        8: 0.05,   # right_elbow
        9: 0.1,    # left_wrist
        10: 0.1,   # right_wrist
        11: 0.0,   # left_hip
        12: 0.0,   # right_hip
        13: 0.05,  # left_knee
        14: 0.05,  # right_knee
        15: 0.0,   # left_ankle
        16: 0.0,   # right_ankle
    }

    for i in range(17):
        pose_3d[i, 2] = depth_map.get(i, 0.0)

    # 根據置信度調整 (低置信度的點深度設為 0)
    for i in range(17):
        if confidence[i] < 0.3:
            pose_3d[i, 2] = 0.0

    # 將整體移動到原點附近
    pose_3d -= pose_3d.mean(axis=0)

    return pose_3d


@app.route('/api/poseaug/rotation-series', methods=['POST'])
def generate_rotation_series():
    """
    生成旋轉系列 - 專門用於 RT 增強
    可以生成 360° 環繞視角或自定義角度序列

    請求 body:
    {
        "pose": [[x, y, z], ...],  // 17 個關鍵點
        "mode": "full_360" | "custom" | "interval",
        "axis": "y" | "x" | "z" | "all",  // 旋轉軸
        "interval_degrees": 30,  // interval 模式下每隔多少度
        "custom_angles": [0, 45, 90, 135, 180],  // custom 模式下的角度列表
        "include_tilt": true  // 是否包含前傾後傾
    }
    """
    if not POSEAUG_AVAILABLE:
        return jsonify({'success': False, 'error': 'PoseAug 模組不可用'}), 503

    try:
        data = request.get_json()
        if not data:
            return jsonify({'success': False, 'error': '缺少請求數據'}), 400

        pose_data = data.get('pose')
        if pose_data is None:
            pose = create_sample_pose()
        else:
            pose = np.array(pose_data, dtype=np.float32)

        mode = data.get('mode', 'interval')
        axis = data.get('axis', 'y')
        interval = data.get('interval_degrees', 30)
        custom_angles = data.get('custom_angles', [])
        include_tilt = data.get('include_tilt', False)

        # 根據模式生成角度列表
        if mode == 'full_360':
            # 完整 360 度，每 30 度一個
            angles = list(range(0, 360, interval))
        elif mode == 'custom':
            angles = custom_angles
        else:  # interval
            angles = list(range(0, 360, interval))

        # 創建增強器 (只啟用旋轉)
        config = AugmentationConfig(
            bone_length_enabled=False,
            bone_angle_enabled=False,
            rotation_enabled=True,
            translation_enabled=False
        )
        augmentor = PoseAugmentor(config)

        # 生成旋轉序列
        rotated_poses = []

        for angle in angles:
            # 根據軸向設定旋轉
            if axis == 'y':
                rotation = (0, angle, 0)
            elif axis == 'x':
                rotation = (angle, 0, 0)
            elif axis == 'z':
                rotation = (0, 0, angle)
            else:  # all - 組合旋轉
                rotation = (angle * 0.3, angle, angle * 0.1)

            # 應用旋轉
            rotated = augmentor._augment_rotation(pose, rotation)

            rotated_poses.append({
                'angle': angle,
                'axis': axis,
                'rotation': rotation,
                'pose': rotated.tolist()
            })

        # 如果需要包含前傾後傾
        if include_tilt:
            tilt_angles = [-30, -15, 15, 30]
            for tilt in tilt_angles:
                rotation = (tilt, 0, 0)  # X 軸旋轉 = 前傾/後傾
                rotated = augmentor._augment_rotation(pose, rotation)
                rotated_poses.append({
                    'angle': tilt,
                    'axis': 'x_tilt',
                    'rotation': rotation,
                    'pose': rotated.tolist()
                })

        return jsonify({
            'success': True,
            'mode': mode,
            'axis': axis,
            'total_rotations': len(rotated_poses),
            'angles_generated': angles,
            'rotated_poses': rotated_poses,
            'original_pose': pose.tolist()
        })

    except Exception as e:
        print(f"旋轉系列生成錯誤: {str(e)}")
        traceback.print_exc()
        return jsonify({'success': False, 'error': f'生成失敗: {str(e)}'}), 500


@app.route('/api/poseaug/gan-status', methods=['GET'])
def poseaug_gan_status():
    """查詢 PoseAug GAN 模組狀態"""
    if not POSEAUG_GAN_AVAILABLE or poseaug_gan is None:
        return jsonify({
            'available': False,
            'message': 'PoseAug GAN 模組不可用'
        })
    return jsonify({
        'available': True,
        **poseaug_gan.get_status()
    })


@app.route('/api/poseaug/gan-augment', methods=['POST'])
def poseaug_gan_augment():
    """
    使用原始 PoseAug GAN 模型進行姿態增強

    請求 body:
    {
        "pose": [[x, y, z], ...],        // COCO 17 關鍵點的 3D 座標
        "num_augmentations": 10,          // 增強數量
        "return_intermediate": false,     // 是否返回 BA/BL 中間結果
        "include_rt": false,             // 是否包含 RT 旋轉 (默認關閉)
        "rt_max_angle": 15.0             // RT 最大旋轉角度
    }
    """
    if not POSEAUG_GAN_AVAILABLE or poseaug_gan is None:
        return jsonify({'success': False, 'error': 'PoseAug GAN 模組不可用'}), 503

    try:
        data = request.get_json()
        if not data:
            return jsonify({'success': False, 'error': '缺少請求數據'}), 400

        pose_data = data.get('pose')
        if pose_data is None:
            return jsonify({'success': False, 'error': '缺少 pose 數據'}), 400

        pose = np.array(pose_data, dtype=np.float32)
        if pose.shape != (17, 3):
            return jsonify({'success': False, 'error': f'姿態格式錯誤: 期望 (17,3), 得到 {pose.shape}'}), 400

        num_augmentations = data.get('num_augmentations', 10)
        return_intermediate = data.get('return_intermediate', False)
        include_rt = data.get('include_rt', False)
        rt_max_angle = data.get('rt_max_angle', 15.0)

        result = poseaug_gan.augment(
            pose,
            num_augmentations=num_augmentations,
            return_intermediate=return_intermediate,
            include_rt=include_rt,
            rt_max_angle=rt_max_angle,
        )

        return jsonify({
            'success': True,
            'method': 'PoseAug GAN (CVPR 2021)',
            'weights_loaded': result['weights_loaded'],
            'original_pose': pose.tolist(),
            'augmented_poses': result['augmented_poses'],
            'augmentation_details': result['augmentation_details'],
            'original_h36m': result['original_h36m'],
            **(
                {'intermediate': result['intermediate']} if return_intermediate and 'intermediate' in result else {}
            ),
        })

    except Exception as e:
        print(f"GAN 增強錯誤: {str(e)}")
        traceback.print_exc()
        return jsonify({'success': False, 'error': f'GAN 增強失敗: {str(e)}'}), 500


@app.route('/api/poseaug/gan-batch-augment', methods=['POST'])
def poseaug_gan_batch_augment():
    """
    使用 PoseAug GAN 批量增強多個姿態

    請求 body:
    {
        "poses": [[[x,y,z],...], ...],   // 多個 COCO 17 姿態
        "num_augmentations_per_pose": 5
    }
    """
    if not POSEAUG_GAN_AVAILABLE or poseaug_gan is None:
        return jsonify({'success': False, 'error': 'PoseAug GAN 模組不可用'}), 503

    try:
        data = request.get_json()
        if not data:
            return jsonify({'success': False, 'error': '缺少請求數據'}), 400

        poses_data = data.get('poses', [])
        if not poses_data:
            return jsonify({'success': False, 'error': '缺少 poses 數據'}), 400

        num_aug = data.get('num_augmentations_per_pose', 5)
        include_rt = data.get('include_rt', False)
        rt_max_angle = data.get('rt_max_angle', 15.0)

        poses = [np.array(p, dtype=np.float32) for p in poses_data]

        # 批量增強，傳遞 RT 選項
        results = []
        for pose in poses:
            r = poseaug_gan.augment(pose, num_augmentations=num_aug, include_rt=include_rt, rt_max_angle=rt_max_angle)
            results.append(r)

        augmented_frames = []
        for i, result in enumerate(results):
            augmented_frames.append({
                'frame_index': i,
                'original_pose': poses[i].tolist(),
                'augmented_poses': result['augmented_poses'],
                'augmentation_details': result['augmentation_details'],
            })

        return jsonify({
            'success': True,
            'method': 'PoseAug GAN (CVPR 2021)',
            'total_frames': len(augmented_frames),
            'augmentations_per_frame': num_aug,
            'total_augmented': len(augmented_frames) * num_aug,
            'augmented_frames': augmented_frames,
        })

    except Exception as e:
        print(f"GAN 批量增強錯誤: {str(e)}")
        traceback.print_exc()
        return jsonify({'success': False, 'error': f'GAN 批量增強失敗: {str(e)}'}), 500


@app.route('/api/poseaug/batch-augment-all', methods=['POST'])
def batch_augment_all_frames():
    """
    批量增強所有幀
    請求 body:
    {
        "poses": [[[x, y, z], ...], ...],  // 多個姿態 (每個姿態 17 個關鍵點)
        "augmentation_types": ["BL", "BA", "RT", "POS"],
        "num_augmentations_per_pose": 5,
        "config": { ... }  // 可選配置
    }
    """
    if not POSEAUG_AVAILABLE:
        return jsonify({'success': False, 'error': 'PoseAug 模組不可用'}), 503

    try:
        data = request.get_json()
        if not data:
            return jsonify({'success': False, 'error': '缺少請求數據'}), 400

        poses_data = data.get('poses', [])
        if not poses_data:
            return jsonify({'success': False, 'error': '沒有提供姿態數據'}), 400

        aug_types = data.get('augmentation_types', ['BL', 'BA', 'RT', 'POS'])
        num_aug_per_pose = data.get('num_augmentations_per_pose', 3)
        config_data = data.get('config', {})

        # 設定增強配置
        config = AugmentationConfig(
            bone_length_enabled='BL' in aug_types,
            bone_angle_enabled='BA' in aug_types,
            rotation_enabled='RT' in aug_types,
            translation_enabled='POS' in aug_types,
            bone_length_scale_range=tuple(config_data.get('bl_scale_range', (0.9, 1.1))),
            bone_angle_range=abs(config_data.get('ba_angle_range', [-15, 15])[1]),
            rotation_range=(
                abs(config_data.get('rt_angle_range', [-30, 30])[1]),
                abs(config_data.get('rt_angle_range', [-30, 30])[1]),
                15.0
            ),
            translation_range=abs(config_data.get('pos_shift_range', [-0.05, 0.05])[1])
        )

        augmentor = PoseAugmentor(config)

        # 批量增強所有幀
        all_augmented = []

        for frame_idx, pose_data in enumerate(poses_data):
            pose = np.array(pose_data, dtype=np.float32)

            frame_augmentations = {
                'frame_index': frame_idx,
                'original_pose': pose.tolist(),
                'augmented_poses': []
            }

            for aug_idx in range(num_aug_per_pose):
                aug_pose = augmentor.augment_pose(pose)
                frame_augmentations['augmented_poses'].append(aug_pose.tolist())

            all_augmented.append(frame_augmentations)

        total_generated = len(poses_data) * num_aug_per_pose

        return jsonify({
            'success': True,
            'original_frame_count': len(poses_data),
            'augmentations_per_frame': num_aug_per_pose,
            'total_generated': total_generated,
            'augmented_frames': all_augmented,
            'applied_augmentations': aug_types
        })

    except Exception as e:
        print(f"批量增強錯誤: {str(e)}")
        traceback.print_exc()
        return jsonify({'success': False, 'error': f'批量增強失敗: {str(e)}'}), 500


@app.route('/api/poseaug/batch-generate', methods=['POST'])
def batch_generate_augmentations():
    """
    批量生成增強姿態 (使用 Generator)
    請求 body:
    {
        "poses": [[[x, y, z], ...], ...],  // 多個姿態
        "num_augmentations_per_pose": 10,
        "preset": "moderate"  // 或自定義 config
    }
    """
    if not POSEAUG_AVAILABLE:
        return jsonify({'success': False, 'error': 'PoseAug 模組不可用'}), 503

    try:
        data = request.get_json()
        if not data:
            return jsonify({'success': False, 'error': '缺少請求數據'}), 400

        poses_data = data.get('poses', [])
        num_aug = data.get('num_augmentations_per_pose', 10)
        preset = data.get('preset', 'moderate')

        # 如果沒有提供姿態，使用範例
        if not poses_data:
            poses_data = [create_sample_pose().tolist()]

        poses = np.array(poses_data, dtype=np.float32)

        # 根據預設選擇配置
        preset_configs = {
            'conservative': AugmentationConfig(
                bl_scale_range=(0.95, 1.05),
                ba_angle_range=(-5, 5),
                rt_angle_range=(-10, 10),
                pos_shift_range=(-0.02, 0.02)
            ),
            'moderate': AugmentationConfig(
                bl_scale_range=(0.9, 1.1),
                ba_angle_range=(-15, 15),
                rt_angle_range=(-30, 30),
                pos_shift_range=(-0.05, 0.05)
            ),
            'aggressive': AugmentationConfig(
                bl_scale_range=(0.8, 1.2),
                ba_angle_range=(-25, 25),
                rt_angle_range=(-45, 45),
                pos_shift_range=(-0.1, 0.1)
            )
        }

        config = preset_configs.get(preset, preset_configs['moderate'])

        # 自定義配置覆蓋
        if 'config' in data:
            config_data = data['config']
            config = AugmentationConfig(
                bl_scale_range=tuple(config_data.get('bl_scale_range', config.bl_scale_range)),
                ba_angle_range=tuple(config_data.get('ba_angle_range', config.ba_angle_range)),
                rt_angle_range=tuple(config_data.get('rt_angle_range', config.rt_angle_range)),
                pos_shift_range=tuple(config_data.get('pos_shift_range', config.pos_shift_range))
            )

        # 創建生成器
        generator = PoseAugmentationGenerator(config)
        augmented_dataset = generator.generate_augmented_dataset(poses, num_aug)

        return jsonify({
            'success': True,
            'original_count': len(poses),
            'augmented_count': len(augmented_dataset),
            'augmented_poses': augmented_dataset.tolist(),
            'config_used': {
                'bl_scale_range': config.bl_scale_range,
                'ba_angle_range': config.ba_angle_range,
                'rt_angle_range': config.rt_angle_range,
                'pos_shift_range': config.pos_shift_range
            }
        })

    except Exception as e:
        print(f"批量增強錯誤: {str(e)}")
        traceback.print_exc()
        return jsonify({'success': False, 'error': f'批量增強失敗: {str(e)}'}), 500


if __name__ == '__main__':
    print("=====================================================")
    print("啟動 Smart Tennis 後端服務...")
    print(f"上傳目錄 (uploads): {UPLOAD_FOLDER}")
    print(f"輸出目錄 (output): {OUTPUT_FOLDER}")
    print(f"元數據檔案: {METADATA_FILE}")
    print(f"姿態辨識: {'已啟用 (YOLO11' + POSE_MODEL_SIZE + '-pose)' if USE_POSE else '已停用'}")
    print(f"3D 重建: {'已啟用' if MULTIVIEW_3D_AVAILABLE else '未啟用'}")
    print(f"PoseAug 姿態增強: {'已啟用' if POSEAUG_AVAILABLE else '未啟用'}")
    print(f"PoseAug GAN: {'已啟用' if POSEAUG_GAN_AVAILABLE else '未啟用'}")
    if VIDEOPOSE3D_AVAILABLE and videopose3d_lifter:
        vp_status = videopose3d_lifter.get_status()
        print(f"VideoPose3D 2D->3D: 已啟用 (weights={'loaded' if vp_status['weights_loaded'] else 'NOT loaded'}, params={vp_status['model_params']:,})")
    else:
        print(f"VideoPose3D 2D->3D: 未啟用 (使用啟發式 fallback)")
    print("伺服器地址: http://localhost:5000")
    print("=====================================================")
    app.run(debug=True, host='0.0.0.0', port=5000)