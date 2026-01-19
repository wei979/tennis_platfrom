from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os
import cv2
import json
from werkzeug.utils import secure_filename
# 假設這些類別在您的專案中存在，如果實際不存在，請確保它們有適當的 Mock
# ❗ 提醒: 確保您的 tennis_tracker, shot_detector, speed_analyzer 類別存在並可匯入
from tennis_tracker import TennisTracker
from shot_detector import ShotDetector
from speed_analyzer import SpeedAnalyzer
import uuid
from datetime import datetime
import subprocess
import traceback

# =========================================================
# 應用程式初始化
# =========================================================
app = Flask(__name__)
CORS(app)

# 獲取 app.py 所在的目錄 (.../main/smart-tennis/backend)
CURRENT_DIR = os.path.abspath(os.path.dirname(__file__)) 
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

# 初始化分析器
try:
    # ❗ 提醒: 檢查您的環境變數 YOLO_MODEL_PATH 是否已設定
    tennis_tracker = TennisTracker(model_path=os.getenv('YOLO_MODEL_PATH', None))
    shot_detector = ShotDetector()
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
                 'initial_speed': speed_results.get('avg_initial_speed_kmh', 0.0), # <--- 修正這行
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


if __name__ == '__main__':
    print("=====================================================")
    print("啟動 Smart Tennis 後端服務...")
    print(f"上傳目錄 (uploads): {UPLOAD_FOLDER}")
    print(f"輸出目錄 (output): {OUTPUT_FOLDER}")
    print(f"元數據檔案: {METADATA_FILE}")
    print("伺服器地址: http://localhost:5000")
    print("=====================================================")
    app.run(debug=True, host='0.0.0.0', port=5000)