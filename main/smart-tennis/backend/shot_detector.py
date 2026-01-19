import cv2
import numpy as np
from collections import deque
import math

class ShotDetector:
    def __init__(self):
        """初始化正反手檢測器（簡化版本，不依賴 mediapipe）"""
        self.mediapipe_available = False
        print("⚠️ 使用簡化的擊球檢測（不依賴 MediaPipe）")
        
        # 檢測參數
        self.swing_threshold = 0.3
        self.shot_window = 30 # 增加窗口以過濾重複
        
    def detect_shots(self, video_path, tracking_results):
        """
        檢測影片中的正反手擊球（簡化版本）。
        【已更新】現在會計算 average_hit_height 和 average_hit_angle。
        """
        print("開始檢測正反手擊球（簡化模式）...")
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"無法開啟影片: {video_path}")
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # 1. 基於網球軌跡變化檢測擊球
        shots = self.detect_shots_from_ball_trajectory(tracking_results, fps)
        
        # 2. 計算平均擊球屬性 (基於像素的簡化估計)
        hit_metrics = self.calculate_hit_metrics(shots, tracking_results, frame_height)
        
        cap.release()
        
        print(f"檢測完成，找到 {len(shots)} 次擊球")
        
        # 3. 整合結果並回傳，包含 Flask 所需的 Key
        return {
            'shots': shots,
            'total_shots': len(shots),
            'forehand_count': len([s for s in shots if s['type'] == 'forehand']),
            'backhand_count': len([s for s in shots if s['type'] == 'backhand']),
            # 將計算出的指標加入回傳字典
            'average_hit_height': hit_metrics['average_hit_height'],
            'average_hit_angle': hit_metrics['average_hit_angle'],
        }
    
    def detect_shots_from_ball_trajectory(self, tracking_results, fps):
        """基於網球軌跡檢測擊球，並記錄擊球後的方向向量 (direction)"""
        shots = []
        ball_positions = tracking_results.get('ball_positions', [])
        
        if len(ball_positions) < 10:
            return shots
        
        # 計算網球速度和方向變化
        velocities = []
        for i in range(1, len(ball_positions)):
            prev_frame = ball_positions[i-1]
            curr_frame = ball_positions[i]
            
            if prev_frame.get('detections') and curr_frame.get('detections'):
                prev_pos = prev_frame['detections'][0]['center']
                curr_pos = curr_frame['detections'][0]['center']
                
                # 計算速度和方向
                dx = curr_pos[0] - prev_pos[0]
                dy = curr_pos[1] - prev_pos[1]
                distance = math.sqrt(dx**2 + dy**2)
                
                frame_diff = curr_frame['frame_number'] - prev_frame['frame_number']
                if frame_diff > 0:
                    velocity = distance / frame_diff
                    velocities.append({
                        'frame': curr_frame['frame_number'],
                        'velocity': velocity,
                        'position': curr_pos,
                        'direction': (dx, dy) # 儲存方向向量
                    })
        
        # 檢測速度突變（可能的擊球點）
        for i in range(2, len(velocities) - 2):
            current_vel = velocities[i]['velocity']
            prev_vel = velocities[i-1]['velocity']
            next_vel = velocities[i+1]['velocity']
            
            # 檢測速度突然增加（擊球）
            if (current_vel > prev_vel * 1.5 and 
                current_vel > 20 and 
                next_vel > current_vel * 0.7):
                
                shot_type = self.classify_shot_simple(velocities[i])
                
                shots.append({
                    'frame': velocities[i]['frame'],
                    'timestamp': velocities[i]['frame'] / fps,
                    'type': shot_type,
                    'side': 'right' if shot_type == 'forehand' else 'left',
                    'confidence': min(current_vel / 100, 1.0),
                    'ball_contact_frame': velocities[i]['frame'],
                    'swing_velocity': current_vel,
                    'direction': velocities[i]['direction'] # 確保擊球點帶有方向
                })
        
        # 過濾重複檢測
        return self.filter_duplicate_shots(shots)
    
    def classify_shot_simple(self, velocity_data):
        """簡化的擊球分類：基於球的 X 軸移動方向"""
        direction = velocity_data['direction']
        
        # 如果球向右移動較多，假設是正手；否則反手
        # 假設攝影機視角是從球場中央拍攝，大多數情況下球員擊球是將球打向對面
        if direction[0] > abs(direction[1]) * 0.5:
            return 'forehand'
        else:
            return 'backhand'
    
    def filter_duplicate_shots(self, shots):
        """過濾重複的擊球檢測，確保擊球間隔足夠"""
        if not shots:
            return shots
        
        filtered_shots = []
        shots.sort(key=lambda x: x['frame'])
        
        for shot in shots:
            is_duplicate = False
            for existing_shot in filtered_shots:
                frame_diff = abs(shot['frame'] - existing_shot['frame'])
                if frame_diff < self.shot_window:  # 使用 self.shot_window 幀作為最小間隔
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                filtered_shots.append(shot)
        
        return filtered_shots
    
    def calculate_hit_metrics(self, shots, tracking_results, frame_height):
        """
        計算平均擊球高度和角度。使用像素座標進行簡化估算。
        這兩個指標用於在 Flask app.py 中填補 summary 欄位。
        """
        hit_y_coords = []
        hit_angles = []
        
        # 創建球位置查找表 (僅包含有檢測的幀，中心座標)
        ball_positions = {
            f['frame_number']: f['detections'][0]['center'] 
            for f in tracking_results.get('ball_positions', []) 
            if f.get('detections')
        }
        
        for shot in shots:
            frame_num = shot.get('ball_contact_frame')
            
            if frame_num in ball_positions and frame_height > 0:
                # 1. 擊球點的 Y 座標 (像素)
                y_coord = ball_positions[frame_num][1] 
                hit_y_coords.append(y_coord)
                
                # 2. 使用擊球後一幀的方向來估算發射角度 (如果 direction 已儲存)
                if 'direction' in shot and shot['direction'][0] != 0:
                    dx, dy = shot['direction']
                    # 角度計算：使用 atan2 得到弧度，轉換為度數
                    # 由於 Y 軸在圖像中是倒置的，-dy 代表實際的向上移動
                    angle_deg = math.degrees(math.atan2(-dy, dx)) 
                    hit_angles.append(angle_deg)


        # --- 計算平均高度 (以米為單位的簡化值) ---
        if hit_y_coords and frame_height > 0:
            avg_y = np.mean(hit_y_coords)
            # 標準化：(frame_height - avg_y) / frame_height 得到 0 (底部) 到 1 (頂部) 的比例
            # 假設最大高度為 1.5 米
            hit_height_m = round(1.5 * (frame_height - avg_y) / frame_height, 2)
        else:
            hit_height_m = 0.0

        # --- 計算平均角度 (度) ---
        if hit_angles:
            avg_angle_deg = round(np.mean(hit_angles), 2)
        else:
            avg_angle_deg = 0.0
            
        return {
            'average_hit_height': hit_height_m,
            'average_hit_angle': avg_angle_deg
        }

# 如果您使用這個版本的 ShotDetector，則無需修改 app.py 中的 `calculate_missing_shot_metrics` 函數和其調用邏輯，只需確保 summary 鍵匹配即可。