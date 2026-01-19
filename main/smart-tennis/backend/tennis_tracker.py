import cv2
import numpy as np
from ultralytics import YOLO
import os
from collections import defaultdict
import math

class TennisTracker:
    def __init__(self, model_path=None):
        """
        初始化網球追蹤器
        """
        if model_path is None:
            # 允許透過環境變數覆寫模型路徑
            model_path = os.getenv('YOLO_MODEL_PATH', '../models/yolov8n.pt')
        
        self.model_path = model_path
        self.model = None
        self.load_model()
        
        # 建立可接受的球類類別ID集合（預設涵蓋 COCO 球類 32..37 與 sports ball=37）
        self.accepted_class_ids = set([32, 33, 34, 35, 36, 37])
        
        # 依據模型類別名稱擴充
        try:
            names = getattr(self.model, 'names', None)
            if names is None:
                names = getattr(getattr(self.model, 'model', None), 'names', None)
            if names is not None:
                if isinstance(names, dict):
                    iterable = sorted(names.items())
                    for idx, name in iterable:
                        n = str(name).lower()
                        if n in ('tennis ball', 'sports ball') or 'tennis' in n:
                            self.accepted_class_ids.add(int(idx))
                else:
                    for idx, name in enumerate(names):
                        n = str(name).lower()
                        if n in ('tennis ball', 'sports ball') or 'tennis' in n:
                            self.accepted_class_ids.add(int(idx))
        except Exception:
            pass
        
        # 網球類別ID (保留原設計，供其他邏輯參考)
        self.tennis_ball_class_id = 37
        
        # 追蹤參數
        self.confidence_threshold = float(os.getenv('CONFIDENCE_THRESHOLD', '0.3'))
        self.max_disappeared = 10
        
    def load_model(self):
        """載入YOLO模型"""
        try:
            # 如果模型不存在，下載預訓練模型
            if not os.path.exists(self.model_path):
                print("下載 YOLOv8 模型...")
                os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
                self.model = YOLO('yolov8n.pt')  # 這會自動下載模型
                self.model.save(self.model_path)
            else:
                self.model = YOLO(self.model_path)
            
            print(f"已載入 YOLO 模型: {self.model_path}")
        except Exception as e:
            print(f"載入模型失敗: {e}")
            # 使用預設模型作為備選
            self.model = YOLO('yolov8n.pt')
    
    def detect_tennis_ball(self, frame):
        """
        在單一幀中檢測網球
        """
        results = self.model(frame, verbose=False)
        detections = []
        
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    # 檢查是否為網球或運動球類
                    class_id = int(box.cls[0])
                    confidence = float(box.conf[0])
                    
                    # 擴展檢測範圍，包含球類物體
                    if (class_id in self.accepted_class_ids) and confidence > self.confidence_threshold:
                        
                        x1, y1, x2, y2 = box.xyxy[0].tolist()
                        center_x = (x1 + x2) / 2
                        center_y = (y1 + y2) / 2
                        width = x2 - x1
                        height = y2 - y1
                        
                        detections.append({
                            'center': (center_x, center_y),
                            'bbox': (x1, y1, x2, y2),
                            'confidence': confidence,
                            'size': (width, height)
                        })
        
        return detections
    
    def track_ball(self, video_path, output_path=None):
        """
        追蹤整個影片中的網球，並計算穩定度分數。
        """
        print(f"開始追蹤網球: {video_path}")
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"無法開啟影片: {video_path}")
        
        # 獲取影片資訊
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # 初始化追蹤結果
        tracking_results = {
            'video_info': {
                'fps': fps,
                'width': width,
                'height': height,
                'total_frames': total_frames
            },
            'ball_positions': [],
            'trajectories': []
        }
        
        # 設置輸出影片
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        frame_count = 0
        ball_tracks = defaultdict(list)
        current_track_id = 0
        
        print(f"處理 {total_frames} 幀...")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # 檢測網球
            detections = self.detect_tennis_ball(frame)
            
            frame_data = {
                'frame_number': frame_count,
                'timestamp': frame_count / fps,
                'detections': detections
            }
            
            tracking_results['ball_positions'].append(frame_data)
            
            # 繪製檢測結果
            if output_path:
                annotated_frame = self.draw_detections(frame, detections, frame_count)
                out.write(annotated_frame)
            
            frame_count += 1
            
            # 進度顯示
            if frame_count % 30 == 0:
                progress = (frame_count / total_frames) * 100
                print(f"處理進度: {progress:.1f}%")
        
        cap.release()
        if output_path:
            out.release()
        
        # 分析軌跡
        tracking_results['trajectories'] = self.analyze_trajectories(tracking_results['ball_positions'])
        
        # 計算穩定度分數
        stability_score = self.calculate_stability_score(
            tracking_results['ball_positions'], 
            total_frames
        )
        tracking_results['stability_score'] = stability_score
        
        print(f"追蹤完成，共檢測到 {len([p for p in tracking_results['ball_positions'] if p['detections']])} 幀包含網球")
        print(f"計算穩定度分數: {stability_score:.2f}/100")
        
        return tracking_results
    
    def draw_detections(self, frame, detections, frame_number):
        """
        在幀上繪製檢測結果
        """
        annotated_frame = frame.copy()
        
        for detection in detections:
            x1, y1, x2, y2 = detection['bbox']
            center_x, center_y = detection['center']
            confidence = detection['confidence']
            
            # 繪製邊界框
            cv2.rectangle(annotated_frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            
            # 繪製中心點
            cv2.circle(annotated_frame, (int(center_x), int(center_y)), 5, (0, 0, 255), -1)
            
            # 顯示信心分數
            label = f"Ball: {confidence:.2f}"
            cv2.putText(annotated_frame, label, (int(x1), int(y1) - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # 顯示幀號
        cv2.putText(annotated_frame, f"Frame: {frame_number}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        return annotated_frame
    
    def analyze_trajectories(self, ball_positions):
        """
        分析網球軌跡
        """
        trajectories = []
        current_trajectory = []
        
        for frame_data in ball_positions:
            if frame_data['detections']:
                # 取最可信的檢測
                best_detection = max(frame_data['detections'], key=lambda x: x['confidence'])
                current_trajectory.append({
                    'frame': frame_data['frame_number'],
                    'timestamp': frame_data['timestamp'],
                    'position': best_detection['center'],
                    'confidence': best_detection['confidence']
                })
            else:
                # 如果檢測中斷，結束當前軌跡
                if len(current_trajectory) > 5:  # 只保留足夠長的軌跡
                    trajectories.append(current_trajectory)
                current_trajectory = []
        
        # 處理最後一段軌跡
        if len(current_trajectory) > 5:
            trajectories.append(current_trajectory)
        
        # 分析每條軌跡
        analyzed_trajectories = []
        for i, trajectory in enumerate(trajectories):
            analysis = self.analyze_single_trajectory(trajectory, i)
            if analysis:
                analyzed_trajectories.append(analysis)
        
        return analyzed_trajectories
    
    def analyze_single_trajectory(self, trajectory, trajectory_id):
        """
        分析單條軌跡
        """
        if len(trajectory) < 2:
            return None
        
        positions = [point['position'] for point in trajectory]
        timestamps = [point['timestamp'] for point in trajectory]
        
        # 計算速度
        velocities = []
        for i in range(1, len(positions)):
            dx = positions[i][0] - positions[i-1][0]
            dy = positions[i][1] - positions[i-1][1]
            dt = timestamps[i] - timestamps[i-1]
            
            if dt > 0:
                speed = np.sqrt(dx**2 + dy**2) / dt  # 像素/秒
                velocities.append(speed)
        
        return {
            'id': trajectory_id,
            'start_frame': trajectory[0]['frame'],
            'end_frame': trajectory[-1]['frame'],
            'duration': timestamps[-1] - timestamps[0],
            'positions': positions,
            'velocities': velocities,
            'avg_velocity': np.mean(velocities) if velocities else 0,
            'max_velocity': np.max(velocities) if velocities else 0,
            'trajectory_length': len(trajectory)
        }
        
    def calculate_stability_score(self, ball_positions, total_frames):
        """
        計算追蹤穩定度分數 (0-100)。
        分數基於: 1. 檢測幀數比例, 2. 平均信心分數。
        """
        
        # 過濾掉沒有檢測結果的幀
        detected_frames = [
            frame_data for frame_data in ball_positions 
            if frame_data.get('detections')
        ]
        
        num_detected_frames = len(detected_frames)
        
        # 1. 檢測幀數比例 (Detection Ratio)
        if total_frames == 0:
            detection_ratio = 0.0
        else:
            detection_ratio = num_detected_frames / total_frames
            
        # 2. 平均信心分數 (Average Confidence)
        avg_confidence = 0.0
        if num_detected_frames > 0:
            total_confidence = 0
            for frame_data in detected_frames:
                # 取得該幀最優檢測的信心分數
                best_detection = max(frame_data['detections'], key=lambda x: x['confidence'])
                total_confidence += best_detection['confidence']
            
            avg_confidence = total_confidence / num_detected_frames
        
        # 3. 穩定度分數計算 (加權平均，並縮放到 0-100)
        # 70% 權重給檢測比例，30% 權重給平均信心分數
        stability_score_raw = (detection_ratio * 0.7) + (avg_confidence * 0.3)
        stability_score_100 = round(stability_score_raw * 100, 2)
        
        return stability_score_100