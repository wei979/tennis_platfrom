"""
網球追蹤模組 - 整合姿態辨識的完整版本
使用 YOLOv8 追蹤網球，YOLO11-Pose 繪製人體骨架
"""
import cv2
import numpy as np
from ultralytics import YOLO
import os
from collections import defaultdict
import math

# 嘗試導入姿態檢測模組
try:
    from pose_detector import PoseDetector
    POSE_AVAILABLE = True
except ImportError:
    POSE_AVAILABLE = False


class TennisTracker:
    """
    網球追蹤器 - 結合球追蹤和姿態繪製
    """

    def __init__(self, model_path=None, use_pose=True, pose_model_size='l'):
        """
        初始化網球追蹤器

        Args:
            model_path: 網球檢測模型路徑 (已棄用，改用專用網球模型)
            use_pose: 是否在輸出影片中繪製姿態骨架
            pose_model_size: 姿態模型大小 ('n', 's', 'm', 'l', 'x')
        """
        # 優先使用自訓練的網球偵測模型
        tennis_ball_model = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
            'models', 'tennis_ball', 'best.pt'
        )
        if os.path.exists(tennis_ball_model):
            self.model_path = tennis_ball_model
            self.use_custom_model = True
            print(f"使用自訓練網球模型: {tennis_ball_model}")
        else:
            # Fallback 到通用模型
            if model_path is None:
                model_path = os.getenv('YOLO_MODEL_PATH', '../models/yolov8n.pt')
            self.model_path = model_path
            self.use_custom_model = False
            print(f"使用通用 YOLO 模型: {self.model_path}")

        self.model = None
        self.load_model()

        # 姿態檢測器 (用於繪製)
        self.use_pose = use_pose and POSE_AVAILABLE
        self.pose_detector = None

        if self.use_pose:
            try:
                print(f"🏃 初始化姿態繪製 (YOLO11{pose_model_size}-pose)...")
                self.pose_detector = PoseDetector(model_size=pose_model_size)
                print("✅ 姿態繪製已啟用")
            except Exception as e:
                print(f"⚠️ 姿態繪製初始化失敗: {e}")
                self.use_pose = False

        # 建立可接受的球類類別ID集合
        self.accepted_class_ids = set()

        if self.use_custom_model:
            # 自訓練模型: 接受所有類別 (只有 tennis-ball)
            names = getattr(self.model, 'names', {})
            if isinstance(names, dict):
                self.accepted_class_ids = set(names.keys())
            else:
                self.accepted_class_ids = set(range(len(names)))
            self.tennis_ball_class_id = 0
            print(f"   網球類別: {names}")
        else:
            # 通用 COCO 模型: 過濾球類相關類別
            self.accepted_class_ids = set([32, 33, 34, 35, 36, 37])
            try:
                names = getattr(self.model, 'names', None)
                if names is None:
                    names = getattr(getattr(self.model, 'model', None), 'names', None)
                if names is not None:
                    if isinstance(names, dict):
                        for idx, name in sorted(names.items()):
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
            self.tennis_ball_class_id = 37

        # 追蹤參數
        self.confidence_threshold = float(os.getenv('CONFIDENCE_THRESHOLD', '0.3'))
        self.max_disappeared = 10
        # 網球最大尺寸限制 (佔畫面比例)：超過此比例的偵測視為誤判 (黃色物體等)
        self.max_ball_ratio = float(os.getenv('MAX_BALL_RATIO', '0.05'))  # 預設 5% 畫面寬度

    def load_model(self):
        """載入YOLO模型"""
        try:
            if not os.path.exists(self.model_path):
                print("下載 YOLOv8 模型...")
                os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
                self.model = YOLO('yolov8n.pt')
                self.model.save(self.model_path)
            else:
                self.model = YOLO(self.model_path)

            print(f"已載入 YOLO 模型: {self.model_path}")
        except Exception as e:
            print(f"載入模型失敗: {e}")
            self.model = YOLO('yolov8n.pt')

    def detect_tennis_ball(self, frame):
        """在單一幀中檢測網球"""
        results = self.model(frame, verbose=False)
        detections = []
        frame_h, frame_w = frame.shape[:2]
        max_size = frame_w * self.max_ball_ratio  # 最大允許像素尺寸

        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    class_id = int(box.cls[0])
                    confidence = float(box.conf[0])

                    if (class_id in self.accepted_class_ids) and confidence > self.confidence_threshold:
                        x1, y1, x2, y2 = box.xyxy[0].tolist()
                        width = x2 - x1
                        height = y2 - y1

                        # 過濾過大的偵測 (黃色物體誤判)
                        if width > max_size or height > max_size:
                            continue

                        center_x = (x1 + x2) / 2
                        center_y = (y1 + y2) / 2

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
        如果啟用姿態繪製，會在輸出影片中繪製人體骨架。
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
            'trajectories': [],
            'pose_enabled': self.use_pose
        }

        # 設置輸出影片
        out = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        frame_count = 0
        ball_tracks = defaultdict(list)

        print(f"處理 {total_frames} 幀...")
        if self.use_pose:
            print("📊 同時進行姿態繪製...")

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
                annotated_frame = frame.copy()

                # 繪製姿態骨架 (只繪製主要球員 - 邊界框最大的人)
                if self.use_pose and self.pose_detector:
                    # 每幀都檢測姿態 (GPU 加速，無延遲)
                    all_poses = self.pose_detector.detect_pose(frame)
                    # 只保留邊界框最大的人 (主要球員)
                    if all_poses:
                        main_pose = max(all_poses, key=lambda p:
                            (p['bbox'][2] - p['bbox'][0]) * (p['bbox'][3] - p['bbox'][1])
                        )
                        annotated_frame = self.pose_detector.draw_pose(
                            annotated_frame, main_pose,
                            draw_skeleton=True,
                            draw_keypoints=True
                        )

                # 繪製網球檢測
                annotated_frame = self.draw_detections(annotated_frame, detections, frame_count)

                out.write(annotated_frame)

            frame_count += 1

            # 進度顯示
            if frame_count % 30 == 0:
                progress = (frame_count / total_frames) * 100
                print(f"處理進度: {progress:.1f}%")

        cap.release()
        if out:
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
        """在幀上繪製檢測結果"""
        annotated_frame = frame.copy()

        for detection in detections:
            x1, y1, x2, y2 = detection['bbox']
            center_x, center_y = detection['center']
            confidence = detection['confidence']

            # 繪製邊界框 (網球用亮綠色)
            cv2.rectangle(annotated_frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

            # 繪製中心點
            cv2.circle(annotated_frame, (int(center_x), int(center_y)), 5, (0, 0, 255), -1)

            # 顯示信心分數
            label = f"Ball: {confidence:.2f}"
            cv2.putText(annotated_frame, label, (int(x1), int(y1) - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # 顯示幀號和狀態
        cv2.putText(annotated_frame, f"Frame: {frame_number}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        if self.use_pose:
            cv2.putText(annotated_frame, "POSE: ON", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        return annotated_frame

    def analyze_trajectories(self, ball_positions):
        """分析網球軌跡"""
        trajectories = []
        current_trajectory = []

        for frame_data in ball_positions:
            if frame_data['detections']:
                best_detection = max(frame_data['detections'], key=lambda x: x['confidence'])
                current_trajectory.append({
                    'frame': frame_data['frame_number'],
                    'timestamp': frame_data['timestamp'],
                    'position': best_detection['center'],
                    'confidence': best_detection['confidence']
                })
            else:
                if len(current_trajectory) > 5:
                    trajectories.append(current_trajectory)
                current_trajectory = []

        if len(current_trajectory) > 5:
            trajectories.append(current_trajectory)

        analyzed_trajectories = []
        for i, trajectory in enumerate(trajectories):
            analysis = self.analyze_single_trajectory(trajectory, i)
            if analysis:
                analyzed_trajectories.append(analysis)

        return analyzed_trajectories

    def analyze_single_trajectory(self, trajectory, trajectory_id):
        """分析單條軌跡"""
        if len(trajectory) < 2:
            return None

        positions = [point['position'] for point in trajectory]
        timestamps = [point['timestamp'] for point in trajectory]

        velocities = []
        for i in range(1, len(positions)):
            dx = positions[i][0] - positions[i-1][0]
            dy = positions[i][1] - positions[i-1][1]
            dt = timestamps[i] - timestamps[i-1]

            if dt > 0:
                speed = np.sqrt(dx**2 + dy**2) / dt
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
        """計算追蹤穩定度分數 (0-100)"""
        detected_frames = [
            frame_data for frame_data in ball_positions
            if frame_data.get('detections')
        ]

        num_detected_frames = len(detected_frames)

        # 檢測幀數比例
        if total_frames == 0:
            detection_ratio = 0.0
        else:
            detection_ratio = num_detected_frames / total_frames

        # 平均信心分數
        avg_confidence = 0.0
        if num_detected_frames > 0:
            total_confidence = 0
            for frame_data in detected_frames:
                best_detection = max(frame_data['detections'], key=lambda x: x['confidence'])
                total_confidence += best_detection['confidence']

            avg_confidence = total_confidence / num_detected_frames

        # 穩定度分數 = 70% 檢測比例 + 30% 平均信心
        stability_score_raw = (detection_ratio * 0.7) + (avg_confidence * 0.3)
        stability_score_100 = round(stability_score_raw * 100, 2)

        return stability_score_100
