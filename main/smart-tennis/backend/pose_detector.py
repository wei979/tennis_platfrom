"""
姿態辨識模組 - 使用 YOLO11l-pose 進行人體姿態檢測
適用於網球擊球姿態分析
"""
import cv2
import numpy as np
from ultralytics import YOLO
import os
import math
from collections import deque

# COCO 17 關鍵點定義
KEYPOINT_NAMES = {
    0: 'nose',
    1: 'left_eye', 2: 'right_eye',
    3: 'left_ear', 4: 'right_ear',
    5: 'left_shoulder', 6: 'right_shoulder',
    7: 'left_elbow', 8: 'right_elbow',
    9: 'left_wrist', 10: 'right_wrist',
    11: 'left_hip', 12: 'right_hip',
    13: 'left_knee', 14: 'right_knee',
    15: 'left_ankle', 16: 'right_ankle'
}

# 骨架連接定義 (用於繪製)
SKELETON_CONNECTIONS = [
    # 頭部
    (0, 1), (0, 2), (1, 3), (2, 4),
    # 軀幹
    (5, 6), (5, 11), (6, 12), (11, 12),
    # 左臂
    (5, 7), (7, 9),
    # 右臂
    (6, 8), (8, 10),
    # 左腿
    (11, 13), (13, 15),
    # 右腿
    (12, 14), (14, 16)
]

# 網球分析重點關鍵點索引
TENNIS_KEY_POINTS = {
    'left_shoulder': 5, 'right_shoulder': 6,
    'left_elbow': 7, 'right_elbow': 8,
    'left_wrist': 9, 'right_wrist': 10,
    'left_hip': 11, 'right_hip': 12,
    'left_knee': 13, 'right_knee': 14
}


class PoseDetector:
    """
    使用 YOLO11l-pose 進行人體姿態檢測的類別
    """

    def __init__(self, model_size='l', device=None):
        """
        初始化姿態檢測器

        Args:
            model_size: 模型大小 ('n', 's', 'm', 'l', 'x')，預設使用 'l' (large)
            device: 運算設備 (None 自動選擇, 'cpu', 'cuda:0' 等)
        """
        self.model_size = model_size
        self.model_name = f"yolo11{model_size}-pose.pt"
        self.model = None
        self.device = device

        # 檢測參數
        self.confidence_threshold = 0.5
        self.iou_threshold = 0.45

        # 姿態歷史 (用於動作分析)
        self.pose_history = deque(maxlen=30)  # 保留最近30幀

        # 載入模型
        self.load_model()

    def load_model(self):
        """載入 YOLO11 Pose 模型"""
        try:
            import torch
            print(f"正在載入 YOLO11 Pose 模型: {self.model_name}")
            self.model = YOLO(self.model_name)

            # 自動選擇設備：優先使用 CUDA GPU
            if self.device is None:
                if torch.cuda.is_available():
                    self.device = 'cuda:0'
                    print(f"   偵測到 GPU: {torch.cuda.get_device_name(0)}")
                else:
                    self.device = 'cpu'

            # 設置設備
            self.model.to(self.device)

            print(f"✅ 成功載入 {self.model_name}")
            print(f"   運算設備: {self.device}")

        except Exception as e:
            print(f"❌ 載入模型失敗: {e}")
            print("   嘗試使用較小的模型...")
            try:
                self.model = YOLO("yolo11n-pose.pt")
                print("✅ 已載入備用模型 yolo11n-pose.pt")
            except Exception as e2:
                raise RuntimeError(f"無法載入任何姿態模型: {e2}")

    def detect_pose(self, frame):
        """
        在單一幀中檢測人體姿態

        Args:
            frame: BGR 格式的影像幀

        Returns:
            list: 檢測到的人體姿態列表，每個元素包含關鍵點和邊界框
        """
        results = self.model(
            frame,
            verbose=False,
            conf=self.confidence_threshold,
            iou=self.iou_threshold
        )

        poses = []

        for result in results:
            if result.keypoints is None:
                continue

            keypoints_data = result.keypoints.data.cpu().numpy()
            boxes_data = result.boxes.xyxy.cpu().numpy() if result.boxes is not None else None
            confs = result.boxes.conf.cpu().numpy() if result.boxes is not None else None

            for i, kpts in enumerate(keypoints_data):
                # kpts 格式: [17, 3] -> (x, y, confidence) for each keypoint
                pose_data = {
                    'keypoints': {},
                    'keypoints_raw': kpts,
                    'bbox': boxes_data[i] if boxes_data is not None else None,
                    'confidence': float(confs[i]) if confs is not None else 0.0
                }

                # 解析每個關鍵點
                for idx, name in KEYPOINT_NAMES.items():
                    if idx < len(kpts):
                        x, y, conf = kpts[idx]
                        pose_data['keypoints'][name] = {
                            'x': float(x),
                            'y': float(y),
                            'confidence': float(conf),
                            'visible': conf > 0.3
                        }

                poses.append(pose_data)

        return poses

    def analyze_tennis_pose(self, pose_data, frame_height):
        """
        分析網球擊球姿態

        Args:
            pose_data: 單一人體的姿態數據
            frame_height: 影片高度 (用於座標標準化)

        Returns:
            dict: 姿態分析結果
        """
        kpts = pose_data['keypoints']
        analysis = {
            'dominant_side': None,  # 慣用手側
            'arm_extension': 0.0,   # 手臂伸展度
            'body_rotation': 0.0,   # 身體旋轉角度
            'knee_bend': 0.0,       # 膝蓋彎曲度
            'wrist_height': 0.0,    # 手腕高度 (標準化)
            'swing_angle': 0.0,     # 揮拍角度
            'stance_width': 0.0,    # 站姿寬度
            'balance_score': 0.0,   # 平衡分數
        }

        # 檢查必要關鍵點是否可見
        required_points = ['left_shoulder', 'right_shoulder', 'left_hip', 'right_hip']
        if not all(kpts.get(p, {}).get('visible', False) for p in required_points):
            return analysis

        # 計算身體中心
        body_center_x = (kpts['left_hip']['x'] + kpts['right_hip']['x']) / 2
        body_center_y = (kpts['left_hip']['y'] + kpts['right_hip']['y']) / 2

        # 判斷慣用手側 (基於哪隻手腕位置更靠外側)
        if kpts.get('left_wrist', {}).get('visible') and kpts.get('right_wrist', {}).get('visible'):
            left_ext = abs(kpts['left_wrist']['x'] - body_center_x)
            right_ext = abs(kpts['right_wrist']['x'] - body_center_x)
            analysis['dominant_side'] = 'left' if left_ext > right_ext else 'right'

        # 計算手臂伸展度 (肩膀到手腕的距離)
        dominant = analysis['dominant_side'] or 'right'
        if kpts.get(f'{dominant}_shoulder', {}).get('visible') and kpts.get(f'{dominant}_wrist', {}).get('visible'):
            shoulder = kpts[f'{dominant}_shoulder']
            wrist = kpts[f'{dominant}_wrist']
            arm_length = math.sqrt(
                (wrist['x'] - shoulder['x'])**2 +
                (wrist['y'] - shoulder['y'])**2
            )
            # 標準化 (假設正常手臂長度約為身高的 40%)
            analysis['arm_extension'] = min(arm_length / (frame_height * 0.4), 1.5)

        # 計算身體旋轉角度 (肩線與髖線的夾角差)
        shoulder_angle = math.atan2(
            kpts['right_shoulder']['y'] - kpts['left_shoulder']['y'],
            kpts['right_shoulder']['x'] - kpts['left_shoulder']['x']
        )
        hip_angle = math.atan2(
            kpts['right_hip']['y'] - kpts['left_hip']['y'],
            kpts['right_hip']['x'] - kpts['left_hip']['x']
        )
        analysis['body_rotation'] = math.degrees(shoulder_angle - hip_angle)

        # 計算膝蓋彎曲度
        if (kpts.get('left_hip', {}).get('visible') and
            kpts.get('left_knee', {}).get('visible') and
            kpts.get('left_ankle', {}).get('visible')):
            knee_angle = self._calculate_angle(
                kpts['left_hip'], kpts['left_knee'], kpts['left_ankle']
            )
            analysis['knee_bend'] = 180 - knee_angle  # 彎曲度 = 180 - 膝蓋角度

        # 計算手腕高度 (標準化到 0-1，0=底部，1=頂部)
        if kpts.get(f'{dominant}_wrist', {}).get('visible'):
            wrist_y = kpts[f'{dominant}_wrist']['y']
            analysis['wrist_height'] = max(0, 1 - (wrist_y / frame_height))

        # 計算揮拍角度 (手腕相對於肩膀的角度)
        if (kpts.get(f'{dominant}_shoulder', {}).get('visible') and
            kpts.get(f'{dominant}_wrist', {}).get('visible')):
            swing_angle = math.atan2(
                kpts[f'{dominant}_wrist']['y'] - kpts[f'{dominant}_shoulder']['y'],
                kpts[f'{dominant}_wrist']['x'] - kpts[f'{dominant}_shoulder']['x']
            )
            analysis['swing_angle'] = math.degrees(swing_angle)

        # 計算站姿寬度 (兩腳踝距離)
        if (kpts.get('left_ankle', {}).get('visible') and
            kpts.get('right_ankle', {}).get('visible')):
            stance = abs(kpts['left_ankle']['x'] - kpts['right_ankle']['x'])
            # 標準化 (假設正常站姿寬度約為身高的 20-30%)
            analysis['stance_width'] = stance / frame_height

        # 計算平衡分數 (基於身體中心與腳部位置)
        if (kpts.get('left_ankle', {}).get('visible') and
            kpts.get('right_ankle', {}).get('visible')):
            feet_center_x = (kpts['left_ankle']['x'] + kpts['right_ankle']['x']) / 2
            balance_offset = abs(body_center_x - feet_center_x)
            # 偏移越小，平衡越好，轉換為 0-100 分數
            analysis['balance_score'] = max(0, 100 - (balance_offset / frame_height * 200))

        return analysis

    def _calculate_angle(self, point_a, point_b, point_c):
        """
        計算三點形成的角度 (角度在 point_b)
        """
        ba = np.array([point_a['x'] - point_b['x'], point_a['y'] - point_b['y']])
        bc = np.array([point_c['x'] - point_b['x'], point_c['y'] - point_b['y']])

        cosine = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
        angle = np.arccos(np.clip(cosine, -1.0, 1.0))

        return math.degrees(angle)

    def detect_swing_motion(self, poses_history):
        """
        檢測揮拍動作

        Args:
            poses_history: 歷史姿態數據列表

        Returns:
            dict: 揮拍動作分析
        """
        if len(poses_history) < 5:
            return {'is_swinging': False}

        # 分析手腕運動軌跡
        wrist_positions = []
        for frame_poses in poses_history:
            if frame_poses and len(frame_poses) > 0:
                pose = frame_poses[0]  # 取第一個人
                for side in ['right', 'left']:
                    wrist = pose['keypoints'].get(f'{side}_wrist')
                    if wrist and wrist.get('visible'):
                        wrist_positions.append({
                            'x': wrist['x'],
                            'y': wrist['y'],
                            'side': side
                        })
                        break

        if len(wrist_positions) < 5:
            return {'is_swinging': False}

        # 計算手腕移動速度
        velocities = []
        for i in range(1, len(wrist_positions)):
            dx = wrist_positions[i]['x'] - wrist_positions[i-1]['x']
            dy = wrist_positions[i]['y'] - wrist_positions[i-1]['y']
            velocity = math.sqrt(dx**2 + dy**2)
            velocities.append(velocity)

        max_velocity = max(velocities) if velocities else 0
        avg_velocity = np.mean(velocities) if velocities else 0

        # 判斷是否在揮拍 (速度超過閾值)
        is_swinging = max_velocity > 30  # 像素/幀

        return {
            'is_swinging': is_swinging,
            'max_velocity': max_velocity,
            'avg_velocity': avg_velocity,
            'dominant_side': wrist_positions[-1]['side'] if wrist_positions else None
        }

    def draw_pose(self, frame, pose_data, draw_skeleton=True, draw_keypoints=True):
        """
        在影像上繪製姿態骨架

        Args:
            frame: 影像幀
            pose_data: 姿態數據
            draw_skeleton: 是否繪製骨架連接線
            draw_keypoints: 是否繪製關鍵點

        Returns:
            繪製後的影像幀
        """
        annotated = frame.copy()
        kpts = pose_data['keypoints']

        # 顏色定義
        colors = {
            'skeleton': (0, 255, 255),  # 黃色
            'keypoint': (0, 255, 0),     # 綠色
            'wrist': (255, 0, 255),      # 紫色 (重點標記手腕)
            'shoulder': (255, 128, 0),   # 橙色
        }

        # 繪製骨架連接線
        if draw_skeleton:
            for start_idx, end_idx in SKELETON_CONNECTIONS:
                start_name = KEYPOINT_NAMES.get(start_idx)
                end_name = KEYPOINT_NAMES.get(end_idx)

                if start_name in kpts and end_name in kpts:
                    start = kpts[start_name]
                    end = kpts[end_name]

                    if start.get('visible') and end.get('visible'):
                        start_pt = (int(start['x']), int(start['y']))
                        end_pt = (int(end['x']), int(end['y']))
                        cv2.line(annotated, start_pt, end_pt, colors['skeleton'], 2)

        # 繪製關鍵點
        if draw_keypoints:
            for name, point in kpts.items():
                if point.get('visible'):
                    x, y = int(point['x']), int(point['y'])

                    # 根據關鍵點類型選擇顏色和大小
                    if 'wrist' in name:
                        color = colors['wrist']
                        radius = 8
                    elif 'shoulder' in name:
                        color = colors['shoulder']
                        radius = 6
                    else:
                        color = colors['keypoint']
                        radius = 4

                    cv2.circle(annotated, (x, y), radius, color, -1)
                    cv2.circle(annotated, (x, y), radius, (255, 255, 255), 1)

        return annotated

    def draw_pose_analysis(self, frame, analysis, position=(10, 30)):
        """
        在影像上繪製姿態分析資訊
        """
        annotated = frame.copy()
        y_offset = position[1]

        info_lines = [
            f"Dominant Side: {analysis.get('dominant_side', 'N/A')}",
            f"Body Rotation: {analysis.get('body_rotation', 0):.1f} deg",
            f"Arm Extension: {analysis.get('arm_extension', 0):.2f}",
            f"Wrist Height: {analysis.get('wrist_height', 0):.2f}",
            f"Balance: {analysis.get('balance_score', 0):.1f}",
        ]

        for line in info_lines:
            cv2.putText(annotated, line, (position[0], y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            cv2.putText(annotated, line, (position[0], y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 0), 1)
            y_offset += 20

        return annotated
