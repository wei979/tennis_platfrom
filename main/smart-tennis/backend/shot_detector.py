"""
擊球檢測模組 - 整合姿態辨識的完整版本
使用 YOLO11-Pose 進行人體姿態分析，結合球軌跡進行擊球檢測
"""
import cv2
import numpy as np
from collections import deque
import math
import os

# 嘗試導入姿態檢測模組
try:
    from pose_detector import PoseDetector
    POSE_AVAILABLE = True
except ImportError:
    POSE_AVAILABLE = False
    print("⚠️ 姿態檢測模組不可用，將使用簡化版擊球檢測")


class ShotDetector:
    """
    擊球檢測器 - 結合姿態辨識和球軌跡分析
    """

    def __init__(self, use_pose=True, pose_model_size='l'):
        """
        初始化擊球檢測器

        Args:
            use_pose: 是否使用姿態辨識 (需要 GPU 支援)
            pose_model_size: 姿態模型大小 ('n', 's', 'm', 'l', 'x')
        """
        self.use_pose = use_pose and POSE_AVAILABLE
        self.pose_detector = None

        # 檢測參數
        self.swing_threshold = 0.3
        self.shot_window = 30  # 擊球間隔最小幀數
        self.velocity_spike_ratio = 1.5  # 速度突增比例

        # 初始化姿態檢測器
        if self.use_pose:
            try:
                print(f"🏃 初始化姿態辨識 (YOLO11{pose_model_size}-pose)...")
                self.pose_detector = PoseDetector(model_size=pose_model_size)
                print("✅ 姿態辨識已啟用")
            except Exception as e:
                print(f"⚠️ 姿態辨識初始化失敗: {e}")
                print("   將使用簡化版擊球檢測")
                self.use_pose = False
                self.pose_detector = None
        else:
            print("⚠️ 使用簡化的擊球檢測（不依賴姿態辨識）")

        # 姿態歷史記錄
        self.pose_history = deque(maxlen=60)

    def detect_shots(self, video_path, tracking_results):
        """
        檢測影片中的擊球動作

        Args:
            video_path: 影片路徑
            tracking_results: 網球追蹤結果

        Returns:
            dict: 擊球檢測結果，包含姿態分析數據
        """
        print("開始檢測擊球動作...")

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"無法開啟影片: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # 收集姿態數據 (如果啟用)
        pose_data_all = []
        pose_analysis_summary = {
            'avg_body_rotation': 0.0,
            'avg_arm_extension': 0.0,
            'avg_knee_bend': 0.0,
            'avg_balance_score': 0.0,
            'dominant_side': None,
            'pose_detected_frames': 0
        }

        if self.use_pose and self.pose_detector:
            print("📊 正在進行姿態分析...")
            frame_count = 0
            rotations = []
            extensions = []
            knee_bends = []
            balance_scores = []
            side_votes = {'left': 0, 'right': 0}

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                # 每隔幾幀做一次姿態檢測 (節省計算資源)
                if frame_count % 3 == 0:
                    poses = self.pose_detector.detect_pose(frame)

                    if poses:
                        pose = poses[0]  # 取主要人物
                        analysis = self.pose_detector.analyze_tennis_pose(pose, frame_height)

                        pose_data_all.append({
                            'frame': frame_count,
                            'timestamp': frame_count / fps,
                            'pose': pose,
                            'analysis': analysis
                        })

                        # 收集統計數據
                        if analysis.get('body_rotation'):
                            rotations.append(abs(analysis['body_rotation']))
                        if analysis.get('arm_extension'):
                            extensions.append(analysis['arm_extension'])
                        if analysis.get('knee_bend'):
                            knee_bends.append(analysis['knee_bend'])
                        if analysis.get('balance_score'):
                            balance_scores.append(analysis['balance_score'])
                        if analysis.get('dominant_side'):
                            side_votes[analysis['dominant_side']] += 1

                        self.pose_history.append(poses)

                frame_count += 1

                # 進度顯示
                if frame_count % 100 == 0:
                    progress = (frame_count / total_frames) * 100
                    print(f"   姿態分析進度: {progress:.1f}%")

            # 計算姿態統計
            pose_analysis_summary['pose_detected_frames'] = len(pose_data_all)
            if rotations:
                pose_analysis_summary['avg_body_rotation'] = round(np.mean(rotations), 2)
            if extensions:
                pose_analysis_summary['avg_arm_extension'] = round(np.mean(extensions), 2)
            if knee_bends:
                pose_analysis_summary['avg_knee_bend'] = round(np.mean(knee_bends), 2)
            if balance_scores:
                pose_analysis_summary['avg_balance_score'] = round(np.mean(balance_scores), 2)
            if side_votes['left'] > 0 or side_votes['right'] > 0:
                pose_analysis_summary['dominant_side'] = 'left' if side_votes['left'] > side_votes['right'] else 'right'

            print(f"✅ 姿態分析完成，共分析 {len(pose_data_all)} 幀")

            # 重置影片位置
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        cap.release()

        # 基於球軌跡檢測擊球
        shots = self.detect_shots_from_ball_trajectory(tracking_results, fps)

        # 如果有姿態數據，用姿態來增強擊球分類
        if self.use_pose and pose_data_all:
            shots = self.enhance_shots_with_pose(shots, pose_data_all, fps)

        # 計算擊球指標
        hit_metrics = self.calculate_hit_metrics(shots, tracking_results, frame_height, pose_data_all)

        print(f"✅ 檢測完成，找到 {len(shots)} 次擊球")

        return {
            'shots': shots,
            'total_shots': len(shots),
            'forehand_count': len([s for s in shots if s['type'] == 'forehand']),
            'backhand_count': len([s for s in shots if s['type'] == 'backhand']),
            'average_hit_height': hit_metrics['average_hit_height'],
            'average_hit_angle': hit_metrics['average_hit_angle'],
            'pose_analysis': pose_analysis_summary,
            'pose_available': self.use_pose
        }

    def detect_shots_from_ball_trajectory(self, tracking_results, fps):
        """基於網球軌跡檢測擊球點"""
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
                        'direction': (dx, dy)
                    })

        # 檢測速度突變（擊球點）
        for i in range(2, len(velocities) - 2):
            current_vel = velocities[i]['velocity']
            prev_vel = velocities[i-1]['velocity']
            next_vel = velocities[i+1]['velocity']

            # 速度突然增加 = 擊球
            if (current_vel > prev_vel * self.velocity_spike_ratio and
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
                    'direction': velocities[i]['direction'],
                    'ball_position': velocities[i]['position']
                })

        return self.filter_duplicate_shots(shots)

    def classify_shot_simple(self, velocity_data):
        """基於球方向的簡化擊球分類"""
        direction = velocity_data['direction']

        if direction[0] > abs(direction[1]) * 0.5:
            return 'forehand'
        else:
            return 'backhand'

    def enhance_shots_with_pose(self, shots, pose_data_all, fps):
        """使用姿態數據增強擊球分類"""
        if not pose_data_all:
            return shots

        enhanced_shots = []

        for shot in shots:
            shot_frame = shot['frame']

            # 找到最接近擊球時刻的姿態數據
            closest_pose = None
            min_diff = float('inf')

            for pd in pose_data_all:
                diff = abs(pd['frame'] - shot_frame)
                if diff < min_diff:
                    min_diff = diff
                    closest_pose = pd

            if closest_pose and min_diff < 10:  # 10幀內視為有效
                analysis = closest_pose['analysis']

                # 基於姿態分析重新分類擊球類型
                if analysis.get('dominant_side'):
                    # 使用身體旋轉和手臂位置來判斷正反手
                    rotation = analysis.get('body_rotation', 0)
                    dominant = analysis['dominant_side']

                    # 正手：身體旋轉方向與慣用手一致
                    # 反手：身體旋轉方向與慣用手相反
                    if dominant == 'right':
                        if rotation > 10:  # 身體右轉
                            shot['type'] = 'forehand'
                        elif rotation < -10:  # 身體左轉
                            shot['type'] = 'backhand'
                    else:
                        if rotation < -10:
                            shot['type'] = 'forehand'
                        elif rotation > 10:
                            shot['type'] = 'backhand'

                # 添加姿態相關數據
                shot['pose_analysis'] = {
                    'body_rotation': analysis.get('body_rotation', 0),
                    'arm_extension': analysis.get('arm_extension', 0),
                    'wrist_height': analysis.get('wrist_height', 0),
                    'balance_score': analysis.get('balance_score', 0),
                    'swing_angle': analysis.get('swing_angle', 0)
                }

                # 計算姿態評分
                shot['pose_score'] = self.calculate_pose_score(analysis)

            enhanced_shots.append(shot)

        return enhanced_shots

    def calculate_pose_score(self, analysis):
        """計算姿態評分 (0-100)"""
        score = 50  # 基礎分

        # 身體旋轉 (適度旋轉加分)
        rotation = abs(analysis.get('body_rotation', 0))
        if 15 <= rotation <= 45:
            score += 15
        elif 10 <= rotation <= 60:
            score += 10

        # 手臂伸展 (適度伸展加分)
        extension = analysis.get('arm_extension', 0)
        if 0.7 <= extension <= 1.2:
            score += 15
        elif 0.5 <= extension <= 1.4:
            score += 10

        # 平衡分數
        balance = analysis.get('balance_score', 0)
        score += balance * 0.2  # 最高加 20 分

        return min(100, max(0, round(score, 1)))

    def filter_duplicate_shots(self, shots):
        """過濾重複擊球"""
        if not shots:
            return shots

        filtered = []
        shots.sort(key=lambda x: x['frame'])

        for shot in shots:
            is_duplicate = False
            for existing in filtered:
                if abs(shot['frame'] - existing['frame']) < self.shot_window:
                    is_duplicate = True
                    break

            if not is_duplicate:
                filtered.append(shot)

        return filtered

    def calculate_hit_metrics(self, shots, tracking_results, frame_height, pose_data_all=None):
        """計算擊球指標"""
        hit_y_coords = []
        hit_angles = []
        wrist_heights = []

        ball_positions = {
            f['frame_number']: f['detections'][0]['center']
            for f in tracking_results.get('ball_positions', [])
            if f.get('detections')
        }

        for shot in shots:
            frame_num = shot.get('ball_contact_frame')

            # 從球位置計算
            if frame_num in ball_positions and frame_height > 0:
                y_coord = ball_positions[frame_num][1]
                hit_y_coords.append(y_coord)

                if 'direction' in shot and shot['direction'][0] != 0:
                    dx, dy = shot['direction']
                    angle_deg = math.degrees(math.atan2(-dy, dx))
                    hit_angles.append(angle_deg)

            # 從姿態數據計算手腕高度
            if shot.get('pose_analysis'):
                wrist_h = shot['pose_analysis'].get('wrist_height', 0)
                if wrist_h > 0:
                    wrist_heights.append(wrist_h)

        # 計算平均高度
        if wrist_heights:
            # 優先使用姿態數據的手腕高度
            hit_height_m = round(np.mean(wrist_heights) * 1.5, 2)  # 轉換為約略公尺
        elif hit_y_coords and frame_height > 0:
            avg_y = np.mean(hit_y_coords)
            hit_height_m = round(1.5 * (frame_height - avg_y) / frame_height, 2)
        else:
            hit_height_m = 0.0

        # 計算平均角度
        if hit_angles:
            avg_angle_deg = round(np.mean(hit_angles), 2)
        else:
            avg_angle_deg = 0.0

        return {
            'average_hit_height': hit_height_m,
            'average_hit_angle': avg_angle_deg
        }
