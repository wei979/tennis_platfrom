import cv2
import numpy as np
# import mediapipe as mp  # 暫時註解，Python 3.13 不支援
from collections import deque
import math

class ShotDetector:
    def __init__(self):
        """
        初始化正反手檢測器
        """
        self.mp_pose = mp.solutions.pose
        self.mp_hands = mp.solutions.hands
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # 檢測參數
        self.swing_threshold = 0.3  # 揮拍動作閾值
        self.shot_window = 15  # 檢測視窗大小（幀數）
        self.velocity_history = deque(maxlen=self.shot_window)
        
    def detect_shots(self, video_path, tracking_results):
        """
        檢測影片中的正反手擊球
        """
        print("開始檢測正反手擊球...")
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"無法開啟影片: {video_path}")
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        shots = []
        frame_count = 0
        
        # 姿態追蹤歷史
        pose_history = deque(maxlen=self.shot_window)
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # 檢測人體姿態
            pose_landmarks = self.detect_pose(frame)
            
            if pose_landmarks:
                # 分析姿態
                pose_data = self.extract_pose_data(pose_landmarks, frame.shape)
                pose_history.append({
                    'frame': frame_count,
                    'timestamp': frame_count / fps,
                    'pose_data': pose_data
                })
                
                # 檢測擊球動作
                if len(pose_history) >= self.shot_window:
                    shot = self.detect_shot_in_window(pose_history, tracking_results, frame_count)
                    if shot:
                        shots.append(shot)
            
            frame_count += 1
            
            if frame_count % 60 == 0:
                progress = (frame_count / total_frames) * 100
                print(f"檢測進度: {progress:.1f}%")
        
        cap.release()
        
        # 過濾重複檢測
        filtered_shots = self.filter_duplicate_shots(shots)
        
        print(f"檢測完成，找到 {len(filtered_shots)} 次擊球")
        
        return {
            'shots': filtered_shots,
            'total_shots': len(filtered_shots),
            'forehand_count': len([s for s in filtered_shots if s['type'] == 'forehand']),
            'backhand_count': len([s for s in filtered_shots if s['type'] == 'backhand'])
        }
    
    def detect_pose(self, frame):
        """
        檢測人體姿態
        """
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(rgb_frame)
        return results.pose_landmarks
    
    def extract_pose_data(self, landmarks, frame_shape):
        """
        提取姿態關鍵數據
        """
        height, width = frame_shape[:2]
        
        # 關鍵點
        left_shoulder = landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_SHOULDER]
        right_shoulder = landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_SHOULDER]
        left_elbow = landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_ELBOW]
        right_elbow = landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_ELBOW]
        left_wrist = landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_WRIST]
        right_wrist = landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_WRIST]
        left_hip = landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_HIP]
        right_hip = landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_HIP]
        
        # 轉換到像素座標
        def to_pixel(landmark):
            return (landmark.x * width, landmark.y * height)
        
        return {
            'left_shoulder': to_pixel(left_shoulder),
            'right_shoulder': to_pixel(right_shoulder),
            'left_elbow': to_pixel(left_elbow),
            'right_elbow': to_pixel(right_elbow),
            'left_wrist': to_pixel(left_wrist),
            'right_wrist': to_pixel(right_wrist),
            'left_hip': to_pixel(left_hip),
            'right_hip': to_pixel(right_hip),
            'body_center': (
                (left_shoulder.x + right_shoulder.x) * width / 2,
                (left_shoulder.y + right_shoulder.y) * height / 2
            )
        }
    
    def detect_shot_in_window(self, pose_history, tracking_results, current_frame):
        """
        在時間窗口內檢測擊球動作
        """
        if len(pose_history) < self.shot_window:
            return None
        
        # 計算手臂速度變化
        arm_velocities = self.calculate_arm_velocities(pose_history)
        
        # 檢測是否有明顯的揮拍動作
        swing_detected, swing_side = self.detect_swing_motion(arm_velocities)
        
        if not swing_detected:
            return None
        
        # 檢查是否有網球接觸
        ball_contact = self.check_ball_contact(pose_history, tracking_results)
        
        if ball_contact:
            shot_type = self.classify_shot_type(pose_history, swing_side)
            
            return {
                'frame': current_frame,
                'timestamp': pose_history[-1]['timestamp'],
                'type': shot_type,
                'side': swing_side,
                'confidence': self.calculate_shot_confidence(arm_velocities, ball_contact),
                'ball_contact_frame': ball_contact['frame'],
                'swing_velocity': max(arm_velocities) if arm_velocities else 0
            }
        
        return None
    
    def calculate_arm_velocities(self, pose_history):
        """
        計算手臂移動速度
        """
        velocities = []
        
        for i in range(1, len(pose_history)):
            prev_pose = pose_history[i-1]['pose_data']
            curr_pose = pose_history[i]['pose_data']
            dt = pose_history[i]['timestamp'] - pose_history[i-1]['timestamp']
            
            if dt > 0:
                # 計算左右手腕速度
                left_vel = self.calculate_point_velocity(
                    prev_pose['left_wrist'], curr_pose['left_wrist'], dt
                )
                right_vel = self.calculate_point_velocity(
                    prev_pose['right_wrist'], curr_pose['right_wrist'], dt
                )
                
                velocities.append({
                    'left_wrist': left_vel,
                    'right_wrist': right_vel,
                    'max_velocity': max(left_vel, right_vel)
                })
        
        return velocities
    
    def calculate_point_velocity(self, point1, point2, dt):
        """
        計算兩點間的速度
        """
        dx = point2[0] - point1[0]
        dy = point2[1] - point1[1]
        distance = math.sqrt(dx**2 + dy**2)
        return distance / dt
    
    def detect_swing_motion(self, arm_velocities):
        """
        檢測揮拍動作
        """
        if not arm_velocities:
            return False, None
        
        # 檢查速度峰值
        max_left_vel = max([v['left_wrist'] for v in arm_velocities])
        max_right_vel = max([v['right_wrist'] for v in arm_velocities])
        
        if max_left_vel > self.swing_threshold or max_right_vel > self.swing_threshold:
            swing_side = 'left' if max_left_vel > max_right_vel else 'right'
            return True, swing_side
        
        return False, None
    
    def check_ball_contact(self, pose_history, tracking_results):
        """
        檢查是否有球拍接觸網球
        """
        # 獲取當前時間窗口的網球位置
        window_start_frame = pose_history[0]['frame']
        window_end_frame = pose_history[-1]['frame']
        
        for frame_data in tracking_results['ball_positions']:
            frame_num = frame_data['frame_number']
            if window_start_frame <= frame_num <= window_end_frame and frame_data['detections']:
                # 檢查網球是否接近手腕位置
                for detection in frame_data['detections']:
                    ball_pos = detection['center']
                    
                    # 找到對應幀的姿態數據
                    for pose_frame in pose_history:
                        if pose_frame['frame'] == frame_num:
                            pose_data = pose_frame['pose_data']
                            
                            # 計算距離
                            left_dist = self.calculate_distance(ball_pos, pose_data['left_wrist'])
                            right_dist = self.calculate_distance(ball_pos, pose_data['right_wrist'])
                            
                            # 接觸閾值（像素）
                            contact_threshold = 100
                            
                            if left_dist < contact_threshold or right_dist < contact_threshold:
                                return {
                                    'frame': frame_num,
                                    'ball_position': ball_pos,
                                    'distance': min(left_dist, right_dist)
                                }
        
        return None
    
    def calculate_distance(self, point1, point2):
        """
        計算兩點間距離
        """
        return math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)
    
    def classify_shot_type(self, pose_history, swing_side):
        """
        分類擊球類型（正手/反手）
        """
        # 簡化的分類邏輯
        # 可以根據身體姿態、手臂位置等進行更精確的分類
        
        middle_frame = pose_history[len(pose_history) // 2]
        pose_data = middle_frame['pose_data']
        
        body_center_x = pose_data['body_center'][0]
        
        if swing_side == 'right':
            wrist_x = pose_data['right_wrist'][0]
            if wrist_x > body_center_x:
                return 'forehand'  # 右手在身體右側，可能是正手
            else:
                return 'backhand'  # 右手在身體左側，可能是反手
        else:
            wrist_x = pose_data['left_wrist'][0]
            if wrist_x < body_center_x:
                return 'forehand'  # 左手在身體左側，可能是正手
            else:
                return 'backhand'  # 左手在身體右側，可能是反手
    
    def calculate_shot_confidence(self, arm_velocities, ball_contact):
        """
        計算擊球檢測的信心分數
        """
        velocity_confidence = min(max([v['max_velocity'] for v in arm_velocities]) / 1000, 1.0)
        contact_confidence = 1.0 - (ball_contact['distance'] / 200) if ball_contact else 0.0
        
        return (velocity_confidence + contact_confidence) / 2
    
    def filter_duplicate_shots(self, shots):
        """
        過濾重複的擊球檢測
        """
        if not shots:
            return shots
        
        filtered_shots = []
        shots.sort(key=lambda x: x['frame'])
        
        for shot in shots:
            # 檢查是否與已有的擊球太接近
            is_duplicate = False
            for existing_shot in filtered_shots:
                frame_diff = abs(shot['frame'] - existing_shot['frame'])
                if frame_diff < 30:  # 30幀內的檢測視為重複
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                filtered_shots.append(shot)
        
        return filtered_shots
