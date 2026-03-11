"""
MediaPipe Hands 手部偵測模組
偵測手部 21 個關鍵點，計算球拍方向和拍面朝向

MediaPipe Hand Landmarks (21 points):
  0: WRIST
  1-4: THUMB (CMC, MCP, IP, TIP)
  5-8: INDEX_FINGER (MCP, PIP, DIP, TIP)
  9-12: MIDDLE_FINGER (MCP, PIP, DIP, TIP)
  13-16: RING_FINGER (MCP, PIP, DIP, TIP)
  17-20: PINKY (MCP, PIP, DIP, TIP)
"""

import os
import numpy as np
import cv2
import urllib.request
from typing import Optional, Dict, List

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


MODEL_URL = 'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task'
MODEL_FILENAME = 'hand_landmarker.task'


class HandDetector:
    def __init__(self, model_dir: Optional[str] = None, max_hands: int = 1):
        if model_dir is None:
            model_dir = os.path.dirname(os.path.abspath(__file__))

        self.model_path = os.path.join(model_dir, MODEL_FILENAME)
        self._ensure_model()

        base_options = python.BaseOptions(model_asset_path=self.model_path)
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            num_hands=max_hands,
            min_hand_detection_confidence=0.3,
            min_hand_presence_confidence=0.3,
            min_tracking_confidence=0.3,
        )
        self.detector = vision.HandLandmarker.create_from_options(options)

    def _ensure_model(self):
        if not os.path.exists(self.model_path):
            print(f'Downloading hand landmarker model...')
            urllib.request.urlretrieve(MODEL_URL, self.model_path)
            print(f'Downloaded: {self.model_path} ({os.path.getsize(self.model_path) / 1024 / 1024:.1f} MB)')

    def detect(self, frame_bgr: np.ndarray) -> Optional[Dict]:
        """
        偵測手部並計算球拍方向

        Args:
            frame_bgr: OpenCV BGR 影像

        Returns:
            dict with:
              - landmarks_2d: (21, 2) 像素座標
              - handedness: 'Left' or 'Right'
              - racket_dir_2d: (2,) 拍柄方向 (2D 像素空間, 歸一化)
              - palm_normal_indicator: (2,) 手掌面朝向指示 (2D)
              - wrist_2d: (2,) 手腕像素座標
            or None if no hand detected
        """
        h, w = frame_bgr.shape[:2]
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

        result = self.detector.detect(mp_image)

        if not result.hand_landmarks or len(result.hand_landmarks) == 0:
            return None

        # 取第一隻偵測到的手
        landmarks = result.hand_landmarks[0]
        handedness = result.handedness[0][0].category_name  # 'Left' or 'Right'

        # 轉換為像素座標 (21, 2)
        landmarks_2d = np.array([[lm.x * w, lm.y * h] for lm in landmarks], dtype=np.float32)

        # 計算球拍方向和手掌朝向
        wrist = landmarks_2d[0]        # WRIST
        index_mcp = landmarks_2d[5]    # INDEX_FINGER_MCP
        middle_mcp = landmarks_2d[9]   # MIDDLE_FINGER_MCP
        middle_tip = landmarks_2d[12]  # MIDDLE_FINGER_TIP
        pinky_mcp = landmarks_2d[17]   # PINKY_MCP

        # 拍柄方向: 中指尖 → 從手腕延伸的方向
        finger_dir = middle_tip - wrist
        finger_len = np.linalg.norm(finger_dir)
        if finger_len > 1:
            finger_dir = finger_dir / finger_len
        else:
            finger_dir = np.array([0, -1], dtype=np.float32)

        # 手掌寬度方向: index_mcp → pinky_mcp
        palm_width = pinky_mcp - index_mcp
        palm_width_len = np.linalg.norm(palm_width)
        if palm_width_len > 1:
            palm_width = palm_width / palm_width_len
        else:
            palm_width = np.array([1, 0], dtype=np.float32)

        return {
            'landmarks_2d': landmarks_2d.tolist(),
            'handedness': handedness,
            'racket_dir_2d': finger_dir.tolist(),
            'palm_width_dir_2d': palm_width.tolist(),
            'wrist_2d': wrist.tolist(),
            'middle_tip_2d': middle_tip.tolist(),
            'index_mcp_2d': index_mcp.tolist(),
            'pinky_mcp_2d': pinky_mcp.tolist(),
        }

    def _crop_around_wrist(self, frame_bgr: np.ndarray, wrist_xy: np.ndarray,
                           crop_size: int = 300) -> Optional[tuple]:
        """
        以手腕為中心裁切放大區域，提高手部偵測率

        Returns:
            (cropped_frame, offset_x, offset_y, scale) or None
        """
        h, w = frame_bgr.shape[:2]
        cx, cy = int(wrist_xy[0]), int(wrist_xy[1])

        # 裁切框: 以手腕為中心，crop_size x crop_size
        half = crop_size // 2
        x1 = max(0, cx - half)
        y1 = max(0, cy - half)
        x2 = min(w, cx + half)
        y2 = min(h, cy + half)

        if x2 - x1 < 50 or y2 - y1 < 50:
            return None

        crop = frame_bgr[y1:y2, x1:x2].copy()

        # 放大到 MediaPipe 更容易偵測的尺寸
        target = 480
        scale = target / max(crop.shape[0], crop.shape[1])
        if scale > 1:
            crop = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
        else:
            scale = 1.0

        return crop, x1, y1, scale

    def detect_racket_hand(self, frame_bgr: np.ndarray,
                           body_wrist_right: Optional[np.ndarray] = None,
                           body_wrist_left: Optional[np.ndarray] = None,
                           body_shoulder_right: Optional[np.ndarray] = None,
                           body_shoulder_left: Optional[np.ndarray] = None) -> Optional[Dict]:
        """
        偵測持拍手的手部關鍵點
        策略: 先用 YOLO 的手腕座標裁切放大手部區域，再送 MediaPipe 偵測

        Args:
            frame_bgr: BGR 影像
            body_wrist_right/left: 手腕 2D 像素座標 (from YOLO Pose)
            body_shoulder_right/left: 肩膀座標 (用於計算裁切大小)
        """
        h, w = frame_bgr.shape[:2]

        # 計算裁切大小: 用軀幹長度的比例
        crop_size = int(min(h, w) * 0.35)  # 預設用畫面的 35%
        if body_shoulder_right is not None and body_wrist_right is not None:
            arm_len = np.linalg.norm(np.array(body_shoulder_right) - np.array(body_wrist_right))
            crop_size = max(int(arm_len * 0.8), 150)  # 用手臂長度的 80%

        # 嘗試兩隻手腕，優先嘗試兩隻都偵測
        wrists_to_try = []
        if body_wrist_right is not None:
            wrists_to_try.append(('right', np.array(body_wrist_right)))
        if body_wrist_left is not None:
            wrists_to_try.append(('left', np.array(body_wrist_left)))

        for hand_label, wrist_pos in wrists_to_try:
            # 裁切手腕附近區域
            crop_result = self._crop_around_wrist(frame_bgr, wrist_pos, crop_size)
            if crop_result is None:
                continue

            crop_frame, off_x, off_y, scale = crop_result
            crop_h, crop_w = crop_frame.shape[:2]

            # 偵測裁切區域中的手部
            frame_rgb = cv2.cvtColor(crop_frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
            result = self.detector.detect(mp_image)

            if not result.hand_landmarks or len(result.hand_landmarks) == 0:
                continue

            # 偵測成功! 將座標映射回原始影像
            landmarks = result.hand_landmarks[0]
            handedness = result.handedness[0][0].category_name

            # MediaPipe 輸出的是 [0,1] 歸一化座標 → 裁切區域像素 → 原始影像像素
            landmarks_2d = np.array([
                [lm.x * crop_w / scale + off_x, lm.y * crop_h / scale + off_y]
                for lm in landmarks
            ], dtype=np.float32)

            wrist = landmarks_2d[0]
            index_mcp = landmarks_2d[5]
            middle_tip = landmarks_2d[12]
            pinky_mcp = landmarks_2d[17]

            finger_dir = middle_tip - wrist
            flen = np.linalg.norm(finger_dir)
            finger_dir = finger_dir / flen if flen > 1 else np.array([0, -1], dtype=np.float32)

            palm_width = pinky_mcp - index_mcp
            plen = np.linalg.norm(palm_width)
            palm_width = palm_width / plen if plen > 1 else np.array([1, 0], dtype=np.float32)

            return {
                'landmarks_2d': landmarks_2d.tolist(),
                'handedness': handedness,
                'matched_hand': hand_label,
                'racket_dir_2d': finger_dir.tolist(),
                'palm_width_dir_2d': palm_width.tolist(),
                'wrist_2d': wrist.tolist(),
                'middle_tip_2d': middle_tip.tolist(),
                'index_mcp_2d': index_mcp.tolist(),
                'pinky_mcp_2d': pinky_mcp.tolist(),
            }

        return None  # 兩隻手都沒偵測到


def compute_racket_vectors_3d(hand_data: Dict, frame_width: float, frame_height: float,
                               wrist_3d: np.ndarray, elbow_3d: np.ndarray) -> Dict:
    """
    用 2D 手指方向 + 3D 前臂方向，計算球拍的 3D 方向

    核心思路：
    1. 從 2D 手指方向計算手指相對前臂的偏轉角
    2. 在 3D 空間中，將前臂方向繞垂直軸旋轉這個角度 = 球拍方向
    """
    wrist_3d = np.array(wrist_3d, dtype=np.float32)
    elbow_3d = np.array(elbow_3d, dtype=np.float32)

    # 3D 前臂方向
    forearm_3d = wrist_3d - elbow_3d
    forearm_len = np.linalg.norm(forearm_3d)
    if forearm_len < 1e-4:
        forearm_3d = np.array([0.0, -0.3, 0.0], dtype=np.float32)
        forearm_len = 0.3
    forearm_norm = forearm_3d / forearm_len

    # 2D: 前臂方向 (elbow→wrist) 和手指方向 (wrist→middle_tip)
    finger_dir_2d = np.array(hand_data['racket_dir_2d'], dtype=np.float32)  # 歸一化的

    # 2D 前臂方向 (用手部的手腕座標近似)
    wrist_2d = np.array(hand_data['wrist_2d'], dtype=np.float32)
    middle_tip_2d = np.array(hand_data['middle_tip_2d'], dtype=np.float32)

    # 手指方向的 2D 角度 (影像座標系, Y 向下)
    finger_vec = middle_tip_2d - wrist_2d
    finger_angle = np.arctan2(-finger_vec[1], finger_vec[0])  # 翻轉 Y 以匹配 3D 座標系

    # 前臂的 2D 角度 (用 3D 前臂投影到 XY 平面近似)
    forearm_angle = np.arctan2(forearm_norm[1], forearm_norm[0])

    # 手指相對於前臂的偏轉角度
    bend_angle = finger_angle - forearm_angle

    # 在 3D 空間中: 繞「垂直於前臂的軸」旋轉 bend_angle
    # 旋轉軸 = cross(forearm, world_up)
    world_up = np.array([0, 1, 0], dtype=np.float32)
    bend_axis = np.cross(forearm_norm, world_up)
    ba_len = np.linalg.norm(bend_axis)
    if ba_len < 0.1:
        bend_axis = np.cross(forearm_norm, np.array([0, 0, 1], dtype=np.float32))
        ba_len = np.linalg.norm(bend_axis)
    bend_axis = bend_axis / (ba_len + 1e-8)

    # Rodrigues 旋轉: v_rot = v*cos(θ) + (k×v)*sin(θ) + k*(k·v)*(1-cos(θ))
    c = np.cos(bend_angle)
    s = np.sin(bend_angle)
    racket_dir = (
        forearm_norm * c +
        np.cross(bend_axis, forearm_norm) * s +
        bend_axis * np.dot(bend_axis, forearm_norm) * (1 - c)
    )
    racket_dir = racket_dir / (np.linalg.norm(racket_dir) + 1e-8)

    # 拍面法線: 垂直於拍柄方向，大致朝 bend_axis 方向
    palm_width_2d = np.array(hand_data['palm_width_dir_2d'], dtype=np.float32)
    # 用 palm_width 的 2D 角度做微調
    palm_normal = bend_axis.copy()
    side_axis = np.cross(racket_dir, bend_axis)
    side_len = np.linalg.norm(side_axis)
    if side_len > 0.1:
        side_axis = side_axis / side_len
        palm_normal = bend_axis * 0.7 + side_axis * palm_width_2d[0] * 0.3
    palm_normal = palm_normal / (np.linalg.norm(palm_normal) + 1e-8)

    return {
        'racket_dir_3d': racket_dir.tolist(),
        'palm_normal_3d': palm_normal.tolist(),
    }
