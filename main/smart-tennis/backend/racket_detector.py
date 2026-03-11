"""
網球拍偵測模組
用 YOLO 物件偵測 (COCO class 38 = tennis racket) 定位球拍位置和方向

方法：
1. YOLO 偵測 tennis racket bounding box
2. 在 bbox 區域內用 OpenCV 取得球拍的旋轉角度 (主軸方向)
3. 球拍中心 + 方向角度 → 映射到 3D 空間
"""

import numpy as np
import cv2
from ultralytics import YOLO
from typing import Optional, Dict, Tuple


# COCO class ID for tennis racket
TENNIS_RACKET_CLASS = 38


class RacketDetector:
    def __init__(self, model_path: str = 'yolov8n.pt'):
        self.model = YOLO(model_path)

    def detect(self, frame_bgr: np.ndarray, min_conf: float = 0.3) -> Optional[Dict]:
        """
        偵測畫面中的網球拍

        Returns:
            dict with:
              - center_2d: (2,) 球拍中心像素座標
              - angle_deg: float, 球拍長軸角度 (度, 0=水平, 90=垂直)
              - direction_2d: (2,) 球拍長軸方向 (歸一化, 影像座標系)
              - bbox: (4,) [x1, y1, x2, y2]
              - confidence: float
            or None
        """
        results = self.model(frame_bgr, classes=[TENNIS_RACKET_CLASS], conf=min_conf, verbose=False)

        if not results or len(results[0].boxes) == 0:
            return None

        # 取信心最高的球拍
        boxes = results[0].boxes
        best_idx = boxes.conf.argmax().item()
        bbox = boxes.xyxy[best_idx].cpu().numpy().astype(int)  # [x1, y1, x2, y2]
        conf = boxes.conf[best_idx].item()

        x1, y1, x2, y2 = bbox
        center = np.array([(x1 + x2) / 2.0, (y1 + y2) / 2.0], dtype=np.float32)

        # 在 bbox 區域用 OpenCV 取得旋轉角度
        angle, direction = self._get_racket_angle(frame_bgr, bbox)

        return {
            'center_2d': center.tolist(),
            'angle_deg': float(angle),
            'direction_2d': direction.tolist(),
            'bbox': bbox.tolist(),
            'confidence': float(conf),
        }

    def _get_racket_angle(self, frame_bgr: np.ndarray, bbox: np.ndarray) -> Tuple[float, np.ndarray]:
        """
        用 OpenCV 在 bbox 區域內偵測球拍的旋轉角度

        方法: 邊緣偵測 → 找輪廓 → minAreaRect → 長軸方向
        """
        x1, y1, x2, y2 = bbox
        h, w = frame_bgr.shape[:2]

        # 擴大 bbox 一點
        pad = 10
        x1 = max(0, x1 - pad)
        y1 = max(0, y1 - pad)
        x2 = min(w, x2 + pad)
        y2 = min(h, y2 + pad)

        crop = frame_bgr[y1:y2, x1:x2]
        if crop.size == 0:
            return 0.0, np.array([0.0, -1.0], dtype=np.float32)

        # 灰度 + 邊緣偵測
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)

        # 找輪廓
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            # 回退: 用 bbox 長寬比估計角度
            bw = x2 - x1
            bh = y2 - y1
            angle = np.degrees(np.arctan2(bh, bw))
            direction = np.array([bw, -bh], dtype=np.float32)
            direction = direction / (np.linalg.norm(direction) + 1e-8)
            return angle, direction

        # 合併所有輪廓點
        all_points = np.vstack(contours)

        # minAreaRect: 返回 ((cx,cy), (width,height), angle)
        rect = cv2.minAreaRect(all_points)
        rect_size = rect[1]  # (width, height)
        rect_angle = rect[2]  # 度

        # minAreaRect 的 angle 是 [-90, 0)
        # 長軸方向: width > height 時 angle 就是長軸角度; 否則加 90
        if rect_size[0] < rect_size[1]:
            rect_angle += 90

        # 轉為方向向量 (影像座標系: X 右, Y 下)
        angle_rad = np.radians(rect_angle)
        direction = np.array([np.cos(angle_rad), np.sin(angle_rad)], dtype=np.float32)

        return rect_angle, direction


def compute_racket_3d_from_detection(
    racket_data: Dict,
    wrist_2d: np.ndarray,
    wrist_3d: np.ndarray,
    elbow_3d: np.ndarray,
    frame_width: float,
    frame_height: float,
    pixel_to_3d_scale: float,
) -> Dict:
    """
    將 2D 球拍偵測結果轉換為 3D 球拍方向

    策略:
    1. 球拍中心相對於手腕的 2D 偏移 → 用比例因子轉 3D 偏移
    2. 球拍 2D 方向角 → 映射到 3D 前臂座標系
    """
    wrist_3d = np.array(wrist_3d, dtype=np.float32)
    elbow_3d = np.array(elbow_3d, dtype=np.float32)
    wrist_2d = np.array(wrist_2d, dtype=np.float32)

    racket_center_2d = np.array(racket_data['center_2d'], dtype=np.float32)
    racket_dir_2d = np.array(racket_data['direction_2d'], dtype=np.float32)

    # 3D 前臂方向
    forearm = wrist_3d - elbow_3d
    forearm_len = np.linalg.norm(forearm)
    if forearm_len < 1e-4:
        forearm = np.array([0.0, -0.3, 0.0], dtype=np.float32)
        forearm_len = 0.3
    forearm_norm = forearm / forearm_len

    # 球拍 2D 方向角 (影像座標系 Y 向下 → 翻轉為 3D Y 向上)
    dir_2d = racket_dir_2d.copy()
    dir_2d[1] = -dir_2d[1]  # Y 翻轉

    # 確保方向指向遠離手腕的方向
    offset_2d = racket_center_2d - wrist_2d
    if np.dot(offset_2d, racket_dir_2d) < 0:
        dir_2d = -dir_2d

    # 建立 3D 局部座標系
    world_up = np.array([0, 1, 0], dtype=np.float32)
    local_x = np.cross(forearm_norm, world_up)
    lx_len = np.linalg.norm(local_x)
    if lx_len < 0.1:
        local_x = np.cross(forearm_norm, np.array([0, 0, 1], dtype=np.float32))
        lx_len = np.linalg.norm(local_x)
    local_x = local_x / (lx_len + 1e-8)
    local_y = np.cross(local_x, forearm_norm)
    local_y = local_y / (np.linalg.norm(local_y) + 1e-8)

    # 將 2D 方向映射到 3D:
    # 2D X 分量 → 3D local_x (側向)
    # 2D Y 分量 → 3D forearm_norm (前臂延伸) + local_y (上下)
    racket_dir_3d = (
        forearm_norm * dir_2d[1] * 0.6 +
        local_x * dir_2d[0] * (-0.5) +  # X 翻轉 (鏡像修正)
        local_y * dir_2d[1] * 0.4
    )
    rd_len = np.linalg.norm(racket_dir_3d)
    if rd_len < 1e-6:
        racket_dir_3d = forearm_norm.copy()
    else:
        racket_dir_3d = racket_dir_3d / rd_len

    # 拍面法線: 垂直於球拍方向和前臂
    palm_normal = np.cross(racket_dir_3d, forearm_norm)
    pn_len = np.linalg.norm(palm_normal)
    if pn_len < 0.1:
        palm_normal = local_x.copy()
    else:
        palm_normal = palm_normal / pn_len

    return {
        'racket_dir_3d': racket_dir_3d.tolist(),
        'palm_normal_3d': palm_normal.tolist(),
    }
