"""
3D 姿態重建模組 - 使用多視角三角測量
從多個 2D 視角重建 3D 人體骨架座標
"""
import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import math

# COCO 17 關鍵點定義
KEYPOINT_NAMES = [
    'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
    'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
    'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
    'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
]

# 骨架連接 (用於 3D 可視化)
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


@dataclass
class Camera3D:
    """3D 空間中的相機"""
    name: str
    position: np.ndarray      # 相機位置 (x, y, z)
    look_at: np.ndarray       # 看向的點
    up: np.ndarray            # 上方向
    fov: float                # 視野角度 (度)
    image_width: int          # 影像寬度
    image_height: int         # 影像高度

    @property
    def intrinsic_matrix(self) -> np.ndarray:
        """計算內參矩陣 (簡化版)"""
        focal_length = self.image_width / (2 * math.tan(math.radians(self.fov / 2)))
        cx = self.image_width / 2
        cy = self.image_height / 2

        return np.array([
            [focal_length, 0, cx],
            [0, focal_length, cy],
            [0, 0, 1]
        ])

    @property
    def rotation_matrix(self) -> np.ndarray:
        """計算旋轉矩陣 (從世界座標到相機座標)"""
        # 計算相機座標系的三個軸
        forward = self.look_at - self.position
        forward = forward / np.linalg.norm(forward)

        right = np.cross(forward, self.up)
        right = right / np.linalg.norm(right)

        up = np.cross(right, forward)

        # 旋轉矩陣
        R = np.array([
            right,
            up,
            -forward
        ])

        return R

    @property
    def projection_matrix(self) -> np.ndarray:
        """計算投影矩陣 P = K[R|t]"""
        R = self.rotation_matrix
        t = -R @ self.position

        K = self.intrinsic_matrix
        Rt = np.hstack([R, t.reshape(3, 1)])

        return K @ Rt


class Pose3DReconstructor:
    """
    3D 姿態重建器
    使用多視角三角測量從 2D 關鍵點重建 3D 座標
    """

    def __init__(self, scene_center: Tuple[float, float, float] = (0, 0, 0)):
        """
        初始化重建器

        Args:
            scene_center: 場景中心點座標 (公尺)
        """
        self.scene_center = np.array(scene_center)
        self.cameras: Dict[str, Camera3D] = {}

    def add_camera_from_config(self, view_name: str, angle_degrees: float,
                               distance: float, height: float,
                               image_width: int = 1920, image_height: int = 1080,
                               fov: float = 60.0):
        """
        根據配置添加相機 (假設相機圍繞場景中心排列)

        Args:
            view_name: 視角名稱
            angle_degrees: 相機角度 (0=前, 90=右, 180=後, 270=左)
            distance: 到場景中心的距離
            height: 相機高度
            image_width: 影像寬度
            image_height: 影像高度
            fov: 視野角度
        """
        # 計算相機位置 (極座標轉直角座標)
        angle_rad = math.radians(angle_degrees)
        x = self.scene_center[0] + distance * math.sin(angle_rad)
        z = self.scene_center[2] + distance * math.cos(angle_rad)
        y = height

        camera = Camera3D(
            name=view_name,
            position=np.array([x, y, z]),
            look_at=self.scene_center + np.array([0, height * 0.5, 0]),  # 看向中心偏上
            up=np.array([0, 1, 0]),
            fov=fov,
            image_width=image_width,
            image_height=image_height
        )

        self.cameras[view_name] = camera
        print(f"📷 添加 3D 相機: {view_name} 位置=({x:.2f}, {y:.2f}, {z:.2f})")

    def setup_cameras_from_multiview_result(self, multiview_result: Dict,
                                            image_width: int = 1920,
                                            image_height: int = 1080):
        """
        從 MultiViewProcessor 的結果設置相機

        Args:
            multiview_result: MultiViewProcessor.process_multiview() 的返回值
        """
        for cam_config in multiview_result.get('camera_configs', []):
            self.add_camera_from_config(
                view_name=cam_config['view_name'],
                angle_degrees=cam_config['angle_degrees'],
                distance=cam_config['distance'],
                height=cam_config['height'],
                image_width=image_width,
                image_height=image_height
            )

    def triangulate_point(self, observations: Dict[str, Tuple[float, float]],
                          max_reprojection_error: float = 500.0) -> Optional[np.ndarray]:
        """
        使用簡化的多視角融合方法估算 3D 點
        由於沒有真正的相機校準，我們使用基於視角的啟發式方法

        Args:
            observations: 視角名稱 -> (u, v) 像素座標
            max_reprojection_error: 未使用 (保留參數兼容性)

        Returns:
            np.ndarray: 3D 點座標 (x, y, z) 或 None
        """
        if len(observations) < 2:
            return None

        # 收集有效的相機和觀測
        valid_cameras = []
        valid_points = []
        normalized_points = []

        for view_name, (u, v) in observations.items():
            if view_name not in self.cameras:
                continue
            camera = self.cameras[view_name]
            valid_cameras.append(view_name)
            valid_points.append((u, v))

            # 將像素座標正規化到 [-1, 1] 範圍
            u_norm = (u - camera.image_width / 2) / (camera.image_width / 2)
            v_norm = (v - camera.image_height / 2) / (camera.image_height / 2)
            normalized_points.append((u_norm, v_norm))

        if len(valid_cameras) < 2:
            return None

        # 使用簡化的多視角融合
        # 根據每個視角的角度，將 2D 座標投影到 3D 空間
        x_estimates = []
        y_estimates = []
        z_estimates = []

        for i, view_name in enumerate(valid_cameras):
            camera = self.cameras[view_name]
            u_norm, v_norm = normalized_points[i]

            # 獲取相機角度
            cam_pos = camera.position
            angle_rad = math.atan2(cam_pos[0], cam_pos[2])

            # 根據視角角度估算 X 和 Z 座標
            # 正面視角 (0°): u_norm 對應 X
            # 側面視角 (90°): u_norm 對應 Z
            cos_a = math.cos(angle_rad)
            sin_a = math.sin(angle_rad)

            # X 座標估算 (水平位置)
            x_contrib = -u_norm * cos_a  # 正面視角的水平位置
            z_contrib = u_norm * sin_a   # 轉換到深度

            x_estimates.append(x_contrib)
            z_estimates.append(z_contrib)

            # Y 座標估算 (垂直位置，所有視角都一樣)
            y_estimates.append(-v_norm)

        # 平均所有估算
        x = np.mean(x_estimates)
        y = np.mean(y_estimates)  # Y 軸是垂直方向
        z = np.mean(z_estimates) if len(z_estimates) > 0 else 0

        # 縮放到合理範圍 (假設人物高度約 1.7 米)
        scale = 0.85  # 縮放因子，使結果在合理範圍內

        return np.array([x * scale, y * scale, z * scale])

    def compute_reprojection_error(self, point_3d: np.ndarray,
                                   observations: Dict[str, Tuple[float, float]]) -> float:
        """
        計算 3D 點的平均重投影誤差

        Args:
            point_3d: 3D 點座標
            observations: 視角名稱 -> (u, v) 像素座標

        Returns:
            float: 平均重投影誤差 (像素)
        """
        total_error = 0
        count = 0

        X_homogeneous = np.append(point_3d, 1)

        for view_name, (u, v) in observations.items():
            if view_name not in self.cameras:
                continue

            P = self.cameras[view_name].projection_matrix
            projected = P @ X_homogeneous

            if abs(projected[2]) < 1e-10:
                continue

            u_proj = projected[0] / projected[2]
            v_proj = projected[1] / projected[2]

            error = np.sqrt((u_proj - u)**2 + (v_proj - v)**2)
            total_error += error
            count += 1

        return total_error / count if count > 0 else float('inf')

    def reconstruct_pose_3d(self, frame_data: Dict,
                            confidence_threshold: float = 0.3) -> Optional[Dict]:
        """
        重建單幀的 3D 姿態

        Args:
            frame_data: 多視角幀數據 (來自 MultiViewProcessor)
            confidence_threshold: 關鍵點信心閾值

        Returns:
            Dict: 3D 姿態數據 或 None
        """
        views = frame_data.get('views', {})

        if len(views) < 2:
            return None

        # 對每個關鍵點進行三角測量
        keypoints_3d = []
        valid_count = 0

        for kp_idx in range(17):  # COCO 17 關鍵點
            observations = {}

            for view_name, view_data in views.items():
                if view_data is None:
                    continue

                keypoints = view_data.get('keypoints', [])
                if kp_idx >= len(keypoints):
                    continue

                kp = keypoints[kp_idx]
                if len(kp) >= 3:
                    x, y, conf = kp[0], kp[1], kp[2]
                else:
                    x, y, conf = kp[0], kp[1], 1.0

                if conf >= confidence_threshold:
                    observations[view_name] = (x, y)

            # 三角測量
            if len(observations) >= 2:
                point_3d = self.triangulate_point(observations)
                if point_3d is not None:
                    keypoints_3d.append({
                        'index': kp_idx,
                        'name': KEYPOINT_NAMES[kp_idx],
                        'position': point_3d.tolist(),
                        'num_views': len(observations)
                    })
                    valid_count += 1
                else:
                    keypoints_3d.append({
                        'index': kp_idx,
                        'name': KEYPOINT_NAMES[kp_idx],
                        'position': None,
                        'num_views': 0
                    })
            else:
                keypoints_3d.append({
                    'index': kp_idx,
                    'name': KEYPOINT_NAMES[kp_idx],
                    'position': None,
                    'num_views': len(observations)
                })

        if valid_count < 3:  # 至少需要 3 個有效點 (降低門檻以便除錯)
            # Debug: 輸出第一幀的資訊
            if frame_data.get('frame_number', 0) == 0:
                print(f"   [DEBUG] 第 0 幀: valid_count={valid_count}, keypoints_3d 數量={len(keypoints_3d)}")
            return None

        return {
            'frame_number': frame_data.get('frame_number', 0),
            'timestamp': frame_data.get('timestamp', 0),
            'keypoints_3d': keypoints_3d,
            'valid_keypoints': valid_count,
            'skeleton_connections': SKELETON_CONNECTIONS
        }

    def reconstruct_sequence(self, multiview_result: Dict,
                             progress_callback: Optional[callable] = None) -> Dict:
        """
        重建整個序列的 3D 姿態

        Args:
            multiview_result: MultiViewProcessor.process_multiview() 的返回值
            progress_callback: 進度回調

        Returns:
            Dict: 3D 重建結果
        """
        # 設置相機
        video_info = multiview_result.get('video_info', {})
        self.setup_cameras_from_multiview_result(
            multiview_result,
            image_width=video_info.get('width', 1920),
            image_height=video_info.get('height', 1080)
        )

        frames_data = multiview_result.get('frames', [])
        total_frames = len(frames_data)

        print(f"🔧 開始 3D 重建 ({total_frames} 幀)...")

        poses_3d = []
        valid_frames = 0

        for idx, frame_data in enumerate(frames_data):
            pose_3d = self.reconstruct_pose_3d(frame_data)

            if pose_3d is not None:
                poses_3d.append(pose_3d)
                valid_frames += 1
            else:
                poses_3d.append({
                    'frame_number': frame_data.get('frame_number', idx),
                    'timestamp': frame_data.get('timestamp', 0),
                    'keypoints_3d': None,
                    'valid_keypoints': 0
                })

            if progress_callback:
                progress_callback(idx, total_frames)

            if idx % 30 == 0:
                progress = (idx / total_frames) * 100
                print(f"   3D 重建進度: {progress:.1f}%")

        result = {
            'video_info': video_info,
            'camera_configs': [
                {
                    'name': cam.name,
                    'position': cam.position.tolist(),
                    'look_at': cam.look_at.tolist()
                }
                for cam in self.cameras.values()
            ],
            'poses_3d': poses_3d,
            'skeleton_connections': SKELETON_CONNECTIONS,
            'keypoint_names': KEYPOINT_NAMES,
            'statistics': {
                'total_frames': total_frames,
                'valid_frames': valid_frames,
                'reconstruction_rate': valid_frames / total_frames if total_frames > 0 else 0
            }
        }

        print(f"✅ 3D 重建完成: {valid_frames}/{total_frames} 幀成功重建")

        return result

    def normalize_pose(self, pose_3d: Dict) -> Dict:
        """
        正規化 3D 姿態 (以髖部中心為原點，標準化尺度)

        Args:
            pose_3d: 單幀 3D 姿態數據

        Returns:
            Dict: 正規化後的姿態
        """
        keypoints = pose_3d.get('keypoints_3d', [])
        if not keypoints:
            return pose_3d

        # 獲取有效的 3D 點
        valid_points = []
        for kp in keypoints:
            if kp.get('position') is not None:
                valid_points.append(np.array(kp['position']))

        if len(valid_points) < 2:
            return pose_3d

        # 計算中心 (使用髖部中點)
        left_hip_idx = 11
        right_hip_idx = 12

        left_hip = keypoints[left_hip_idx].get('position')
        right_hip = keypoints[right_hip_idx].get('position')

        if left_hip and right_hip:
            center = (np.array(left_hip) + np.array(right_hip)) / 2
        else:
            center = np.mean(valid_points, axis=0)

        # 計算尺度 (使用軀幹長度)
        shoulder_idx = [5, 6]  # 左右肩
        hip_idx = [11, 12]     # 左右髖

        shoulders = [keypoints[i].get('position') for i in shoulder_idx]
        hips = [keypoints[i].get('position') for i in hip_idx]

        if all(s is not None for s in shoulders) and all(h is not None for h in hips):
            shoulder_center = (np.array(shoulders[0]) + np.array(shoulders[1])) / 2
            hip_center = (np.array(hips[0]) + np.array(hips[1])) / 2
            torso_length = np.linalg.norm(shoulder_center - hip_center)
            scale = 1.0 / torso_length if torso_length > 0.1 else 1.0
        else:
            scale = 1.0

        # 正規化
        normalized_keypoints = []
        for kp in keypoints:
            if kp.get('position') is not None:
                pos = np.array(kp['position'])
                normalized_pos = (pos - center) * scale
                normalized_keypoints.append({
                    **kp,
                    'position': normalized_pos.tolist()
                })
            else:
                normalized_keypoints.append(kp)

        return {
            **pose_3d,
            'keypoints_3d': normalized_keypoints,
            'normalized': True,
            'center': center.tolist(),
            'scale': scale
        }


def reconstruct_3d_from_multiview(multiview_result: Dict,
                                   normalize: bool = True) -> Dict:
    """
    從多視角處理結果重建 3D 姿態的便捷函數

    Args:
        multiview_result: MultiViewProcessor.process_multiview() 的返回值
        normalize: 是否正規化姿態

    Returns:
        Dict: 3D 重建結果
    """
    reconstructor = Pose3DReconstructor()
    result = reconstructor.reconstruct_sequence(multiview_result)

    if normalize:
        normalized_poses = []
        for pose in result.get('poses_3d', []):
            if pose.get('keypoints_3d') is not None:
                normalized_poses.append(reconstructor.normalize_pose(pose))
            else:
                normalized_poses.append(pose)
        result['poses_3d'] = normalized_poses

    return result
