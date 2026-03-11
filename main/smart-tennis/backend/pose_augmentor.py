"""
PoseAug 風格姿態增強模組
基於 CVPR 2021 PoseAug 論文的概念實現
支援骨骼長度、角度、旋轉和位置的可微分增強

參考: https://github.com/jfzhang95/PoseAug
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import math

# COCO 17 關鍵點定義
KEYPOINT_NAMES = [
    'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
    'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
    'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
    'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
]

# 骨骼連接 (parent -> child)
SKELETON_BONES = [
    # 頭部
    (0, 1), (0, 2), (1, 3), (2, 4),  # nose to eyes to ears
    # 軀幹
    (5, 6),   # left_shoulder to right_shoulder
    (11, 12), # left_hip to right_hip
    (5, 11),  # left_shoulder to left_hip
    (6, 12),  # right_shoulder to right_hip
    # 左臂
    (5, 7), (7, 9),   # shoulder -> elbow -> wrist
    # 右臂
    (6, 8), (8, 10),  # shoulder -> elbow -> wrist
    # 左腿
    (11, 13), (13, 15),  # hip -> knee -> ankle
    # 右腿
    (12, 14), (14, 16),  # hip -> knee -> ankle
]

# 骨骼層級 (用於分層增強)
BONE_GROUPS = {
    'head': [(0, 1), (0, 2), (1, 3), (2, 4)],
    'torso': [(5, 6), (11, 12), (5, 11), (6, 12)],
    'left_arm': [(5, 7), (7, 9)],
    'right_arm': [(6, 8), (8, 10)],
    'left_leg': [(11, 13), (13, 15)],
    'right_leg': [(12, 14), (14, 16)],
}


@dataclass
class AugmentationConfig:
    """增強配置參數"""
    # 骨骼長度增強 (BL)
    bone_length_scale_range: Tuple[float, float] = (0.8, 1.2)
    bone_length_enabled: bool = True

    # 骨骼角度增強 (BA)
    bone_angle_range: float = 15.0  # 度
    bone_angle_enabled: bool = True

    # 全局旋轉增強 (RT)
    rotation_range: Tuple[float, float, float] = (30.0, 30.0, 30.0)  # x, y, z 度
    rotation_enabled: bool = True

    # 位置增強 (POS)
    translation_range: float = 0.3  # 米
    translation_enabled: bool = True

    # 隨機種子 (用於可重複實驗)
    random_seed: Optional[int] = None


class PoseAugmentor:
    """
    PoseAug 風格的 3D 姿態增強器
    實現四種主要增強:
    1. Bone Length (BL) - 骨骼長度縮放
    2. Bone Angle (BA) - 骨骼角度旋轉
    3. Rotation (RT) - 全局旋轉
    4. Position (POS) - 位置平移
    """

    def __init__(self, config: Optional[AugmentationConfig] = None):
        self.config = config or AugmentationConfig()
        if self.config.random_seed is not None:
            np.random.seed(self.config.random_seed)

    def augment_pose(self, keypoints_3d: np.ndarray,
                     augmentation_params: Optional[Dict] = None) -> np.ndarray:
        """
        對單個 3D 姿態進行增強

        Args:
            keypoints_3d: shape (17, 3) 的 3D 關鍵點
            augmentation_params: 可選的自定義增強參數

        Returns:
            增強後的 3D 關鍵點
        """
        if keypoints_3d is None or len(keypoints_3d) == 0:
            return keypoints_3d

        augmented = keypoints_3d.copy()
        params = augmentation_params or {}

        # 1. 骨骼長度增強
        if self.config.bone_length_enabled:
            scale = params.get('bone_length_scale',
                              np.random.uniform(*self.config.bone_length_scale_range))
            augmented = self._augment_bone_length(augmented, scale)

        # 2. 骨骼角度增強
        if self.config.bone_angle_enabled:
            angle = params.get('bone_angle',
                              np.random.uniform(-self.config.bone_angle_range,
                                               self.config.bone_angle_range))
            augmented = self._augment_bone_angle(augmented, angle)

        # 3. 全局旋轉增強
        if self.config.rotation_enabled:
            rotation = params.get('rotation', self._random_rotation())
            augmented = self._augment_rotation(augmented, rotation)

        # 4. 位置增強
        if self.config.translation_enabled:
            translation = params.get('translation', self._random_translation())
            augmented = self._augment_translation(augmented, translation)

        return augmented

    def _augment_bone_length(self, keypoints: np.ndarray, scale: float) -> np.ndarray:
        """
        骨骼長度增強 - 縮放骨骼長度

        基於 PoseAug 的 BL 增強:
        - 保持骨骼方向不變
        - 按比例縮放骨骼長度
        - 從根節點向外傳播
        """
        augmented = keypoints.copy()

        # 計算中心點 (hip center)
        hip_center = (keypoints[11] + keypoints[12]) / 2

        # 從中心向外縮放
        for i in range(len(keypoints)):
            direction = keypoints[i] - hip_center
            augmented[i] = hip_center + direction * scale

        return augmented

    def _augment_bone_angle(self, keypoints: np.ndarray, angle_deg: float) -> np.ndarray:
        """
        骨骼角度增強 - 旋轉關節角度

        基於 PoseAug 的 BA 增強:
        - 對特定關節進行小角度旋轉
        - 保持物理合理性
        """
        augmented = keypoints.copy()
        angle_rad = math.radians(angle_deg)

        # 對肘部和膝蓋應用角度變化
        joint_pairs = [
            (5, 7, 9),   # left arm: shoulder-elbow-wrist
            (6, 8, 10),  # right arm: shoulder-elbow-wrist
            (11, 13, 15), # left leg: hip-knee-ankle
            (12, 14, 16), # right leg: hip-knee-ankle
        ]

        for parent, joint, child in joint_pairs:
            if all(keypoints[i] is not None for i in [parent, joint, child]):
                # 計算旋轉軸 (垂直於骨骼平面)
                v1 = keypoints[joint] - keypoints[parent]
                v2 = keypoints[child] - keypoints[joint]

                # 旋轉 child 點
                rotation_axis = np.cross(v1, v2)
                if np.linalg.norm(rotation_axis) > 1e-6:
                    rotation_axis = rotation_axis / np.linalg.norm(rotation_axis)
                    R = self._rotation_matrix_from_axis_angle(rotation_axis, angle_rad)
                    relative_pos = augmented[child] - augmented[joint]
                    augmented[child] = augmented[joint] + R @ relative_pos

        return augmented

    def _augment_rotation(self, keypoints: np.ndarray,
                          rotation: Tuple[float, float, float]) -> np.ndarray:
        """
        全局旋轉增強

        基於 PoseAug 的 RT 增強:
        - 對整個骨架進行 3D 旋轉
        - 模擬不同的視角
        """
        augmented = keypoints.copy()

        # 計算中心點
        center = np.mean(keypoints, axis=0)

        # 構建旋轉矩陣 (Euler angles: x, y, z)
        rx, ry, rz = [math.radians(r) for r in rotation]
        Rx = self._rotation_matrix_x(rx)
        Ry = self._rotation_matrix_y(ry)
        Rz = self._rotation_matrix_z(rz)
        R = Rz @ Ry @ Rx

        # 應用旋轉
        for i in range(len(keypoints)):
            relative_pos = keypoints[i] - center
            augmented[i] = center + R @ relative_pos

        return augmented

    def _augment_translation(self, keypoints: np.ndarray,
                             translation: np.ndarray) -> np.ndarray:
        """
        位置增強 - 平移

        基於 PoseAug 的位置增強:
        - 在 3D 空間中平移整個骨架
        """
        return keypoints + translation

    def _random_rotation(self) -> Tuple[float, float, float]:
        """生成隨機旋轉角度"""
        rx = np.random.uniform(-self.config.rotation_range[0],
                               self.config.rotation_range[0])
        ry = np.random.uniform(-self.config.rotation_range[1],
                               self.config.rotation_range[1])
        rz = np.random.uniform(-self.config.rotation_range[2],
                               self.config.rotation_range[2])
        return (rx, ry, rz)

    def _random_translation(self) -> np.ndarray:
        """生成隨機平移向量"""
        return np.random.uniform(-self.config.translation_range,
                                 self.config.translation_range, 3)

    @staticmethod
    def _rotation_matrix_x(angle: float) -> np.ndarray:
        """繞 X 軸的旋轉矩陣"""
        c, s = math.cos(angle), math.sin(angle)
        return np.array([
            [1, 0, 0],
            [0, c, -s],
            [0, s, c]
        ])

    @staticmethod
    def _rotation_matrix_y(angle: float) -> np.ndarray:
        """繞 Y 軸的旋轉矩陣"""
        c, s = math.cos(angle), math.sin(angle)
        return np.array([
            [c, 0, s],
            [0, 1, 0],
            [-s, 0, c]
        ])

    @staticmethod
    def _rotation_matrix_z(angle: float) -> np.ndarray:
        """繞 Z 軸的旋轉矩陣"""
        c, s = math.cos(angle), math.sin(angle)
        return np.array([
            [c, -s, 0],
            [s, c, 0],
            [0, 0, 1]
        ])

    @staticmethod
    def _rotation_matrix_from_axis_angle(axis: np.ndarray, angle: float) -> np.ndarray:
        """從軸角表示構建旋轉矩陣 (Rodrigues' formula)"""
        K = np.array([
            [0, -axis[2], axis[1]],
            [axis[2], 0, -axis[0]],
            [-axis[1], axis[0], 0]
        ])
        R = np.eye(3) + math.sin(angle) * K + (1 - math.cos(angle)) * (K @ K)
        return R


class PoseAugmentationGenerator:
    """
    批量姿態增強生成器
    用於生成多個增強變體
    """

    def __init__(self, augmentor: Optional[PoseAugmentor] = None):
        self.augmentor = augmentor or PoseAugmentor()

    def generate_augmentations(self, keypoints_3d: np.ndarray,
                               num_augmentations: int = 10) -> List[Dict]:
        """
        生成多個增強變體

        Args:
            keypoints_3d: 原始 3D 關鍵點
            num_augmentations: 要生成的增強數量

        Returns:
            增強結果列表，每個包含增強後的關鍵點和使用的參數
        """
        results = []

        for i in range(num_augmentations):
            # 生成隨機參數
            params = {
                'bone_length_scale': np.random.uniform(0.85, 1.15),
                'bone_angle': np.random.uniform(-10, 10),
                'rotation': (
                    np.random.uniform(-20, 20),
                    np.random.uniform(-20, 20),
                    np.random.uniform(-10, 10)
                ),
                'translation': np.random.uniform(-0.2, 0.2, 3)
            }

            augmented = self.augmentor.augment_pose(keypoints_3d, params)

            results.append({
                'index': i,
                'keypoints_3d': augmented.tolist() if isinstance(augmented, np.ndarray) else augmented,
                'parameters': {
                    'bone_length_scale': float(params['bone_length_scale']),
                    'bone_angle': float(params['bone_angle']),
                    'rotation': [float(r) for r in params['rotation']],
                    'translation': [float(t) for t in params['translation']]
                }
            })

        return results

    def generate_preset_augmentations(self, keypoints_3d: np.ndarray) -> Dict[str, Dict]:
        """
        生成預設的增強變體 (用於展示不同增強效果)

        Returns:
            字典，包含各種預設增強
        """
        presets = {
            'original': {
                'name': '原始姿態',
                'params': None,
                'keypoints_3d': keypoints_3d.tolist() if isinstance(keypoints_3d, np.ndarray) else keypoints_3d
            },
            'larger': {
                'name': '放大 (BL 1.2)',
                'params': {'bone_length_scale': 1.2},
            },
            'smaller': {
                'name': '縮小 (BL 0.8)',
                'params': {'bone_length_scale': 0.8},
            },
            'rotated_left': {
                'name': '左旋轉 30°',
                'params': {'rotation': (0, 30, 0)},
            },
            'rotated_right': {
                'name': '右旋轉 30°',
                'params': {'rotation': (0, -30, 0)},
            },
            'tilted_forward': {
                'name': '前傾 20°',
                'params': {'rotation': (20, 0, 0)},
            },
            'bent_arms': {
                'name': '手臂彎曲',
                'params': {'bone_angle': 15},
            },
            'combined': {
                'name': '組合增強',
                'params': {
                    'bone_length_scale': 1.1,
                    'rotation': (10, 15, 5),
                    'bone_angle': 8
                },
            },
        }

        # 創建增強器 (禁用隨機增強)
        config = AugmentationConfig(
            bone_length_enabled=False,
            bone_angle_enabled=False,
            rotation_enabled=False,
            translation_enabled=False
        )
        augmentor = PoseAugmentor(config)

        # 生成每個預設的增強
        for key, preset in presets.items():
            if key == 'original':
                continue

            # 為這個預設啟用相應的增強
            params = preset['params']
            if params:
                config_copy = AugmentationConfig(
                    bone_length_enabled='bone_length_scale' in params,
                    bone_angle_enabled='bone_angle' in params,
                    rotation_enabled='rotation' in params,
                    translation_enabled='translation' in params
                )
                temp_augmentor = PoseAugmentor(config_copy)
                augmented = temp_augmentor.augment_pose(keypoints_3d, params)
                preset['keypoints_3d'] = augmented.tolist() if isinstance(augmented, np.ndarray) else augmented

        return presets


# 輔助函數
def create_sample_pose() -> np.ndarray:
    """
    創建一個樣本 T-pose 用於測試
    """
    # T-pose 的標準位置 (單位: 米)
    keypoints = np.array([
        [0, 1.7, 0],       # 0: nose
        [-0.05, 1.75, 0],  # 1: left_eye
        [0.05, 1.75, 0],   # 2: right_eye
        [-0.1, 1.7, 0],    # 3: left_ear
        [0.1, 1.7, 0],     # 4: right_ear
        [-0.2, 1.5, 0],    # 5: left_shoulder
        [0.2, 1.5, 0],     # 6: right_shoulder
        [-0.5, 1.5, 0],    # 7: left_elbow
        [0.5, 1.5, 0],     # 8: right_elbow
        [-0.75, 1.5, 0],   # 9: left_wrist
        [0.75, 1.5, 0],    # 10: right_wrist
        [-0.15, 1.0, 0],   # 11: left_hip
        [0.15, 1.0, 0],    # 12: right_hip
        [-0.15, 0.5, 0],   # 13: left_knee
        [0.15, 0.5, 0],    # 14: right_knee
        [-0.15, 0, 0],     # 15: left_ankle
        [0.15, 0, 0],      # 16: right_ankle
    ], dtype=np.float32)

    return keypoints


def validate_pose(keypoints: np.ndarray) -> Dict:
    """
    驗證姿態的物理合理性

    Returns:
        驗證結果，包含是否有效和問題描述
    """
    issues = []

    if keypoints is None or len(keypoints) != 17:
        return {'valid': False, 'issues': ['無效的關鍵點數量']}

    # 檢查骨骼長度合理性
    bone_lengths = []
    for parent, child in SKELETON_BONES:
        length = np.linalg.norm(keypoints[child] - keypoints[parent])
        bone_lengths.append(length)

        # 骨骼長度應該在合理範圍內 (0.01m - 1m)
        if length < 0.01:
            issues.append(f'骨骼 {parent}-{child} 太短: {length:.4f}m')
        elif length > 1.0:
            issues.append(f'骨骼 {parent}-{child} 太長: {length:.4f}m')

    # 檢查對稱性
    left_arm_length = np.linalg.norm(keypoints[9] - keypoints[5])
    right_arm_length = np.linalg.norm(keypoints[10] - keypoints[6])
    arm_asymmetry = abs(left_arm_length - right_arm_length) / max(left_arm_length, right_arm_length)

    if arm_asymmetry > 0.3:
        issues.append(f'手臂不對稱: {arm_asymmetry:.1%}')

    return {
        'valid': len(issues) == 0,
        'issues': issues,
        'bone_lengths': bone_lengths
    }
