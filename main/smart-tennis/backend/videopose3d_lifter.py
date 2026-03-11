"""
VideoPose3D 2D→3D 姿態提升模組

整合自 Facebook Research 的 VideoPose3D，通過 PoseAug (CVPR 2021) 的預訓練權重
將 2D 關鍵點提升為精確的 3D 座標。

原始碼: https://github.com/facebookresearch/VideoPose3D
權重來源: https://github.com/jfzhang95/PoseAug

輸入: COCO 17 格式的 2D 關鍵點 (像素座標)
輸出: COCO 17 格式的 3D 座標 (root-relative, 米)
"""

import os
import numpy as np
import torch
import torch.nn as nn
from typing import Optional, Tuple


# ============================================================
# VideoPose3D 模型架構 (移植自 models_baseline/videopose/)
# ============================================================

class TemporalModelBase(nn.Module):
    def __init__(self, num_joints_in, in_features, num_joints_out,
                 filter_widths, causal, dropout, channels):
        super().__init__()
        for fw in filter_widths:
            assert fw % 2 != 0, 'Only odd filter widths are supported'

        self.num_joints_in = num_joints_in
        self.in_features = in_features
        self.num_joints_out = num_joints_out
        self.filter_widths = filter_widths

        self.drop = nn.Dropout(dropout)
        self.relu = nn.ReLU(inplace=True)

        self.pad = [filter_widths[0] // 2]
        self.expand_bn = nn.BatchNorm1d(channels, momentum=0.1)
        self.shrink = nn.Conv1d(channels, num_joints_out * 3, 1)

    def forward(self, x):
        """
        input: (B, 16, 2) 或 (B, 32)
        output: (B, 16, 3) root-relative 3D 座標
        """
        if len(x.shape) == 2:
            x = x.view(x.shape[0], 16, 2)
        x = x.view(x.shape[0], 1, 16, 2)

        assert len(x.shape) == 4
        assert x.shape[-2] == self.num_joints_in
        assert x.shape[-1] == self.in_features

        sz = x.shape[:3]
        x = x.view(x.shape[0], x.shape[1], -1)
        x = x.permute(0, 2, 1)

        x = self._forward_blocks(x)

        x = x.permute(0, 2, 1)
        x = x.view(sz[0], -1, self.num_joints_out, 3)
        x = x.view(sz[0], self.num_joints_out * 3)

        # 15 joints → 16 joints: 在最前面補 hip root (0,0,0)
        out = torch.cat([torch.zeros_like(x)[:, :3], x], 1).view(sz[0], 16, 3)
        return out


class TemporalModelOptimized1f(TemporalModelBase):
    """
    VideoPose3D 單幀推理優化版本
    使用 strided convolution 替代 dilated convolution
    """

    def __init__(self, num_joints_in, in_features, num_joints_out,
                 filter_widths, causal=False, dropout=0.25, channels=1024):
        super().__init__(num_joints_in, in_features, num_joints_out,
                         filter_widths, causal, dropout, channels)

        self.expand_conv = nn.Conv1d(num_joints_in * in_features, channels,
                                     filter_widths[0], stride=filter_widths[0], bias=False)

        layers_conv = []
        layers_bn = []

        self.causal_shift = [(filter_widths[0] // 2) if causal else 0]
        next_dilation = filter_widths[0]
        for i in range(1, len(filter_widths)):
            self.pad.append((filter_widths[i] - 1) * next_dilation // 2)
            self.causal_shift.append((filter_widths[i] // 2) if causal else 0)

            layers_conv.append(nn.Conv1d(channels, channels, filter_widths[i],
                                         stride=filter_widths[i], bias=False))
            layers_bn.append(nn.BatchNorm1d(channels, momentum=0.1))
            layers_conv.append(nn.Conv1d(channels, channels, 1, dilation=1, bias=False))
            layers_bn.append(nn.BatchNorm1d(channels, momentum=0.1))
            next_dilation *= filter_widths[i]

        self.layers_conv = nn.ModuleList(layers_conv)
        self.layers_bn = nn.ModuleList(layers_bn)

    def _forward_blocks(self, x):
        x = self.drop(self.relu(self.expand_bn(self.expand_conv(x))))

        for i in range(len(self.pad) - 1):
            res = x[:, :, self.causal_shift[i + 1] + self.filter_widths[i + 1] // 2::self.filter_widths[i + 1]]
            x = self.drop(self.relu(self.layers_bn[2 * i](self.layers_conv[2 * i](x))))
            x = res + self.drop(self.relu(self.layers_bn[2 * i + 1](self.layers_conv[2 * i + 1](x))))

        x = self.shrink(x)
        return x


# ============================================================
# COCO 17 2D → H36M 16 2D 映射
# ============================================================

def coco17_2d_to_h36m16_2d(coco_2d: np.ndarray) -> np.ndarray:
    """
    將 COCO 17 格式的 2D 關鍵點映射為 H36M 16 格式

    Args:
        coco_2d: (17, 2) 或 (N, 17, 2)

    Returns:
        h36m_2d: (16, 2) 或 (N, 16, 2)
    """
    single = coco_2d.ndim == 2
    if single:
        coco_2d = coco_2d[np.newaxis, ...]

    N = coco_2d.shape[0]
    h36m = np.zeros((N, 16, 2), dtype=np.float32)

    h36m[:, 0] = (coco_2d[:, 11] + coco_2d[:, 12]) / 2.0   # Pelvis
    h36m[:, 1] = coco_2d[:, 12]                               # RHip
    h36m[:, 2] = coco_2d[:, 14]                               # RKnee
    h36m[:, 3] = coco_2d[:, 16]                               # RAnkle
    h36m[:, 4] = coco_2d[:, 11]                               # LHip
    h36m[:, 5] = coco_2d[:, 13]                               # LKnee
    h36m[:, 6] = coco_2d[:, 15]                               # LAnkle

    shoulder_center = (coco_2d[:, 5] + coco_2d[:, 6]) / 2.0
    hip_center = h36m[:, 0]
    h36m[:, 7] = (shoulder_center + hip_center) / 2.0         # Spine
    h36m[:, 8] = shoulder_center                               # Neck
    h36m[:, 9] = coco_2d[:, 0]                                # Head (nose)
    h36m[:, 10] = coco_2d[:, 5]                               # LShoulder
    h36m[:, 11] = coco_2d[:, 7]                               # LElbow
    h36m[:, 12] = coco_2d[:, 9]                               # LWrist
    h36m[:, 13] = coco_2d[:, 6]                               # RShoulder
    h36m[:, 14] = coco_2d[:, 8]                               # RElbow
    h36m[:, 15] = coco_2d[:, 10]                              # RWrist

    return h36m[0] if single else h36m


def h36m16_3d_to_coco17_3d(h36m_3d: np.ndarray, original_coco_2d: Optional[np.ndarray] = None) -> np.ndarray:
    """
    將 H36M 16 格式的 3D 座標映射回 COCO 17 格式

    Args:
        h36m_3d: (16, 3) 或 (N, 16, 3)
        original_coco_2d: 原始 COCO 2D 座標，用於估計眼睛/耳朵位置

    Returns:
        coco_3d: (17, 3) 或 (N, 17, 3)
    """
    single = h36m_3d.ndim == 2
    if single:
        h36m_3d = h36m_3d[np.newaxis, ...]

    N = h36m_3d.shape[0]
    coco = np.zeros((N, 17, 3), dtype=np.float32)

    coco[:, 0] = h36m_3d[:, 9]     # nose = Head
    coco[:, 5] = h36m_3d[:, 10]    # left_shoulder
    coco[:, 6] = h36m_3d[:, 13]    # right_shoulder
    coco[:, 7] = h36m_3d[:, 11]    # left_elbow
    coco[:, 8] = h36m_3d[:, 14]    # right_elbow
    coco[:, 9] = h36m_3d[:, 12]    # left_wrist
    coco[:, 10] = h36m_3d[:, 15]   # right_wrist
    coco[:, 11] = h36m_3d[:, 4]    # left_hip
    coco[:, 12] = h36m_3d[:, 1]    # right_hip
    coco[:, 13] = h36m_3d[:, 5]    # left_knee
    coco[:, 14] = h36m_3d[:, 2]    # right_knee
    coco[:, 15] = h36m_3d[:, 6]    # left_ankle
    coco[:, 16] = h36m_3d[:, 3]    # right_ankle

    # 眼睛和耳朵: 從頭部位置推算
    head = h36m_3d[:, 9]
    neck = h36m_3d[:, 8]
    head_dir = head - neck
    head_len = np.linalg.norm(head_dir, axis=-1, keepdims=True) + 1e-8

    # 用左右肩膀推算橫向方向
    l_shoulder = h36m_3d[:, 10]
    r_shoulder = h36m_3d[:, 13]
    lateral = l_shoulder - r_shoulder
    lateral_norm = lateral / (np.linalg.norm(lateral, axis=-1, keepdims=True) + 1e-8)

    eye_offset = head_dir * 0.15  # 眼睛在頭頂方向略前
    ear_offset = head_dir * 0.05

    coco[:, 1] = head + eye_offset - lateral_norm * head_len * 0.15  # left_eye
    coco[:, 2] = head + eye_offset + lateral_norm * head_len * 0.15  # right_eye
    coco[:, 3] = head + ear_offset - lateral_norm * head_len * 0.35  # left_ear
    coco[:, 4] = head + ear_offset + lateral_norm * head_len * 0.35  # right_ear

    return coco[0] if single else coco


def normalize_screen_coordinates(X: np.ndarray, w: float, h: float) -> np.ndarray:
    """
    將像素座標歸一化到 [-1, 1]，保持長寬比
    與 PoseAug 原始碼中 common/camera.py 一致

    Args:
        X: (..., 2) 像素座標
        w: 影像寬度
        h: 影像高度

    Returns:
        歸一化後的座標
    """
    assert X.shape[-1] == 2
    return X / w * 2 - np.array([1, h / w])


# ============================================================
# VideoPose3D Lifter 推理封裝
# ============================================================

class VideoPose3DLifter:
    """
    VideoPose3D 2D→3D 姿態提升器

    用法:
        lifter = VideoPose3DLifter()
        lifter.load_weights('/path/to/ckpt_best_dhp_p1.pth.tar')
        pose_3d = lifter.lift(coco_2d_keypoints, frame_width, frame_height)
    """

    def __init__(self, device: Optional[str] = None):
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        # 初始化 VideoPose3D: 16 joints in, 2 features, 15 joints out
        # filter_widths = [1,1,1,1,1] 對應 PoseAug 預設的 4 stages
        filter_widths = [1, 1, 1, 1, 1]
        self.model = TemporalModelOptimized1f(
            16, 2, 15,
            filter_widths=filter_widths,
            causal=False,
            dropout=0.25,
            channels=1024
        )
        self.model.to(self.device)
        self.model.eval()
        self.weights_loaded = False

    def load_weights(self, checkpoint_path: str) -> bool:
        """
        載入預訓練權重

        支援兩種 checkpoint 格式:
        - PoseAug 格式: {'epoch': int, 'model_pos': state_dict}
        - Baseline 格式: {'state_dict': state_dict, 'epoch': int}
        """
        try:
            ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=False)

            if 'model_pos' in ckpt:
                self.model.load_state_dict(ckpt['model_pos'])
            elif 'state_dict' in ckpt:
                self.model.load_state_dict(ckpt['state_dict'])
            else:
                self.model.load_state_dict(ckpt)

            self.model.eval()
            self.weights_loaded = True
            epoch = ckpt.get('epoch', '?')
            print(f"[OK] VideoPose3D weights loaded: {checkpoint_path} (epoch {epoch})")
            return True

        except Exception as e:
            print(f"[FAIL] VideoPose3D weights load failed: {e}")
            self.weights_loaded = False
            return False

    def lift(self, coco_2d: np.ndarray, frame_width: float, frame_height: float,
             confidence: Optional[np.ndarray] = None) -> np.ndarray:
        """
        將 COCO 17 格式的 2D 關鍵點提升為 3D 座標

        Args:
            coco_2d: (17, 2) 像素座標
            frame_width: 影像寬度
            frame_height: 影像高度
            confidence: (17,) 置信度

        Returns:
            coco_3d: (17, 3) root-relative 3D 座標
        """
        # 1. COCO 17 2D → H36M 16 2D
        h36m_2d = coco17_2d_to_h36m16_2d(coco_2d)

        # 2. 歸一化到 [-1, 1]
        h36m_2d_norm = normalize_screen_coordinates(h36m_2d, frame_width, frame_height)

        # 3. 轉為 tensor
        input_tensor = torch.from_numpy(h36m_2d_norm).float().unsqueeze(0).to(self.device)

        # 4. 推理
        with torch.no_grad():
            h36m_3d = self.model(input_tensor)

        # 5. 轉回 numpy
        h36m_3d_np = h36m_3d.cpu().numpy()[0]  # (16, 3)

        # 5.5 座標系修正:
        #   H36M 影像座標系: X 向右, Y 向下
        #   Three.js 3D 座標系: X 向右, Y 向上
        #   翻轉 X 修正鏡像, 翻轉 Y 修正上下
        h36m_3d_np[:, 0] = -h36m_3d_np[:, 0]
        h36m_3d_np[:, 1] = -h36m_3d_np[:, 1]

        # 6. H36M 16 3D → COCO 17 3D
        coco_3d = h36m16_3d_to_coco17_3d(h36m_3d_np)

        # 7. 置信度過濾
        if confidence is not None:
            for i in range(17):
                if confidence[i] < 0.3:
                    coco_3d[i] = 0.0

        return coco_3d

    def lift_batch(self, coco_2d_batch: np.ndarray, frame_width: float, frame_height: float) -> np.ndarray:
        """
        批量提升多個 2D 姿態

        Args:
            coco_2d_batch: (N, 17, 2) 像素座標
            frame_width, frame_height: 影像尺寸

        Returns:
            coco_3d_batch: (N, 17, 3) 3D 座標
        """
        # 1. COCO 17 → H36M 16
        h36m_2d = coco17_2d_to_h36m16_2d(coco_2d_batch)

        # 2. 歸一化
        h36m_2d_norm = normalize_screen_coordinates(h36m_2d, frame_width, frame_height)

        # 3. 推理
        input_tensor = torch.from_numpy(h36m_2d_norm).float().to(self.device)
        with torch.no_grad():
            h36m_3d = self.model(input_tensor)

        # 4. 座標系修正 + 轉換回 COCO 17
        h36m_3d_np = h36m_3d.cpu().numpy()
        h36m_3d_np[:, :, 0] = -h36m_3d_np[:, :, 0]  # X 軸翻轉 (修正鏡像)
        h36m_3d_np[:, :, 1] = -h36m_3d_np[:, :, 1]  # Y 軸翻轉 (修正上下)
        coco_3d = h36m16_3d_to_coco17_3d(h36m_3d_np)

        return coco_3d

    def get_status(self) -> dict:
        return {
            'available': True,
            'weights_loaded': self.weights_loaded,
            'device': str(self.device),
            'model_params': sum(p.numel() for p in self.model.parameters()),
            'model_name': 'VideoPose3D (TemporalModelOptimized1f)',
        }
