"""
PoseAug GAN 模組 — 整合自原始 PoseAug (CVPR 2021) 倉庫
https://github.com/jfzhang95/PoseAug

此模組包含：
- GAN 生成器 (BAGenerator, BLGenerator, RTGenerator)
- 骨骼向量工具函數
- COCO 17 ↔ H36M 16 關鍵點轉換
- 推理封裝類 PoseAugGAN

原始論文:
  PoseAug: A Differentiable Pose Augmentation Framework for 3D Human Pose Estimation
  Kehong Gong, Jianfeng Zhang, Jiashi Feng. CVPR 2021 (Oral)
"""

import os
import math
import numpy as np
import torch
import torch.nn as nn
from typing import List, Dict, Optional, Tuple


# ============================================================
# torchgeometry 替代實現 (避免額外依賴)
# ============================================================

def angle_axis_to_rotation_matrix(angle_axis: torch.Tensor) -> torch.Tensor:
    """
    將 axis-angle 旋轉表示轉換為 4x4 旋轉矩陣 (Rodrigues 公式)
    替代 torchgeometry.angle_axis_to_rotation_matrix

    Args:
        angle_axis: (N, 3) tensor

    Returns:
        rotation_matrix: (N, 4, 4) tensor
    """
    batch_size = angle_axis.shape[0]
    device = angle_axis.device

    theta_squared = torch.sum(angle_axis * angle_axis, dim=1, keepdim=True)
    theta = torch.sqrt(theta_squared + 1e-8)
    half_theta = 0.5 * theta

    k = angle_axis / (theta + 1e-8)  # (N, 3) 單位軸

    kx = k[:, 0:1]
    ky = k[:, 1:2]
    kz = k[:, 2:3]

    cos_t = torch.cos(theta)
    sin_t = torch.sin(theta)
    one_minus_cos = 1.0 - cos_t

    # Rodrigues formula: R = I + sin(θ)*K + (1-cos(θ))*K²
    # 展開為顯式矩陣元素
    r00 = cos_t + kx * kx * one_minus_cos
    r01 = kx * ky * one_minus_cos - kz * sin_t
    r02 = kx * kz * one_minus_cos + ky * sin_t
    r10 = ky * kx * one_minus_cos + kz * sin_t
    r11 = cos_t + ky * ky * one_minus_cos
    r12 = ky * kz * one_minus_cos - kx * sin_t
    r20 = kz * kx * one_minus_cos - ky * sin_t
    r21 = kz * ky * one_minus_cos + kx * sin_t
    r22 = cos_t + kz * kz * one_minus_cos

    zeros = torch.zeros(batch_size, 1, device=device)
    ones = torch.ones(batch_size, 1, device=device)

    rotation_matrix = torch.cat([
        r00, r01, r02, zeros,
        r10, r11, r12, zeros,
        r20, r21, r22, zeros,
        zeros, zeros, zeros, ones
    ], dim=1).view(batch_size, 4, 4)

    # 處理角度接近 0 的情況 (返回單位矩陣)
    small_angle_mask = (theta_squared.squeeze(1) < 1e-6).unsqueeze(1).unsqueeze(2)
    identity = torch.eye(4, device=device).unsqueeze(0).expand(batch_size, -1, -1)
    rotation_matrix = torch.where(small_angle_mask, identity, rotation_matrix)

    return rotation_matrix


# ============================================================
# 骨骼向量工具函數 (移植自 utils/gan_utils.py)
# ============================================================

def get_BoneVecbypose3d(x, num_joints=16):
    """將 3D 姿態轉換為骨骼向量 (N x 15 x 3)"""
    Ct = torch.Tensor([
        [1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [1, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [1, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 1, -1, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 1, -1, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, -1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, -1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, -1, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, -1, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, -1, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, -1],
    ]).transpose(1, 0)

    Ct = Ct.to(x.device)
    C = Ct.repeat([x.size(0), 1, 1]).view(-1, num_joints, num_joints - 1)
    pose3 = x.permute(0, 2, 1).contiguous()
    B = torch.matmul(pose3, C)
    B = B.permute(0, 2, 1)
    return B


def get_pose3dbyBoneVec(bones, num_joints=16):
    """將骨骼向量轉回 3D 姿態 (N x 16 x 3)"""
    Ctinverse = torch.Tensor([
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [-1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [-1, -1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, -1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, -1, -1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, -1, -1, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, -1, -1, -1, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, -1, -1, 0, -1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, -1, -1, 0, -1, -1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, -1, -1, 0, -1, -1, -1, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, -1, -1, 0, 0, 0, 0, -1, 0, 0],
        [0, 0, 0, 0, 0, 0, -1, -1, 0, 0, 0, 0, -1, -1, 0],
        [0, 0, 0, 0, 0, 0, -1, -1, 0, 0, 0, 0, -1, -1, -1],
    ]).transpose(1, 0)

    Ctinverse = Ctinverse.to(bones.device)
    C = Ctinverse.repeat([bones.size(0), 1, 1]).view(-1, num_joints - 1, num_joints)
    bonesT = bones.permute(0, 2, 1).contiguous()
    pose3d = torch.matmul(bonesT, C)
    pose3d = pose3d.permute(0, 2, 1).contiguous()
    return pose3d


def get_bone_lengthbypose3d(x):
    """取得骨骼長度 (N x 15 x 1)"""
    bonevec = get_BoneVecbypose3d(x)
    bones_length = torch.norm(bonevec, dim=2, keepdim=True)
    return bones_length


def get_bone_unit_vecbypose3d(x):
    """取得骨骼單位向量 (N x 15 x 3)"""
    bonevec = get_BoneVecbypose3d(x)
    bonelength = get_bone_lengthbypose3d(x)
    bone_unitvec = bonevec / (bonelength + 1e-8)
    return bone_unitvec


def blaugment9to15(x, bl, blr, num_bone=15):
    """將 9 個骨骼長度比率擴展到 15 段並應用"""
    blr9to15 = torch.Tensor([
        [1, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0, 0, 0],
        [1, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 1],
        [0, 0, 0, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 1],
    ]).transpose(1, 0)

    blr9to15 = blr9to15.to(blr.device)
    blr9to15 = blr9to15.repeat([blr.size(0), 1, 1]).view(blr.size(0), 9, 15)
    blr_T = blr.permute(0, 2, 1).contiguous()
    blr_15_T = torch.matmul(blr_T, blr9to15)
    blr_15 = blr_15_T.permute(0, 2, 1).contiguous()

    root = x[:, :1, :] * 1.0
    x = x - x[:, :1, :]
    bones_unit = get_bone_unit_vecbypose3d(x)
    bones_length = torch.mul(bl, blr_15) + bl
    modifyed_bone = bones_unit * bones_length
    out = get_pose3dbyBoneVec(modifyed_bone)
    return out + root


# ============================================================
# GAN 生成器模型 (移植自 models_poseaug/gan_generator.py)
# ============================================================

class LinearBlock(nn.Module):
    def __init__(self, linear_size):
        super().__init__()
        self.w1 = nn.Linear(linear_size, linear_size)
        self.batch_norm1 = nn.BatchNorm1d(linear_size)
        self.w2 = nn.Linear(linear_size, linear_size)
        self.batch_norm2 = nn.BatchNorm1d(linear_size)
        self.relu = nn.LeakyReLU(inplace=True)

    def forward(self, x):
        y = self.relu(self.batch_norm1(self.w1(x)))
        y = self.relu(self.batch_norm2(self.w2(y)))
        return y


class BAGenerator(nn.Module):
    """骨骼角度 (Bone Angle) 增強生成器 — 用神經網路學習合理的骨骼角度修改"""

    def __init__(self, input_size=16 * 3, noise_channel=48, linear_size=256, num_stage=2):
        super().__init__()
        self.input_size = input_size
        self.noise_channel = noise_channel

        self.w1 = nn.Linear(input_size + noise_channel, linear_size)
        self.batch_norm1 = nn.BatchNorm1d(linear_size)
        self.linear_stages = nn.ModuleList([LinearBlock(linear_size) for _ in range(num_stage)])
        self.w2 = nn.Linear(linear_size, input_size - 3)  # 輸出 15 個骨骼的角度修改
        self.relu = nn.LeakyReLU(inplace=True)

    def forward(self, inputs_3d):
        root_origin = inputs_3d[:, :1, :] * 1.0
        x = inputs_3d - inputs_3d[:, :1, :]  # root-relative

        bones_unit = get_bone_unit_vecbypose3d(x)
        bones_length = get_bone_lengthbypose3d(x)

        x = x.view(x.size(0), -1)
        noise = torch.randn(x.shape[0], self.noise_channel, device=x.device)

        y = self.relu(self.batch_norm1(self.w1(torch.cat((x, noise), dim=1))))
        for stage in self.linear_stages:
            y = stage(y)
        y = self.w2(y).view(x.size(0), -1, 3)

        # 修改骨骼方向，保持長度不變
        modifyed = bones_unit + y
        modifyed_unit = modifyed / (torch.norm(modifyed, dim=2, keepdim=True) + 1e-8)

        # 固定骨盆到軀幹的骨骼方向，避免整體旋轉
        tmp_mask = torch.ones_like(bones_unit)
        tmp_mask[:, [6, 7], :] = 0.0
        modifyed_unit = modifyed_unit * tmp_mask + bones_unit * (1 - tmp_mask)

        cos_angle = torch.sum(modifyed_unit * bones_unit, dim=2)
        ba_diff = 1 - cos_angle

        modifyed_bone = modifyed_unit * bones_length
        out = get_pose3dbyBoneVec(modifyed_bone) + root_origin

        return out, ba_diff


class BLGenerator(nn.Module):
    """骨骼長度 (Bone Length) 增強生成器 — 用神經網路學習合理的體型變化"""

    def __init__(self, input_size=16 * 3, noise_channel=48, linear_size=256, num_stage=2,
                 blr_tanhlimit=0.2):
        super().__init__()
        self.input_size = input_size + 15  # pose + bone lengths
        self.noise_channel = noise_channel
        self.blr_tanhlimit = blr_tanhlimit

        self.w1 = nn.Linear(self.input_size + noise_channel, linear_size)
        self.batch_norm1 = nn.BatchNorm1d(linear_size)
        self.linear_stages = nn.ModuleList([LinearBlock(linear_size) for _ in range(num_stage)])
        self.w2 = nn.Linear(linear_size, 9)  # 9 個獨立骨骼長度比率
        self.relu = nn.LeakyReLU(inplace=True)

    def forward(self, inputs_3d, augx):
        root_origin = inputs_3d[:, :1, :] * 1.0
        x = inputs_3d - inputs_3d[:, :1, :]

        x = x.view(x.size(0), -1)
        bones_length_x = get_bone_lengthbypose3d(x.view(x.size(0), -1, 3)).squeeze(2)
        noise = torch.randn(x.shape[0], self.noise_channel, device=x.device)

        blr = self.relu(self.batch_norm1(self.w1(torch.cat((x, bones_length_x, noise), dim=1))))
        for stage in self.linear_stages:
            blr = stage(blr)
        blr = self.w2(blr)

        # 遮蔽第 8 段骨骼 (避免身高-距離歧義)
        tmp_mask = torch.from_numpy(
            np.array([[1, 1, 1, 1, 0, 1, 1, 1, 1]]).astype('float32')
        ).to(blr.device)
        blr = blr * tmp_mask
        blr = nn.Tanh()(blr) * self.blr_tanhlimit  # 限制 ±20% 的長度變化

        bones_length = get_bone_lengthbypose3d(augx)
        augx_bl = blaugment9to15(augx, bones_length, blr.unsqueeze(2))
        return augx_bl, blr


class RTGenerator(nn.Module):
    """旋轉平移 (Rotation-Translation) 增強生成器 — 用神經網路學習合理的視角變換"""

    def __init__(self, input_size=16 * 3, noise_channel=48, linear_size=256, num_stage=2):
        super().__init__()
        self.input_size = input_size
        self.noise_channel = noise_channel

        # R 分支
        self.w1_R = nn.Linear(input_size + noise_channel, linear_size)
        self.batch_norm_R = nn.BatchNorm1d(linear_size)
        self.linear_stages_R = nn.ModuleList([LinearBlock(linear_size) for _ in range(num_stage)])
        self.w2_R = nn.Linear(linear_size, 3)

        # T 分支
        self.w1_T = nn.Linear(input_size + noise_channel, linear_size)
        self.batch_norm_T = nn.BatchNorm1d(linear_size)
        self.linear_stages_T = nn.ModuleList([LinearBlock(linear_size) for _ in range(num_stage)])
        self.w2_T = nn.Linear(linear_size, 3)

        self.relu = nn.LeakyReLU(inplace=True)

    def forward(self, inputs_3d, augx):
        root_origin = inputs_3d[:, :1, :] * 1.0
        x = inputs_3d - inputs_3d[:, :1, :]
        x = x.view(x.size(0), -1)

        # 計算 R (旋轉)
        noise = torch.randn(x.shape[0], self.noise_channel, device=x.device)
        r = self.relu(self.batch_norm_R(self.w1_R(torch.cat((x, noise), dim=1))))
        for stage in self.linear_stages_R:
            r = stage(r)
        r = self.w2_R(r)
        r = nn.Tanh()(r) * math.pi  # 限制在 [-π, π]
        r = r.view(x.size(0), 3)
        rM = angle_axis_to_rotation_matrix(r)[..., :3, :3]  # (N, 3, 3)

        # 計算 T (平移)
        noise = torch.randn(x.shape[0], self.noise_channel, device=x.device)
        t = self.relu(self.batch_norm_T(self.w1_T(torch.cat((x, noise), dim=1))))
        for stage in self.linear_stages_T:
            t = stage(t)
        t = self.w2_T(t)
        t[:, 2] = t[:, 2].clone() * t[:, 2].clone()  # z 平移保持正值
        t = t.view(x.size(0), 1, 3)

        # 應用 RT 到增強後的姿態
        augx = augx - augx[:, :1, :]
        augx = augx.permute(0, 2, 1).contiguous()
        augx_r = torch.matmul(rM, augx)
        augx_r = augx_r.permute(0, 2, 1).contiguous()
        augx_rt = augx_r + t

        return augx_rt, (r, t)


class PoseGenerator(nn.Module):
    """PoseAug 完整生成器 — 依序進行 BA → BL → RT 增強"""

    def __init__(self, blr_tanhlimit=0.2, input_size=16 * 3):
        super().__init__()
        self.BAprocess = BAGenerator(input_size=input_size)
        self.BLprocess = BLGenerator(input_size=input_size, blr_tanhlimit=blr_tanhlimit)
        self.RTprocess = RTGenerator(input_size=input_size)

    def forward(self, inputs_3d):
        """
        輸入: inputs_3d (N, 16, 3) — H36M 格式 root-relative 3D 姿態
        輸出: dict 包含各階段增強結果
        """
        pose_ba, ba_diff = self.BAprocess(inputs_3d)
        pose_bl, blr = self.BLprocess(inputs_3d, pose_ba)
        pose_rt, rt = self.RTprocess(inputs_3d, pose_bl)

        return {
            'pose_ba': pose_ba,
            'ba_diff': ba_diff,
            'pose_bl': pose_bl,
            'blr': blr,
            'pose_rt': pose_rt,
            'rt': rt,
        }


# ============================================================
# COCO 17 ↔ H36M 16 關鍵點轉換
# ============================================================

# H36M 16 關鍵點定義：
# 0:Pelvis 1:RHip 2:RKnee 3:RAnkle 4:LHip 5:LKnee 6:LAnkle
# 7:Spine 8:Neck 9:Head 10:LShoulder 11:LElbow 12:LWrist
# 13:RShoulder 14:RElbow 15:RWrist

# COCO 17 關鍵點定義：
# 0:nose 1:left_eye 2:right_eye 3:left_ear 4:right_ear
# 5:left_shoulder 6:right_shoulder 7:left_elbow 8:right_elbow
# 9:left_wrist 10:right_wrist 11:left_hip 12:right_hip
# 13:left_knee 14:right_knee 15:left_ankle 16:right_ankle

def coco17_to_h36m16(coco_pose: np.ndarray) -> np.ndarray:
    """
    將 COCO 17 關鍵點轉換為 H36M 16 關鍵點

    Args:
        coco_pose: (17, 3) 或 (N, 17, 3)

    Returns:
        h36m_pose: (16, 3) 或 (N, 16, 3)
    """
    single = coco_pose.ndim == 2
    if single:
        coco_pose = coco_pose[np.newaxis, ...]

    N = coco_pose.shape[0]
    h36m = np.zeros((N, 16, 3), dtype=np.float32)

    # 直接映射
    h36m[:, 0] = (coco_pose[:, 11] + coco_pose[:, 12]) / 2.0   # Pelvis = mid(LHip, RHip)
    h36m[:, 1] = coco_pose[:, 12]                                # RHip
    h36m[:, 2] = coco_pose[:, 14]                                # RKnee
    h36m[:, 3] = coco_pose[:, 16]                                # RAnkle
    h36m[:, 4] = coco_pose[:, 11]                                # LHip
    h36m[:, 5] = coco_pose[:, 13]                                # LKnee
    h36m[:, 6] = coco_pose[:, 15]                                # LAnkle

    shoulder_center = (coco_pose[:, 5] + coco_pose[:, 6]) / 2.0
    hip_center = h36m[:, 0]
    h36m[:, 7] = (shoulder_center + hip_center) / 2.0            # Spine = mid(shoulders, hips)
    h36m[:, 8] = shoulder_center                                  # Neck = mid(LShoulder, RShoulder)
    h36m[:, 9] = coco_pose[:, 0]                                 # Head = nose
    h36m[:, 10] = coco_pose[:, 5]                                # LShoulder
    h36m[:, 11] = coco_pose[:, 7]                                # LElbow
    h36m[:, 12] = coco_pose[:, 9]                                # LWrist
    h36m[:, 13] = coco_pose[:, 6]                                # RShoulder
    h36m[:, 14] = coco_pose[:, 8]                                # RElbow
    h36m[:, 15] = coco_pose[:, 10]                               # RWrist

    return h36m[0] if single else h36m


def h36m16_to_coco17(h36m_pose: np.ndarray, original_coco: Optional[np.ndarray] = None) -> np.ndarray:
    """
    將 H36M 16 關鍵點轉回 COCO 17 關鍵點

    Args:
        h36m_pose: (16, 3) 或 (N, 16, 3)
        original_coco: 原始 COCO 姿態，用於保留眼睛/耳朵的相對位置

    Returns:
        coco_pose: (17, 3) 或 (N, 17, 3)
    """
    single = h36m_pose.ndim == 2
    if single:
        h36m_pose = h36m_pose[np.newaxis, ...]
        if original_coco is not None:
            original_coco = original_coco[np.newaxis, ...]

    N = h36m_pose.shape[0]
    coco = np.zeros((N, 17, 3), dtype=np.float32)

    # 直接映射
    coco[:, 0] = h36m_pose[:, 9]    # nose = Head
    coco[:, 5] = h36m_pose[:, 10]   # left_shoulder = LShoulder
    coco[:, 6] = h36m_pose[:, 13]   # right_shoulder = RShoulder
    coco[:, 7] = h36m_pose[:, 11]   # left_elbow = LElbow
    coco[:, 8] = h36m_pose[:, 14]   # right_elbow = RElbow
    coco[:, 9] = h36m_pose[:, 12]   # left_wrist = LWrist
    coco[:, 10] = h36m_pose[:, 15]  # right_wrist = RWrist
    coco[:, 11] = h36m_pose[:, 4]   # left_hip = LHip
    coco[:, 12] = h36m_pose[:, 1]   # right_hip = RHip
    coco[:, 13] = h36m_pose[:, 5]   # left_knee = LKnee
    coco[:, 14] = h36m_pose[:, 2]   # right_knee = RKnee
    coco[:, 15] = h36m_pose[:, 6]   # left_ankle = LAnkle
    coco[:, 16] = h36m_pose[:, 3]   # right_ankle = RAnkle

    # 眼睛和耳朵：保留相對偏移 (如果有原始 COCO 姿態)
    if original_coco is not None:
        head = h36m_pose[:, 9]  # Head
        orig_head = original_coco[:, 0]  # 原始 nose

        for idx in [1, 2, 3, 4]:  # eyes and ears
            offset = original_coco[:, idx] - orig_head
            coco[:, idx] = head + offset
    else:
        # 近似估計眼睛和耳朵位置
        head = h36m_pose[:, 9]
        neck = h36m_pose[:, 8]
        head_dir = head - neck
        head_dir_norm = head_dir / (np.linalg.norm(head_dir, axis=-1, keepdims=True) + 1e-8)

        # 用頭部方向的垂直分量近似
        coco[:, 1] = head + head_dir_norm * 0.02 + np.array([[-0.03, 0, 0]])  # left_eye
        coco[:, 2] = head + head_dir_norm * 0.02 + np.array([[0.03, 0, 0]])   # right_eye
        coco[:, 3] = head + np.array([[-0.06, -0.02, 0]])                     # left_ear
        coco[:, 4] = head + np.array([[0.06, -0.02, 0]])                      # right_ear

    return coco[0] if single else coco


# ============================================================
# PoseAug GAN 推理封裝
# ============================================================

class PoseAugGAN:
    """
    PoseAug GAN 推理封裝類

    用法:
        gan = PoseAugGAN()
        gan.load_weights('/path/to/checkpoint.bin')  # 可選
        results = gan.augment(coco_17_pose, num_augmentations=10)
    """

    def __init__(self, device: Optional[str] = None, blr_tanhlimit: float = 0.2):
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        self.generator = PoseGenerator(blr_tanhlimit=blr_tanhlimit)
        self.generator.to(self.device)
        self.generator.eval()  # 推理模式

        self.weights_loaded = False

    def load_weights(self, checkpoint_path: str) -> bool:
        """
        載入預訓練的 Generator 權重

        Args:
            checkpoint_path: .bin 或 .pth 檔案路徑

        Returns:
            是否成功載入
        """
        try:
            checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)

            # 嘗試各種常見的 checkpoint 格式
            if isinstance(checkpoint, dict):
                if 'model_G' in checkpoint:
                    self.generator.load_state_dict(checkpoint['model_G'])
                elif 'state_dict' in checkpoint:
                    self.generator.load_state_dict(checkpoint['state_dict'])
                elif 'generator' in checkpoint:
                    self.generator.load_state_dict(checkpoint['generator'])
                else:
                    # 嘗試直接作為 state_dict 載入
                    self.generator.load_state_dict(checkpoint)
            else:
                self.generator.load_state_dict(checkpoint)

            self.generator.eval()
            self.weights_loaded = True
            print(f"✅ PoseAug GAN 權重已載入: {checkpoint_path}")
            return True

        except Exception as e:
            print(f"⚠️ PoseAug GAN 權重載入失敗: {e}")
            print("   將使用隨機初始化的生成器（增強品質會較低）")
            self.weights_loaded = False
            return False

    def augment(self, coco_pose: np.ndarray, num_augmentations: int = 10,
                return_intermediate: bool = False,
                include_rt: bool = False,
                rt_max_angle: float = 15.0) -> Dict:
        """
        對 COCO 17 格式的 3D 姿態進行 GAN 增強

        Args:
            coco_pose: (17, 3) 的 COCO 格式 3D 姿態
            num_augmentations: 要生成的增強數量
            return_intermediate: 是否返回 BA/BL 中間結果
            include_rt: 是否包含 RT 旋轉增強 (默認關閉，避免整體亂轉)
            rt_max_angle: RT 旋轉最大角度 (度)，僅在 include_rt=True 時有效

        Returns:
            dict: {
                'augmented_poses': [(17,3), ...],
                'augmentation_details': [...],
                'original_h36m': (16,3),
                'weights_loaded': bool,
            }
        """
        # COCO 17 → H36M 16
        h36m_pose = coco17_to_h36m16(coco_pose)

        # 準備批次輸入 (複製 N 次以生成 N 個不同的增強)
        h36m_batch = np.tile(h36m_pose, (num_augmentations, 1, 1))
        h36m_tensor = torch.from_numpy(h36m_batch).float().to(self.device)

        # GAN 推理
        with torch.no_grad():
            results = self.generator(h36m_tensor)

        # 選擇輸出階段
        if include_rt:
            # 包含 RT: 使用完整 pipeline 的最終輸出，但限制旋轉角度
            augmented_h36m = results['pose_rt'].cpu().numpy()

            # 限制旋轉角度: 如果旋轉角度超出限制，改用 BL 階段的結果
            r_angles = results['rt'][0].cpu().numpy()  # (N, 3) axis-angle
            for i in range(num_augmentations):
                angle_magnitude = np.linalg.norm(r_angles[i]) * 180.0 / math.pi
                if angle_magnitude > rt_max_angle:
                    # 旋轉太大，回退到 BL 階段結果
                    augmented_h36m[i] = results['pose_bl'].cpu().numpy()[i]
        else:
            # 不包含 RT: 只使用 BA + BL 階段的結果 (骨骼角度+長度變化，不整體旋轉)
            augmented_h36m = results['pose_bl'].cpu().numpy()

        # H36M 16 → COCO 17
        original_coco_batch = np.tile(coco_pose, (num_augmentations, 1, 1))
        augmented_coco = h36m16_to_coco17(augmented_h36m, original_coco=original_coco_batch)

        # 組裝增強類型列表
        applied = ['BA (GAN)', 'BL (GAN)']
        if include_rt:
            applied.append(f'RT (GAN, max {rt_max_angle}deg)')

        # 組裝輸出
        output = {
            'augmented_poses': [augmented_coco[i].tolist() for i in range(num_augmentations)],
            'original_h36m': h36m_pose.tolist(),
            'weights_loaded': self.weights_loaded,
            'augmentation_details': [],
        }

        for i in range(num_augmentations):
            detail = {
                'index': i,
                'applied_augmentations': applied,
                'method': 'PoseAug GAN',
            }

            # 添加 RT 參數信息
            r, t = results['rt']
            detail['rotation_axis_angle'] = r[i].cpu().numpy().tolist()
            detail['translation'] = t[i].cpu().numpy().tolist()

            # BL 比率
            detail['bone_length_ratios'] = results['blr'][i].cpu().numpy().tolist()

            output['augmentation_details'].append(detail)

        if return_intermediate:
            pose_ba_coco = h36m16_to_coco17(results['pose_ba'].cpu().numpy(), original_coco=original_coco_batch)
            pose_bl_coco = h36m16_to_coco17(results['pose_bl'].cpu().numpy(), original_coco=original_coco_batch)
            output['intermediate'] = {
                'pose_ba': [pose_ba_coco[i].tolist() for i in range(num_augmentations)],
                'pose_bl': [pose_bl_coco[i].tolist() for i in range(num_augmentations)],
            }

        return output

    def augment_batch(self, coco_poses: List[np.ndarray], num_augmentations_per_pose: int = 5) -> List[Dict]:
        """
        批量增強多個姿態

        Args:
            coco_poses: list of (17, 3) COCO 格式姿態
            num_augmentations_per_pose: 每個姿態生成的增強數量

        Returns:
            list of augmentation results
        """
        results = []
        for pose in coco_poses:
            result = self.augment(np.array(pose), num_augmentations=num_augmentations_per_pose)
            results.append(result)
        return results

    def get_status(self) -> Dict:
        """返回 GAN 模型狀態"""
        return {
            'available': True,
            'weights_loaded': self.weights_loaded,
            'device': str(self.device),
            'generator_params': sum(p.numel() for p in self.generator.parameters()),
            'mode': 'pretrained' if self.weights_loaded else 'random_init',
        }
