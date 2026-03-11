"""
多視角影片處理模組 - 用於 3D 姿態重建
處理多個相機視角的影片，提取同步的 2D 姿態座標
"""
import cv2
import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import os

# 導入姿態檢測器
try:
    from pose_detector import PoseDetector
    POSE_AVAILABLE = True
except ImportError:
    POSE_AVAILABLE = False
    print("⚠️ 姿態檢測模組不可用")


@dataclass
class CameraConfig:
    """相機配置"""
    view_name: str          # 視角名稱 (front, back, left, right)
    angle_degrees: float    # 相對於正面的角度 (0, 90, 180, 270)
    distance: float         # 到場地中心的距離 (公尺)
    height: float           # 相機高度 (公尺)
    video_path: str         # 影片路徑


@dataclass
class MultiViewFrame:
    """多視角單幀數據"""
    frame_number: int
    timestamp: float
    views: Dict[str, Dict]  # view_name -> pose_data


class MultiViewProcessor:
    """
    多視角影片處理器
    負責從多個相機視角提取同步的 2D 姿態座標
    """

    # 預設相機角度配置
    DEFAULT_CAMERA_ANGLES = {
        'front': 0,
        'right': 90,
        'back': 180,
        'left': 270
    }

    def __init__(self, pose_model_size: str = 'l'):
        """
        初始化多視角處理器

        Args:
            pose_model_size: 姿態模型大小 ('n', 's', 'm', 'l', 'x')
        """
        self.pose_detector = None
        self.pose_model_size = pose_model_size

        if POSE_AVAILABLE:
            print(f"🎥 初始化多視角處理器 (YOLO11{pose_model_size}-pose)...")
            self.pose_detector = PoseDetector(model_size=pose_model_size)
            print("✅ 多視角處理器就緒")
        else:
            raise RuntimeError("姿態檢測模組不可用，無法初始化多視角處理器")

        self.camera_configs: List[CameraConfig] = []
        self.video_captures: Dict[str, cv2.VideoCapture] = {}

    def add_camera(self, view_name: str, video_path: str,
                   angle_degrees: Optional[float] = None,
                   distance: float = 5.0, height: float = 1.5) -> bool:
        """
        添加相機視角

        Args:
            view_name: 視角名稱 (front, back, left, right 或自訂)
            video_path: 影片檔案路徑
            angle_degrees: 相機角度 (若為預設視角名稱則自動設定)
            distance: 相機到中心距離 (公尺)
            height: 相機高度 (公尺)

        Returns:
            bool: 是否成功添加
        """
        if not os.path.exists(video_path):
            print(f"❌ 影片檔案不存在: {video_path}")
            return False

        # 自動設定預設角度
        if angle_degrees is None:
            angle_degrees = self.DEFAULT_CAMERA_ANGLES.get(view_name.lower(), 0)

        config = CameraConfig(
            view_name=view_name,
            angle_degrees=angle_degrees,
            distance=distance,
            height=height,
            video_path=video_path
        )

        self.camera_configs.append(config)
        print(f"📷 添加相機: {view_name} (角度: {angle_degrees}°, 距離: {distance}m)")

        return True

    def get_video_info(self, video_path: str) -> Dict:
        """獲取影片資訊"""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return {}

        info = {
            'fps': cap.get(cv2.CAP_PROP_FPS),
            'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            'total_frames': int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        }
        cap.release()
        return info

    def synchronize_videos(self) -> Tuple[float, int]:
        """
        同步多個影片
        返回共同的 FPS 和總幀數
        """
        if not self.camera_configs:
            raise ValueError("沒有添加任何相機")

        fps_list = []
        frame_counts = []

        for config in self.camera_configs:
            info = self.get_video_info(config.video_path)
            if info:
                fps_list.append(info['fps'])
                frame_counts.append(info['total_frames'])

        # 使用最小 FPS 和最小幀數作為共同基準
        common_fps = min(fps_list)
        common_frames = min(frame_counts)

        print(f"📊 同步設定: FPS={common_fps}, 總幀數={common_frames}")

        return common_fps, common_frames

    def process_multiview(self, output_dir: Optional[str] = None,
                          progress_callback: Optional[callable] = None) -> Dict:
        """
        處理多視角影片，提取同步的 2D 姿態座標

        Args:
            output_dir: 輸出目錄 (用於儲存處理後的影片)
            progress_callback: 進度回調函數 (frame_number, total_frames)

        Returns:
            Dict: 多視角處理結果
        """
        if not self.camera_configs:
            raise ValueError("沒有添加任何相機")

        if len(self.camera_configs) < 2:
            print("⚠️ 少於 2 個視角，3D 重建效果可能有限")

        # 同步影片
        common_fps, total_frames = self.synchronize_videos()

        # 開啟所有影片
        caps = {}
        for config in self.camera_configs:
            cap = cv2.VideoCapture(config.video_path)
            if not cap.isOpened():
                raise ValueError(f"無法開啟影片: {config.video_path}")
            caps[config.view_name] = cap

        # 處理結果
        results = {
            'video_info': {
                'fps': common_fps,
                'total_frames': total_frames,
                'num_views': len(self.camera_configs)
            },
            'camera_configs': [
                {
                    'view_name': c.view_name,
                    'angle_degrees': c.angle_degrees,
                    'distance': c.distance,
                    'height': c.height
                }
                for c in self.camera_configs
            ],
            'frames': []  # 每幀的多視角數據
        }

        print(f"🎬 開始處理 {len(self.camera_configs)} 個視角的影片...")

        for frame_idx in range(total_frames):
            frame_data = {
                'frame_number': frame_idx,
                'timestamp': frame_idx / common_fps,
                'views': {}
            }

            # 讀取並處理每個視角
            for config in self.camera_configs:
                cap = caps[config.view_name]
                ret, frame = cap.read()

                if not ret:
                    continue

                # 檢測姿態
                poses = self.pose_detector.detect_pose(frame)

                if poses:
                    # 取最大的人（主要球員）
                    main_pose = max(poses, key=lambda p:
                        (p['bbox'][2] - p['bbox'][0]) * (p['bbox'][3] - p['bbox'][1])
                    )

                    # 使用 keypoints_raw (numpy array [17, 3]) 而非 keypoints (dict)
                    keypoints_raw = main_pose.get('keypoints_raw')
                    if keypoints_raw is not None:
                        keypoints_list = keypoints_raw.tolist() if hasattr(keypoints_raw, 'tolist') else keypoints_raw
                    else:
                        # 備用：從 dict 轉換為 list 格式
                        keypoints_list = []
                        for i in range(17):
                            kp = main_pose['keypoints'].get(list(main_pose['keypoints'].keys())[i] if i < len(main_pose['keypoints']) else None)
                            if kp:
                                keypoints_list.append([kp['x'], kp['y'], kp['confidence']])
                            else:
                                keypoints_list.append([0, 0, 0])

                    frame_data['views'][config.view_name] = {
                        'keypoints': keypoints_list,
                        'confidence': main_pose['confidence'],
                        'bbox': main_pose['bbox']
                    }
                else:
                    frame_data['views'][config.view_name] = None

            results['frames'].append(frame_data)

            # 進度回調
            if progress_callback:
                progress_callback(frame_idx, total_frames)

            # 進度顯示
            if frame_idx % 30 == 0:
                progress = (frame_idx / total_frames) * 100
                print(f"   處理進度: {progress:.1f}%")

        # 關閉所有影片
        for cap in caps.values():
            cap.release()

        # 統計
        frames_with_all_views = sum(
            1 for f in results['frames']
            if len(f['views']) == len(self.camera_configs) and all(v is not None for v in f['views'].values())
        )

        results['statistics'] = {
            'frames_processed': total_frames,
            'frames_with_all_views': frames_with_all_views,
            'detection_rate': frames_with_all_views / total_frames if total_frames > 0 else 0
        }

        print(f"✅ 處理完成: {frames_with_all_views}/{total_frames} 幀在所有視角都檢測到姿態")

        return results

    def get_2d_coordinates_for_frame(self, frame_data: Dict, view_name: str) -> Optional[np.ndarray]:
        """
        獲取指定視角指定幀的 2D 座標

        Returns:
            np.ndarray: shape (17, 3) - (x, y, confidence) 或 None
        """
        if view_name not in frame_data.get('views', {}):
            return None

        view_data = frame_data['views'][view_name]
        if view_data is None:
            return None

        return np.array(view_data['keypoints'])


def create_default_camera_setup(video_paths: Dict[str, str],
                                 distance: float = 5.0,
                                 height: float = 1.5) -> MultiViewProcessor:
    """
    建立預設的四視角相機設置

    Args:
        video_paths: 視角名稱到影片路徑的映射
                     例如: {'front': 'front.mp4', 'right': 'right.mp4', ...}
        distance: 相機到中心距離
        height: 相機高度

    Returns:
        MultiViewProcessor: 配置好的處理器
    """
    processor = MultiViewProcessor()

    for view_name, video_path in video_paths.items():
        processor.add_camera(
            view_name=view_name,
            video_path=video_path,
            distance=distance,
            height=height
        )

    return processor
