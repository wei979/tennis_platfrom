/**
 * PoseAug 姿態增強頁面
 * 流程: 上傳影片 → VideoPose3D 提取 3D → 選擇增強方式 → 一鍵執行 → 瀏覽結果
 */
import React, { useState, useEffect, useRef, useMemo } from 'react';
import axios from 'axios';
import * as THREE from 'three';
import { Canvas, useFrame } from '@react-three/fiber';
import { OrbitControls, Line } from '@react-three/drei';

const API_BASE_URL = 'http://localhost:5000/api';

// ============================================================
// 類型定義
// ============================================================

interface PoseAugStatus {
  available: boolean;
  message: string;
  keypoint_names: string[];
  skeleton_bones: [number, number][];
  gan_available?: boolean;
  gan_status?: {
    available: boolean;
    weights_loaded: boolean;
    device: string;
    generator_params: number;
    mode: string;
  };
  videopose3d_available?: boolean;
  videopose3d_status?: {
    available: boolean;
    weights_loaded: boolean;
    device: string;
    model_params: number;
    model_name: string;
  };
}

interface HandInfo {
  matched_hand: 'left' | 'right' | 'unknown';
  racket_dir_3d: number[];     // [x,y,z] 拍柄方向
  palm_normal_3d: number[];    // [x,y,z] 拍面法線
  racket_bbox?: number[];      // [x1,y1,x2,y2] YOLO bbox
  racket_conf?: number;        // 偵測信心度
}

interface BallInfo {
  center_2d: number[];
  bbox: number[];
  confidence: number;
  size: number[];
  position_3d?: number[];  // [x, y, z]
}

interface ExtractedPose {
  frame_number: number;
  timestamp: number;
  pose_2d: number[][];
  pose_3d: number[][];
  root_position?: number[];
  confidence: number[];
  lifting_method?: string;
  hand_info?: HandInfo | null;
  ball_info?: BallInfo | null;
}

// ============================================================
// 3D 渲染組件
// ============================================================

const TennisBall3D: React.FC<{ position: number[] }> = ({ position }) => {
  if (!position || position.length < 3) return null;
  return (
    <mesh position={[position[0], position[1], position[2]]}>
      <sphereGeometry args={[0.03, 16, 16]} />
      <meshStandardMaterial color="#CCFF00" emissive="#556B00" emissiveIntensity={0.3} />
    </mesh>
  );
};

// COCO 17: 7=left_elbow, 8=right_elbow, 9=left_wrist, 10=right_wrist
const TennisRacket3D: React.FC<{
  wrist: number[];
  elbow: number[];
  handInfo?: HandInfo | null;  // MediaPipe hand data (optional)
}> = ({ wrist, elbow, handInfo }) => {
  const { handlePos, handleQuat, headPos, headQuat } = useMemo(() => {
    const w = new THREE.Vector3(wrist[0], wrist[1], wrist[2]);
    const e = new THREE.Vector3(elbow[0], elbow[1], elbow[2]);
    const forearm = new THREE.Vector3().subVectors(w, e).normalize();

    let racketDir: THREE.Vector3;
    let faceNormal: THREE.Vector3;

    if (handInfo?.racket_dir_3d && handInfo?.palm_normal_3d) {
      // === MediaPipe Hands 數據: 用真實手指方向 ===
      racketDir = new THREE.Vector3(
        handInfo.racket_dir_3d[0],
        handInfo.racket_dir_3d[1],
        handInfo.racket_dir_3d[2]
      ).normalize();
      faceNormal = new THREE.Vector3(
        handInfo.palm_normal_3d[0],
        handInfo.palm_normal_3d[1],
        handInfo.palm_normal_3d[2]
      ).normalize();
    } else {
      // === Fallback: 幾何估計 (前臂 + 30° 手腕屈曲) ===
      const worldUp = new THREE.Vector3(0, 1, 0);
      const bendAxis = new THREE.Vector3().crossVectors(forearm, worldUp).normalize();
      if (bendAxis.length() < 0.1) {
        bendAxis.crossVectors(forearm, new THREE.Vector3(0, 0, 1)).normalize();
      }
      const bendQuat = new THREE.Quaternion().setFromAxisAngle(
        bendAxis, THREE.MathUtils.degToRad(30)
      );
      racketDir = forearm.clone().applyQuaternion(bendQuat).normalize();
      faceNormal = bendAxis.clone().normalize();
    }

    // 拍柄位置 & 旋轉
    const handleHalfLen = 0.13;
    const hPos = new THREE.Vector3().copy(w).addScaledVector(racketDir, handleHalfLen);
    const hQuat = new THREE.Quaternion().setFromUnitVectors(
      new THREE.Vector3(0, 1, 0), racketDir
    );

    // 拍面位置
    const headDist = 0.34;
    const headP = new THREE.Vector3().copy(w).addScaledVector(racketDir, headDist);

    // 拍面旋轉
    const hQuatHead = new THREE.Quaternion().setFromUnitVectors(
      new THREE.Vector3(0, 0, 1), faceNormal
    );

    return { handlePos: hPos, handleQuat: hQuat, headPos: headP, headQuat: hQuatHead };
  }, [wrist, elbow, handInfo]);

  return (
    <group>
      {/* 拍柄 */}
      <mesh position={handlePos} quaternion={handleQuat}>
        <cylinderGeometry args={[0.008, 0.012, 0.26, 8]} />
        <meshStandardMaterial color="#8B4513" />
      </mesh>
      {/* 拍面框 */}
      <mesh position={headPos} quaternion={headQuat}>
        <ringGeometry args={[0.07, 0.085, 24]} />
        <meshStandardMaterial color="#FF4444" side={THREE.DoubleSide} />
      </mesh>
      {/* 拍面網 */}
      <mesh position={headPos} quaternion={headQuat}>
        <circleGeometry args={[0.07, 24]} />
        <meshStandardMaterial color="#FFFFFF" transparent opacity={0.15} side={THREE.DoubleSide} />
      </mesh>
    </group>
  );
};

const Skeleton3D: React.FC<{
  pose: number[][];
  connections: [number, number][];
  color?: string;
  opacity?: number;
  racketHand?: 'left' | 'right' | 'none';
  handInfo?: HandInfo | null;
}> = ({ pose, connections, color = '#ffffff', opacity = 1, racketHand = 'none', handInfo }) => {
  if (!pose || pose.length !== 17) return null;

  // COCO: 7=left_elbow, 9=left_wrist, 8=right_elbow, 10=right_wrist
  const racketWrist = racketHand === 'right' ? pose[10] : racketHand === 'left' ? pose[9] : null;
  const racketElbow = racketHand === 'right' ? pose[8] : racketHand === 'left' ? pose[7] : null;
  // 只在匹配的手傳遞 handInfo
  const activeHandInfo = handInfo?.matched_hand === racketHand ? handInfo : null;

  return (
    <group>
      {pose.map((point, idx) => {
        if (!point || point.length < 3) return null;
        return (
          <mesh key={`kp-${idx}`} position={[point[0], point[1], point[2]]}>
            <sphereGeometry args={[0.02, 16, 16]} />
            <meshStandardMaterial color={color} transparent opacity={opacity} />
          </mesh>
        );
      })}
      {connections.map(([startIdx, endIdx], idx) => {
        const start = pose[startIdx];
        const end = pose[endIdx];
        if (!start || !end || start.length < 3 || end.length < 3) return null;
        return (
          <Line
            key={`bone-${idx}`}
            points={[[start[0], start[1], start[2]], [end[0], end[1], end[2]]]}
            color={color}
            lineWidth={2}
            transparent
            opacity={opacity}
          />
        );
      })}
      {racketWrist && racketElbow && racketWrist.length >= 3 && racketElbow.length >= 3 && (
        <TennisRacket3D wrist={racketWrist} elbow={racketElbow} handInfo={activeHandInfo} />
      )}
    </group>
  );
};

const AnimatedPoseViewer: React.FC<{
  poses: number[][][];
  connections: [number, number][];
  playing: boolean;
  fps: number;
  racketHand?: 'left' | 'right' | 'none';
  handInfo?: HandInfo | null;
}> = ({ poses, connections, playing, fps, racketHand = 'none', handInfo }) => {
  const [currentIndex, setCurrentIndex] = useState(0);
  const lastTimeRef = useRef(0);

  useFrame((state) => {
    if (!playing || poses.length === 0) return;
    const elapsed = state.clock.getElapsedTime();
    if (elapsed - lastTimeRef.current >= 1 / fps) {
      lastTimeRef.current = elapsed;
      setCurrentIndex((prev) => (prev + 1) % poses.length);
    }
  });

  if (poses.length === 0) return null;
  return <Skeleton3D pose={poses[currentIndex]} connections={connections} color="#00ff88" opacity={1} racketHand={racketHand} handInfo={handInfo} />;
};

// ============================================================
// 主頁面
// ============================================================

const PoseAugPage: React.FC = () => {
  // 系統狀態
  const [status, setStatus] = useState<PoseAugStatus | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [processing, setProcessing] = useState(false);

  // Step 1: 影片上傳
  const [videoFile, setVideoFile] = useState<File | null>(null);
  const [videoPreview, setVideoPreview] = useState<string | null>(null);
  const videoRef = useRef<HTMLVideoElement>(null);

  // Step 2: 提取的姿態
  const [extractedPoses, setExtractedPoses] = useState<ExtractedPose[]>([]);
  const [extractedFrameIndex, setExtractedFrameIndex] = useState<number>(0);
  const [extractionProgress, setExtractionProgress] = useState<string>('');
  const [liftingMethod, setLiftingMethod] = useState<string>('');

  // Step 3: 增強設定
  const [useGanMode, setUseGanMode] = useState<boolean>(true); // 默認 GAN
  const [ganIncludeRT, setGanIncludeRT] = useState<boolean>(false);
  const [numAugmentations, setNumAugmentations] = useState<number>(5);
  const [rulePreset, setRulePreset] = useState<string>('moderate');

  // Step 4: 結果
  const [originalPose, setOriginalPose] = useState<number[][] | null>(null);
  const [augmentedFrames, setAugmentedFrames] = useState<any[] | null>(null);
  const [resultFrameIndex, setResultFrameIndex] = useState<number>(0);
  const [resultAugIndex, setResultAugIndex] = useState<number>(0);

  // 視圖控制
  const [showOriginal, setShowOriginal] = useState(true);
  const [showAugmented, setShowAugmented] = useState(true);
  const [playAnimation, setPlayAnimation] = useState(false);
  const [racketHand, setRacketHand] = useState<'left' | 'right' | 'none'>('right');

  // 載入狀態
  useEffect(() => {
    const fetchStatus = async () => {
      try {
        const response = await axios.get(`${API_BASE_URL}/poseaug-status`);
        setStatus(response.data);
      } catch {
        setError('無法連接到後端服務');
      } finally {
        setLoading(false);
      }
    };
    fetchStatus();
  }, []);

  // 將 root_position 偏移加到姿態上
  const applyRootOffset = (pose: number[][], rootPos?: number[]): number[][] => {
    if (!rootPos || rootPos.length < 3) return pose;
    return pose.map((kp) => [kp[0] + rootPos[0], kp[1] + rootPos[1], kp[2] + rootPos[2]]);
  };

  // 從提取的姿態選擇並套用全局位置
  const getPoseWithPosition = (index: number): number[][] | null => {
    if (index < 0 || index >= extractedPoses.length) return null;
    const frame = extractedPoses[index];
    return applyRootOffset(frame.pose_3d, frame.root_position);
  };

  // 骨骼連接
  const skeletonConnections: [number, number][] = status?.skeleton_bones || [
    [0, 1], [0, 2], [1, 3], [2, 4],
    [5, 6], [5, 7], [7, 9], [6, 8], [8, 10],
    [5, 11], [6, 12], [11, 12],
    [11, 13], [13, 15], [12, 14], [14, 16],
  ];

  // ---- Step 1: 影片操作 ----

  const handleVideoSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      setVideoFile(file);
      setVideoPreview(URL.createObjectURL(file));
      setExtractedPoses([]);
      setOriginalPose(null);
      setAugmentedFrames(null);
      setLiftingMethod('');
    }
  };

  // ---- Step 2: 提取姿態 ----

  const extractPoses = async () => {
    if (!videoFile) return;
    try {
      setProcessing(true);
      setError(null);
      setExtractionProgress('正在上傳影片...');
      setAugmentedFrames(null);

      const formData = new FormData();
      formData.append('video', videoFile);
      formData.append('max_frames', '50');
      formData.append('frame_skip', '3');

      setExtractionProgress('正在分析姿態 (VideoPose3D)...');

      const response = await axios.post(
        `${API_BASE_URL}/poseaug/extract-from-video`,
        formData,
        { headers: { 'Content-Type': 'multipart/form-data' }, timeout: 120000 }
      );

      if (response.data.success) {
        setExtractedPoses(response.data.poses);
        setExtractedFrameIndex(0);
        setLiftingMethod(response.data.lifting_method || 'heuristic');
        if (response.data.poses.length > 0) {
          const p = response.data.poses[0];
          setOriginalPose(applyRootOffset(p.pose_3d, p.root_position));
        }
        const methodName = response.data.lifting_method === 'VideoPose3D'
          ? 'VideoPose3D' : '啟發式';
        const ballCount = response.data.poses.filter((p: any) => p.ball_info).length;
        const ballStr = ballCount > 0 ? `, 球${ballCount}幀` : '';
        setExtractionProgress(`${response.data.poses.length} 幀 (${methodName}${ballStr})`);
      } else {
        setError(response.data.error || '姿態提取失敗');
      }
    } catch (err: any) {
      setError(err.response?.data?.error || '影片處理失敗');
    } finally {
      setProcessing(false);
    }
  };

  // 選擇幀 (同步影片)
  const selectFrame = (index: number) => {
    if (index >= 0 && index < extractedPoses.length) {
      setExtractedFrameIndex(index);
      setOriginalPose(getPoseWithPosition(index));
      if (videoRef.current && extractedPoses[index]) {
        videoRef.current.currentTime = extractedPoses[index].timestamp;
      }
    }
  };

  // 影片時間更新同步
  const handleVideoTimeUpdate = () => {
    if (!videoRef.current || extractedPoses.length === 0) return;
    const currentTime = videoRef.current.currentTime;
    let closestIndex = 0;
    let minDiff = Math.abs(extractedPoses[0].timestamp - currentTime);
    for (let i = 1; i < extractedPoses.length; i++) {
      const diff = Math.abs(extractedPoses[i].timestamp - currentTime);
      if (diff < minDiff) { minDiff = diff; closestIndex = i; }
    }
    if (closestIndex !== extractedFrameIndex) {
      setExtractedFrameIndex(closestIndex);
      setOriginalPose(getPoseWithPosition(closestIndex));
    }
  };

  // ---- Step 3: 執行增強 (一鍵增強所有幀) ----

  const runAugmentation = async () => {
    if (extractedPoses.length === 0) return;

    try {
      setProcessing(true);
      setError(null);

      const allPoses = extractedPoses.map((p) => p.pose_3d);

      if (useGanMode) {
        // GAN 模式
        const response = await axios.post(
          `${API_BASE_URL}/poseaug/gan-batch-augment`,
          { poses: allPoses, num_augmentations_per_pose: numAugmentations, include_rt: ganIncludeRT, rt_max_angle: 15.0 },
          { timeout: 300000 }
        );
        if (response.data.success) {
          setAugmentedFrames(response.data.augmented_frames);
          setResultFrameIndex(0);
          setResultAugIndex(0);
        } else {
          setError(response.data.error || 'GAN 增強失敗');
        }
      } else {
        // 規則模式
        const presetConfigs: { [key: string]: any } = {
          conservative: { bl_scale_range: [0.95, 1.05], ba_angle_range: [-5, 5], rt_angle_range: [-10, 10], pos_shift_range: [-0.02, 0.02] },
          moderate: { bl_scale_range: [0.9, 1.1], ba_angle_range: [-15, 15], rt_angle_range: [-30, 30], pos_shift_range: [-0.05, 0.05] },
          aggressive: { bl_scale_range: [0.8, 1.2], ba_angle_range: [-25, 25], rt_angle_range: [-45, 45], pos_shift_range: [-0.1, 0.1] },
        };
        const response = await axios.post(
          `${API_BASE_URL}/poseaug/batch-augment-all`,
          {
            poses: allPoses,
            augmentation_types: ['BL', 'BA', 'RT', 'POS'],
            num_augmentations_per_pose: numAugmentations,
            config: presetConfigs[rulePreset],
          },
          { timeout: 300000 }
        );
        if (response.data.success) {
          setAugmentedFrames(response.data.augmented_frames);
          setResultFrameIndex(0);
          setResultAugIndex(0);
        } else {
          setError(response.data.error || '增強失敗');
        }
      }
    } catch (err: any) {
      setError(err.response?.data?.error || '增強請求失敗');
    } finally {
      setProcessing(false);
    }
  };

  // 切換結果幀時更新 3D 預覽
  const selectResultFrame = (frameIdx: number) => {
    if (!augmentedFrames || frameIdx < 0 || frameIdx >= augmentedFrames.length) return;
    setResultFrameIndex(frameIdx);
    setResultAugIndex(0);
    // 套用對應幀的全局位置
    const rootPos = extractedPoses[frameIdx]?.root_position;
    setOriginalPose(applyRootOffset(augmentedFrames[frameIdx].original_pose, rootPos));
    if (videoRef.current && extractedPoses[frameIdx]) {
      videoRef.current.currentTime = extractedPoses[frameIdx].timestamp;
    }
  };

  // 取得當前增強姿態 (也要套用全局位置)
  const getRootPosForCurrentResult = (): number[] | undefined => {
    return extractedPoses[resultFrameIndex]?.root_position;
  };

  // 當前顯示的增強姿態
  const currentAugPoseRaw = augmentedFrames
    ? augmentedFrames[resultFrameIndex]?.augmented_poses?.[resultAugIndex] || null
    : null;
  const currentAugPose = currentAugPoseRaw
    ? applyRootOffset(currentAugPoseRaw, getRootPosForCurrentResult())
    : null;

  // 當前幀的手部資訊
  const currentHandInfo: HandInfo | null =
    extractedPoses[extractedFrameIndex]?.hand_info || null;

  // 當前幀的球體 3D 位置
  const currentBallPos: number[] | null =
    extractedPoses[extractedFrameIndex]?.ball_info?.position_3d || null;

  // ---- UI ----

  if (loading) {
    return <div className="flex items-center justify-center min-h-screen"><div className="text-xl">載入中...</div></div>;
  }
  if (!status?.available) {
    return (
      <div className="max-w-4xl mx-auto p-6">
        <div className="bg-yellow-100 border-l-4 border-yellow-500 p-4 rounded">
          <h2 className="text-lg font-semibold text-yellow-700">PoseAug 模組未啟用</h2>
          <p className="text-yellow-600">{status?.message || '請確認後端已正確載入'}</p>
        </div>
      </div>
    );
  }

  const hasExtracted = extractedPoses.length > 0;
  const hasResults = augmentedFrames !== null && augmentedFrames.length > 0;

  return (
    <div className="max-w-7xl mx-auto p-6">
      {/* 標題 */}
      <div className="mb-6">
        <h1 className="text-2xl font-bold text-gray-800">PoseAug 姿態增強</h1>
        <p className="text-gray-500 text-sm mt-1">
          上傳影片 → VideoPose3D 提取 3D 座標 → GAN/規則增強 → 瀏覽結果
        </p>
      </div>

      {/* 錯誤提示 */}
      {error && (
        <div className="bg-red-100 border-l-4 border-red-500 p-3 rounded mb-4 flex justify-between items-center">
          <p className="text-red-700 text-sm">{error}</p>
          <button onClick={() => setError(null)} className="text-red-400 hover:text-red-600 text-xs">關閉</button>
        </div>
      )}

      <div className="grid grid-cols-1 lg:grid-cols-5 gap-6">

        {/* ===== 左側: 操作流程 (2 cols) ===== */}
        <div className="lg:col-span-2 space-y-4">

          {/* Step 1: 上傳影片 */}
          <div className="bg-white rounded-lg shadow p-4">
            <div className="flex items-center gap-2 mb-3">
              <span className={`w-6 h-6 rounded-full flex items-center justify-center text-xs font-bold ${videoFile ? 'bg-green-500 text-white' : 'bg-gray-300 text-gray-600'}`}>1</span>
              <h3 className="font-semibold text-gray-800">上傳影片</h3>
            </div>

            <input
              type="file"
              accept="video/*"
              onChange={handleVideoSelect}
              className="w-full text-sm text-gray-500 file:mr-3 file:py-1.5 file:px-3 file:rounded-lg file:border-0 file:bg-blue-50 file:text-blue-700 hover:file:bg-blue-100 mb-3"
            />

            {videoPreview && (
              <div className="relative aspect-video bg-black rounded-lg overflow-hidden">
                <video
                  ref={videoRef}
                  src={videoPreview}
                  className="w-full h-full object-contain"
                  controls
                  muted
                  onTimeUpdate={handleVideoTimeUpdate}
                />
              </div>
            )}
          </div>

          {/* Step 2: 提取 3D 姿態 */}
          <div className="bg-white rounded-lg shadow p-4">
            <div className="flex items-center gap-2 mb-3">
              <span className={`w-6 h-6 rounded-full flex items-center justify-center text-xs font-bold ${hasExtracted ? 'bg-green-500 text-white' : 'bg-gray-300 text-gray-600'}`}>2</span>
              <h3 className="font-semibold text-gray-800">提取 3D 姿態</h3>
              {status?.videopose3d_available && (
                <span className="text-xs bg-indigo-100 text-indigo-700 px-2 py-0.5 rounded">VideoPose3D</span>
              )}
            </div>

            <button
              onClick={extractPoses}
              disabled={!videoFile || processing}
              className="w-full bg-blue-600 hover:bg-blue-700 text-white py-2 px-4 rounded-lg transition-colors disabled:opacity-50 disabled:cursor-not-allowed text-sm font-medium"
            >
              {processing && !hasExtracted ? extractionProgress || '處理中...' : '提取 3D 姿態'}
            </button>

            {/* 提取結果 + 幀選擇器 */}
            {hasExtracted && (
              <div className="mt-3 p-3 bg-green-50 rounded-lg">
                <div className="flex items-center justify-between mb-2">
                  <span className="text-sm font-medium text-green-700">
                    {extractionProgress}
                  </span>
                  {liftingMethod === 'VideoPose3D' && (
                    <span className="text-xs bg-green-200 text-green-800 px-1.5 py-0.5 rounded">深度學習</span>
                  )}
                </div>
                <div className="flex items-center gap-2">
                  <input
                    type="range"
                    min={0}
                    max={extractedPoses.length - 1}
                    value={extractedFrameIndex}
                    onChange={(e) => selectFrame(parseInt(e.target.value))}
                    className="flex-1"
                  />
                  <span className="text-xs font-medium text-green-800 w-16 text-right">
                    {extractedFrameIndex + 1}/{extractedPoses.length}
                  </span>
                </div>
                <p className="text-xs text-green-500 mt-1">拖動滑桿或播放影片，兩者自動同步</p>
              </div>
            )}
          </div>

          {/* Step 3: 增強設定 + 執行 */}
          <div className={`bg-white rounded-lg shadow p-4 ${!hasExtracted ? 'opacity-50 pointer-events-none' : ''}`}>
            <div className="flex items-center gap-2 mb-3">
              <span className={`w-6 h-6 rounded-full flex items-center justify-center text-xs font-bold ${hasResults ? 'bg-green-500 text-white' : 'bg-gray-300 text-gray-600'}`}>3</span>
              <h3 className="font-semibold text-gray-800">增強設定</h3>
            </div>

            {/* 增強模式切換 */}
            <div className="flex gap-2 mb-3">
              <button
                onClick={() => setUseGanMode(true)}
                className={`flex-1 py-2 px-3 rounded-lg transition-colors text-sm font-medium ${
                  useGanMode ? 'bg-indigo-600 text-white' : 'bg-gray-100 hover:bg-gray-200 text-gray-700'
                }`}
              >
                GAN 增強
              </button>
              <button
                onClick={() => setUseGanMode(false)}
                className={`flex-1 py-2 px-3 rounded-lg transition-colors text-sm font-medium ${
                  !useGanMode ? 'bg-blue-600 text-white' : 'bg-gray-100 hover:bg-gray-200 text-gray-700'
                }`}
              >
                規則增強
              </button>
            </div>

            {/* GAN 模式選項 */}
            {useGanMode && (
              <div className="p-3 bg-indigo-50 rounded-lg text-xs text-indigo-700 mb-3">
                <p className="mb-2">神經網路自動生成合理的骨骼角度 + 長度變化</p>
                <label className="flex items-center gap-2 cursor-pointer">
                  <input type="checkbox" checked={ganIncludeRT} onChange={(e) => setGanIncludeRT(e.target.checked)} className="rounded" />
                  包含視角旋轉 (RT, 限制 15°)
                </label>
              </div>
            )}

            {/* 規則模式選項 */}
            {!useGanMode && (
              <div className="flex gap-1 mb-3">
                {[
                  { key: 'conservative', label: '保守' },
                  { key: 'moderate', label: '適中' },
                  { key: 'aggressive', label: '激進' },
                ].map(({ key, label }) => (
                  <button
                    key={key}
                    onClick={() => setRulePreset(key)}
                    className={`flex-1 py-1.5 px-2 rounded text-xs font-medium transition-colors ${
                      rulePreset === key ? 'bg-blue-600 text-white' : 'bg-gray-100 hover:bg-gray-200 text-gray-600'
                    }`}
                  >
                    {label}
                  </button>
                ))}
              </div>
            )}

            {/* 生成數量 */}
            <div className="flex items-center gap-3 mb-4">
              <span className="text-sm text-gray-600 whitespace-nowrap">每幀變體:</span>
              <input
                type="range" min="1" max="20" value={numAugmentations}
                onChange={(e) => setNumAugmentations(parseInt(e.target.value))}
                className="flex-1"
              />
              <span className="text-sm font-bold w-6 text-center">{numAugmentations}</span>
            </div>

            {/* 統計預覽 */}
            {hasExtracted && (
              <p className="text-xs text-gray-500 mb-3">
                {extractedPoses.length} 幀 x {numAugmentations} 變體 = <strong>{extractedPoses.length * numAugmentations}</strong> 個增強姿態
              </p>
            )}

            {/* 執行按鈕 */}
            <button
              onClick={runAugmentation}
              disabled={!hasExtracted || processing}
              className={`w-full py-3 px-4 rounded-lg font-semibold transition-colors disabled:opacity-50 disabled:cursor-not-allowed ${
                useGanMode
                  ? 'bg-indigo-600 hover:bg-indigo-700 text-white'
                  : 'bg-blue-600 hover:bg-blue-700 text-white'
              }`}
            >
              {processing && hasExtracted
                ? '增強處理中...'
                : `${useGanMode ? 'GAN' : '規則'} 增強所有幀`}
            </button>
          </div>

          {/* 系統狀態 (折疊) */}
          <details className="bg-gray-50 rounded-lg p-3">
            <summary className="text-xs font-medium text-gray-500 cursor-pointer">系統狀態</summary>
            <div className="mt-2 text-xs text-gray-600 space-y-1">
              <p>PoseAug: {status?.available ? 'OK' : 'N/A'}</p>
              <p>GAN: {status?.gan_available ? `OK (${status.gan_status?.mode})` : 'N/A'}</p>
              <p>VideoPose3D: {status?.videopose3d_available ? `OK (${(status.videopose3d_status as any)?.model_params?.toLocaleString()} params)` : 'N/A'}</p>
            </div>
          </details>
        </div>

        {/* ===== 右側: 3D 預覽 + 結果 (3 cols) ===== */}
        <div className="lg:col-span-3 space-y-4">

          {/* 3D 預覽 */}
          <div className="bg-white rounded-lg shadow overflow-hidden">
            <div className="p-3 bg-gray-50 border-b flex items-center justify-between">
              <h3 className="font-semibold text-sm">3D 姿態預覽</h3>
              <div className="flex items-center gap-3">
                {/* 持拍手選擇 */}
                <div className="flex items-center gap-1 text-xs">
                  <span className="text-gray-500">拍:</span>
                  {(['right', 'left', 'none'] as const).map((hand) => (
                    <button
                      key={hand}
                      onClick={() => setRacketHand(hand)}
                      className={`px-1.5 py-0.5 rounded transition-colors ${
                        racketHand === hand
                          ? 'bg-orange-500 text-white'
                          : 'bg-gray-200 hover:bg-gray-300 text-gray-600'
                      }`}
                    >
                      {hand === 'right' ? '右' : hand === 'left' ? '左' : '無'}
                    </button>
                  ))}
                  {currentHandInfo && (
                    <span className="text-green-500 ml-1" title="YOLO 球拍偵測">YR</span>
                  )}
                </div>
                <label className="flex items-center gap-1.5 text-xs cursor-pointer">
                  <input type="checkbox" checked={showOriginal} onChange={(e) => setShowOriginal(e.target.checked)} />
                  <span className="text-blue-600">原始</span>
                </label>
                {hasResults && (
                  <label className="flex items-center gap-1.5 text-xs cursor-pointer">
                    <input type="checkbox" checked={showAugmented} onChange={(e) => setShowAugmented(e.target.checked)} />
                    <span className="text-green-600">增強</span>
                  </label>
                )}
              </div>
            </div>
            <div className="h-[420px] bg-gray-900">
              <Canvas camera={{ position: [2, 1.5, 2], fov: 50 }}>
                <ambientLight intensity={0.5} />
                <pointLight position={[10, 10, 10]} />
                <gridHelper args={[4, 20, '#333', '#222']} />

                {/* 網球 */}
                {currentBallPos && <TennisBall3D position={currentBallPos} />}

                {showOriginal && originalPose && (
                  <Skeleton3D pose={originalPose} connections={skeletonConnections} color="#4A90D9" opacity={0.8} racketHand={racketHand} handInfo={currentHandInfo} />
                )}

                {showAugmented && currentAugPose && !playAnimation && (
                  <Skeleton3D pose={currentAugPose} connections={skeletonConnections} color="#00ff88" opacity={1} racketHand={racketHand} handInfo={currentHandInfo} />
                )}

                {showAugmented && hasResults && playAnimation && (
                  <AnimatedPoseViewer
                    poses={(augmentedFrames![resultFrameIndex]?.augmented_poses || []).map(
                      (p: number[][]) => applyRootOffset(p, getRootPosForCurrentResult())
                    )}
                    connections={skeletonConnections}
                    playing={playAnimation}
                    fps={3}
                    racketHand={racketHand}
                    handInfo={currentHandInfo}
                  />
                )}

                <OrbitControls enableDamping dampingFactor={0.05} />
              </Canvas>
            </div>

            {/* 無數據提示 */}
            {!originalPose && (
              <div className="p-6 text-center text-gray-400 text-sm">
                上傳影片並提取姿態後，3D 骨架將顯示在此
              </div>
            )}
          </div>

          {/* 結果瀏覽器 */}
          {hasResults && (
            <div className="bg-white rounded-lg shadow p-4">
              <div className="flex items-center justify-between mb-3">
                <h3 className="font-semibold text-sm">增強結果</h3>
                <div className="flex items-center gap-2">
                  <button
                    onClick={() => setPlayAnimation(!playAnimation)}
                    className={`px-3 py-1 rounded text-xs transition-colors ${
                      playAnimation ? 'bg-red-500 text-white' : 'bg-gray-200 hover:bg-gray-300 text-gray-700'
                    }`}
                  >
                    {playAnimation ? '停止' : '播放'}
                  </button>
                  <span className="text-xs text-gray-500">
                    {augmentedFrames!.length} 幀 x {augmentedFrames![0]?.augmented_poses?.length || numAugmentations} 變體
                  </span>
                </div>
              </div>

              {/* 幀選擇 */}
              <div className="space-y-3">
                <div className="flex items-center gap-2">
                  <span className="text-xs text-gray-500 w-8">幀:</span>
                  <input
                    type="range"
                    min={0}
                    max={augmentedFrames!.length - 1}
                    value={resultFrameIndex}
                    onChange={(e) => selectResultFrame(parseInt(e.target.value))}
                    className="flex-1"
                  />
                  <span className="text-xs font-medium w-16 text-right">
                    {resultFrameIndex + 1}/{augmentedFrames!.length}
                  </span>
                </div>

                {/* 變體選擇 */}
                {!playAnimation && augmentedFrames![resultFrameIndex]?.augmented_poses?.length > 1 && (
                  <div className="flex items-center gap-2">
                    <span className="text-xs text-gray-500 w-8">變體:</span>
                    <input
                      type="range"
                      min={0}
                      max={augmentedFrames![resultFrameIndex].augmented_poses.length - 1}
                      value={resultAugIndex}
                      onChange={(e) => setResultAugIndex(parseInt(e.target.value))}
                      className="flex-1"
                    />
                    <span className="text-xs font-medium w-16 text-right">
                      {resultAugIndex + 1}/{augmentedFrames![resultFrameIndex].augmented_poses.length}
                    </span>
                  </div>
                )}
              </div>

              {/* 清除結果 */}
              <button
                onClick={() => { setAugmentedFrames(null); setPlayAnimation(false); }}
                className="mt-3 text-xs text-gray-400 hover:text-gray-600 underline"
              >
                清除結果
              </button>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default PoseAugPage;
