/**
 * 3D 骨架可視化組件
 * 使用 React Three Fiber 渲染 3D 人體骨架
 */
import React, { useRef, useState, useEffect, useMemo } from 'react';
import { Canvas, useFrame } from '@react-three/fiber';
import { OrbitControls, Line, Text } from '@react-three/drei';
import * as THREE from 'three';

// 關鍵點顏色配置
const KEYPOINT_COLORS: { [key: string]: string } = {
  // 頭部 - 藍色
  nose: '#4A90D9',
  left_eye: '#4A90D9',
  right_eye: '#4A90D9',
  left_ear: '#4A90D9',
  right_ear: '#4A90D9',
  // 軀幹 - 綠色
  left_shoulder: '#4CAF50',
  right_shoulder: '#4CAF50',
  left_hip: '#4CAF50',
  right_hip: '#4CAF50',
  // 手臂 - 橙色
  left_elbow: '#FF9800',
  right_elbow: '#FF9800',
  left_wrist: '#FF5722',
  right_wrist: '#FF5722',
  // 腿部 - 紫色
  left_knee: '#9C27B0',
  right_knee: '#9C27B0',
  left_ankle: '#673AB7',
  right_ankle: '#673AB7',
};

// 骨骼連接顏色
const BONE_COLORS = {
  head: '#4A90D9',
  torso: '#4CAF50',
  leftArm: '#FF9800',
  rightArm: '#FF9800',
  leftLeg: '#9C27B0',
  rightLeg: '#9C27B0',
};

interface Keypoint3D {
  index: number;
  name: string;
  position: [number, number, number] | null;
  num_views: number;
}

interface Pose3DFrame {
  frame_number: number;
  timestamp: number;
  keypoints_3d: Keypoint3D[] | null;
  valid_keypoints: number;
}

interface Skeleton3DViewerProps {
  poses3D: Pose3DFrame[];
  skeletonConnections: [number, number][];
  keypointNames: string[];
  fps?: number;
  autoPlay?: boolean;
  showLabels?: boolean;
  showGrid?: boolean;
}

// 單個關鍵點組件
const Keypoint: React.FC<{
  position: [number, number, number];
  color: string;
  size?: number;
  name?: string;
  showLabel?: boolean;
}> = ({ position, color, size = 0.03, name, showLabel = false }) => {
  return (
    <group position={position}>
      <mesh>
        <sphereGeometry args={[size, 16, 16]} />
        <meshStandardMaterial color={color} />
      </mesh>
      {showLabel && name && (
        <Text
          position={[0, size + 0.05, 0]}
          fontSize={0.05}
          color="white"
          anchorX="center"
          anchorY="bottom"
        >
          {name}
        </Text>
      )}
    </group>
  );
};

// 骨骼連接線組件
const Bone: React.FC<{
  start: [number, number, number];
  end: [number, number, number];
  color: string;
  lineWidth?: number;
}> = ({ start, end, color, lineWidth = 3 }) => {
  return (
    <Line
      points={[start, end]}
      color={color}
      lineWidth={lineWidth}
    />
  );
};

// 獲取骨骼顏色
const getBoneColor = (startIdx: number, endIdx: number): string => {
  // 頭部
  if ([0, 1, 2, 3, 4].includes(startIdx) && [0, 1, 2, 3, 4].includes(endIdx)) {
    return BONE_COLORS.head;
  }
  // 軀幹
  if ([5, 6, 11, 12].includes(startIdx) && [5, 6, 11, 12].includes(endIdx)) {
    return BONE_COLORS.torso;
  }
  // 左臂
  if ([5, 7, 9].includes(startIdx) && [5, 7, 9].includes(endIdx)) {
    return BONE_COLORS.leftArm;
  }
  // 右臂
  if ([6, 8, 10].includes(startIdx) && [6, 8, 10].includes(endIdx)) {
    return BONE_COLORS.rightArm;
  }
  // 左腿
  if ([11, 13, 15].includes(startIdx) && [11, 13, 15].includes(endIdx)) {
    return BONE_COLORS.leftLeg;
  }
  // 右腿
  if ([12, 14, 16].includes(startIdx) && [12, 14, 16].includes(endIdx)) {
    return BONE_COLORS.rightLeg;
  }
  return '#FFFFFF';
};

// 骨架組件
const Skeleton: React.FC<{
  keypoints: Keypoint3D[];
  connections: [number, number][];
  showLabels: boolean;
}> = ({ keypoints, connections, showLabels }) => {
  // 渲染關鍵點
  const keypointElements = keypoints.map((kp, idx) => {
    if (!kp.position) return null;
    const color = KEYPOINT_COLORS[kp.name] || '#FFFFFF';
    return (
      <Keypoint
        key={`kp-${idx}`}
        position={kp.position}
        color={color}
        name={kp.name}
        showLabel={showLabels}
      />
    );
  });

  // 渲染骨骼連接
  const boneElements = connections.map(([startIdx, endIdx], idx) => {
    const startKp = keypoints[startIdx];
    const endKp = keypoints[endIdx];
    if (!startKp?.position || !endKp?.position) return null;
    return (
      <Bone
        key={`bone-${idx}`}
        start={startKp.position}
        end={endKp.position}
        color={getBoneColor(startIdx, endIdx)}
      />
    );
  });

  return (
    <group>
      {keypointElements}
      {boneElements}
    </group>
  );
};

// 動畫骨架組件
const AnimatedSkeleton: React.FC<{
  poses: Pose3DFrame[];
  connections: [number, number][];
  showLabels: boolean;
  playing: boolean;
  fps: number;
  currentFrame: number;
  onFrameChange: (frame: number) => void;
}> = ({ poses, connections, showLabels, playing, fps, currentFrame, onFrameChange }) => {
  const frameRef = useRef(0);
  const lastTimeRef = useRef(0);

  useFrame((state) => {
    if (!playing || poses.length === 0) return;

    const elapsed = state.clock.getElapsedTime();
    const frameDuration = 1 / fps;

    if (elapsed - lastTimeRef.current >= frameDuration) {
      lastTimeRef.current = elapsed;
      const nextFrame = (currentFrame + 1) % poses.length;
      onFrameChange(nextFrame);
    }
  });

  const currentPose = poses[currentFrame];
  if (!currentPose?.keypoints_3d) return null;

  return (
    <Skeleton
      keypoints={currentPose.keypoints_3d}
      connections={connections}
      showLabels={showLabels}
    />
  );
};

// 地板網格
const Floor: React.FC = () => {
  return (
    <mesh rotation={[-Math.PI / 2, 0, 0]} position={[0, -1, 0]}>
      <planeGeometry args={[10, 10]} />
      <meshStandardMaterial color="#1a1a2e" transparent opacity={0.5} />
    </mesh>
  );
};

// 坐標軸
const Axes: React.FC = () => {
  return (
    <group>
      {/* X 軸 - 紅色 */}
      <Line points={[[0, 0, 0], [1, 0, 0]]} color="red" lineWidth={2} />
      {/* Y 軸 - 綠色 */}
      <Line points={[[0, 0, 0], [0, 1, 0]]} color="green" lineWidth={2} />
      {/* Z 軸 - 藍色 */}
      <Line points={[[0, 0, 0], [0, 0, 1]]} color="blue" lineWidth={2} />
    </group>
  );
};

// 主組件
const Skeleton3DViewer: React.FC<Skeleton3DViewerProps> = ({
  poses3D,
  skeletonConnections,
  keypointNames,
  fps = 30,
  autoPlay = false,
  showLabels = false,
  showGrid = true,
}) => {
  const [playing, setPlaying] = useState(autoPlay);
  const [currentFrame, setCurrentFrame] = useState(0);
  const [showKeyLabels, setShowKeyLabels] = useState(showLabels);

  // 過濾有效的幀
  const validPoses = useMemo(() => {
    return poses3D.filter(p => p.keypoints_3d && p.valid_keypoints >= 3);
  }, [poses3D]);

  const totalFrames = validPoses.length;

  // 播放控制
  const handlePlayPause = () => setPlaying(!playing);
  const handleReset = () => {
    setCurrentFrame(0);
    setPlaying(false);
  };
  const handleSliderChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    setCurrentFrame(parseInt(e.target.value, 10));
  };

  if (totalFrames === 0) {
    return (
      <div className="flex items-center justify-center h-full bg-gray-900 text-white">
        <p>沒有有效的 3D 姿態數據</p>
      </div>
    );
  }

  return (
    <div className="flex flex-col h-full bg-gray-900">
      {/* 3D 視圖 */}
      <div className="flex-1 relative">
        <Canvas
          camera={{ position: [3, 2, 3], fov: 50 }}
          style={{ background: '#0f0f1a' }}
        >
          <ambientLight intensity={0.5} />
          <pointLight position={[10, 10, 10]} intensity={1} />
          <pointLight position={[-10, -10, -10]} intensity={0.5} />

          {showGrid && <gridHelper args={[10, 10, '#333', '#222']} />}
          <Axes />
          <Floor />

          <AnimatedSkeleton
            poses={validPoses}
            connections={skeletonConnections}
            showLabels={showKeyLabels}
            playing={playing}
            fps={fps}
            currentFrame={currentFrame}
            onFrameChange={setCurrentFrame}
          />

          <OrbitControls
            enableDamping
            dampingFactor={0.05}
            minDistance={1}
            maxDistance={10}
          />
        </Canvas>

        {/* 幀數顯示 */}
        <div className="absolute top-4 left-4 bg-black/50 px-3 py-1 rounded text-white text-sm">
          幀: {currentFrame + 1} / {totalFrames}
        </div>
      </div>

      {/* 控制面板 */}
      <div className="bg-gray-800 p-4 space-y-3">
        {/* 時間軸 */}
        <div className="flex items-center gap-4">
          <input
            type="range"
            min={0}
            max={totalFrames - 1}
            value={currentFrame}
            onChange={handleSliderChange}
            className="flex-1 h-2 bg-gray-700 rounded-lg appearance-none cursor-pointer"
          />
        </div>

        {/* 控制按鈕 */}
        <div className="flex items-center justify-between">
          <div className="flex gap-2">
            <button
              onClick={handlePlayPause}
              className="px-4 py-2 bg-green-600 hover:bg-green-700 text-white rounded-lg transition-colors"
            >
              {playing ? '⏸ 暫停' : '▶ 播放'}
            </button>
            <button
              onClick={handleReset}
              className="px-4 py-2 bg-gray-600 hover:bg-gray-700 text-white rounded-lg transition-colors"
            >
              ⏹ 重置
            </button>
          </div>

          <div className="flex items-center gap-4">
            <label className="flex items-center gap-2 text-white text-sm">
              <input
                type="checkbox"
                checked={showKeyLabels}
                onChange={(e) => setShowKeyLabels(e.target.checked)}
                className="w-4 h-4"
              />
              顯示標籤
            </label>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Skeleton3DViewer;
