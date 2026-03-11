/**
 * 相機配置選擇器組件
 * 提供圖形化界面讓用戶精確調整每個視角的角度、距離和高度
 */
import React, { useRef, useState, useCallback } from 'react';

interface CameraConfig {
  angle: number;
  distance: number;
  height: number;
}

interface CameraConfigSelectorProps {
  viewName: string;
  label: string;
  defaultAngle: number;
  config: CameraConfig;
  onChange: (config: CameraConfig) => void;
  size?: number;
}

const CameraConfigSelector: React.FC<CameraConfigSelectorProps> = ({
  viewName,
  label,
  defaultAngle,
  config,
  onChange,
  size = 100,
}) => {
  const svgRef = useRef<SVGSVGElement>(null);
  const [isDragging, setIsDragging] = useState(false);

  const center = size / 2;
  const maxRadius = size / 2 - 12;

  // 距離映射到半徑 (1-20m -> 20%-100% of maxRadius)
  const distanceToRadius = (dist: number) => {
    const normalized = (dist - 1) / 19; // 0-1
    return maxRadius * (0.2 + normalized * 0.8);
  };

  const radiusToDistance = (radius: number) => {
    const normalized = (radius / maxRadius - 0.2) / 0.8;
    return Math.max(1, Math.min(20, 1 + normalized * 19));
  };

  // 將角度轉換為弧度
  const toRadians = (deg: number) => (deg - 90) * (Math.PI / 180);

  // 計算指示器位置
  const currentRadius = distanceToRadius(config.distance);
  const indicatorX = center + currentRadius * Math.cos(toRadians(config.angle));
  const indicatorY = center + currentRadius * Math.sin(toRadians(config.angle));

  // 處理滑鼠/觸控事件
  const handlePointerEvent = useCallback((e: React.PointerEvent | PointerEvent) => {
    if (!svgRef.current) return;

    const rect = svgRef.current.getBoundingClientRect();
    const x = e.clientX - rect.left - center;
    const y = e.clientY - rect.top - center;

    // 計算角度
    let newAngle = Math.atan2(y, x) * (180 / Math.PI) + 90;
    if (newAngle < 0) newAngle += 360;

    // 計算距離
    const radius = Math.sqrt(x * x + y * y);
    const newDistance = radiusToDistance(radius);

    onChange({
      ...config,
      angle: Math.round(newAngle) % 360,
      distance: Math.round(newDistance * 2) / 2, // 0.5m 精度
    });
  }, [center, config, onChange]);

  const handlePointerDown = (e: React.PointerEvent) => {
    setIsDragging(true);
    handlePointerEvent(e);
    (e.target as HTMLElement).setPointerCapture(e.pointerId);
  };

  const handlePointerMove = (e: React.PointerEvent) => {
    if (isDragging) {
      handlePointerEvent(e);
    }
  };

  const handlePointerUp = () => {
    setIsDragging(false);
  };

  // 重置到預設
  const handleReset = () => {
    onChange({
      angle: defaultAngle,
      distance: 5.0,
      height: 1.5,
    });
  };

  // 角度差異
  const angleDiff = config.angle - defaultAngle;
  const normalizedDiff = angleDiff > 180 ? angleDiff - 360 : (angleDiff < -180 ? angleDiff + 360 : angleDiff);

  return (
    <div className="flex flex-col items-center bg-white rounded-lg p-3 shadow-sm border border-gray-200">
      <div className="text-sm font-semibold text-gray-700 mb-2">{label}</div>

      {/* 角度/距離選擇器 */}
      <svg
        ref={svgRef}
        width={size}
        height={size}
        className="cursor-crosshair select-none"
        onPointerDown={handlePointerDown}
        onPointerMove={handlePointerMove}
        onPointerUp={handlePointerUp}
        onPointerLeave={handlePointerUp}
      >
        {/* 距離圓環 */}
        {[5, 10, 15, 20].map((dist) => (
          <circle
            key={dist}
            cx={center}
            cy={center}
            r={distanceToRadius(dist)}
            fill="none"
            stroke="#e5e7eb"
            strokeWidth={1}
            strokeDasharray="2,2"
          />
        ))}

        {/* 角度線 */}
        {[0, 90, 180, 270].map((deg) => {
          const x = center + maxRadius * Math.cos(toRadians(deg));
          const y = center + maxRadius * Math.sin(toRadians(deg));
          return (
            <line
              key={deg}
              x1={center}
              y1={center}
              x2={x}
              y2={y}
              stroke="#e5e7eb"
              strokeWidth={1}
            />
          );
        })}

        {/* 中心人物 */}
        <circle cx={center} cy={center} r={6} fill="#3b82f6" />

        {/* 預設位置標記 */}
        <circle
          cx={center + distanceToRadius(5) * Math.cos(toRadians(defaultAngle))}
          cy={center + distanceToRadius(5) * Math.sin(toRadians(defaultAngle))}
          r={3}
          fill="none"
          stroke="#9ca3af"
          strokeWidth={1.5}
          strokeDasharray="2,2"
        />

        {/* 當前位置指示器 */}
        <line
          x1={center}
          y1={center}
          x2={indicatorX}
          y2={indicatorY}
          stroke="#2563eb"
          strokeWidth={1.5}
        />
        <circle
          cx={indicatorX}
          cy={indicatorY}
          r={7}
          fill="#2563eb"
          stroke="white"
          strokeWidth={1.5}
          className={isDragging ? 'opacity-80' : ''}
        />
        <text
          x={indicatorX}
          y={indicatorY + 3}
          textAnchor="middle"
          fontSize={8}
          fill="white"
        >
          📷
        </text>
      </svg>

      {/* 數值控制 */}
      <div className="w-full mt-2 space-y-2">
        {/* 角度 */}
        <div className="flex items-center justify-between text-xs">
          <span className="text-gray-500 w-12">角度</span>
          <input
            type="number"
            value={config.angle}
            onChange={(e) => onChange({ ...config, angle: Math.max(0, Math.min(359, parseInt(e.target.value) || 0)) })}
            className="w-14 px-1 py-0.5 text-center text-xs border border-gray-300 rounded"
            min={0}
            max={359}
          />
          <span className="text-gray-400 w-4">°</span>
          {normalizedDiff !== 0 && (
            <span className={`text-xs ${Math.abs(normalizedDiff) > 30 ? 'text-red-500' : 'text-green-600'}`}>
              {normalizedDiff > 0 ? '+' : ''}{normalizedDiff}°
            </span>
          )}
        </div>

        {/* 距離 */}
        <div className="flex items-center justify-between text-xs">
          <span className="text-gray-500 w-12">距離</span>
          <input
            type="range"
            value={config.distance}
            onChange={(e) => onChange({ ...config, distance: parseFloat(e.target.value) })}
            className="w-16 h-1"
            min={1}
            max={20}
            step={0.5}
          />
          <input
            type="number"
            value={config.distance}
            onChange={(e) => onChange({ ...config, distance: Math.max(1, Math.min(20, parseFloat(e.target.value) || 5)) })}
            className="w-12 px-1 py-0.5 text-center text-xs border border-gray-300 rounded"
            min={1}
            max={20}
            step={0.5}
          />
          <span className="text-gray-400 w-4">m</span>
        </div>

        {/* 高度 */}
        <div className="flex items-center justify-between text-xs">
          <span className="text-gray-500 w-12">高度</span>
          <input
            type="range"
            value={config.height}
            onChange={(e) => onChange({ ...config, height: parseFloat(e.target.value) })}
            className="w-16 h-1"
            min={0.5}
            max={5}
            step={0.1}
          />
          <input
            type="number"
            value={config.height}
            onChange={(e) => onChange({ ...config, height: Math.max(0.5, Math.min(5, parseFloat(e.target.value) || 1.5)) })}
            className="w-12 px-1 py-0.5 text-center text-xs border border-gray-300 rounded"
            min={0.5}
            max={5}
            step={0.1}
          />
          <span className="text-gray-400 w-4">m</span>
        </div>
      </div>

      {/* 重置按鈕 */}
      <button
        onClick={handleReset}
        className="mt-2 px-3 py-1 text-xs bg-gray-100 hover:bg-gray-200 rounded transition-colors"
        title="重置到預設值"
      >
        重置
      </button>
    </div>
  );
};

export default CameraConfigSelector;
