/**
 * 相機角度選擇器組件
 * 提供圖形化界面讓用戶精確調整每個視角的實際拍攝角度
 */
import React, { useRef, useState, useEffect, useCallback } from 'react';

interface CameraAngleSelectorProps {
  viewName: string;
  label: string;
  defaultAngle: number;  // 預設角度 (0, 90, 180, 270)
  angle: number;         // 當前角度
  onChange: (angle: number) => void;
  size?: number;         // 組件大小
}

const CameraAngleSelector: React.FC<CameraAngleSelectorProps> = ({
  viewName,
  label,
  defaultAngle,
  angle,
  onChange,
  size = 120,
}) => {
  const svgRef = useRef<SVGSVGElement>(null);
  const [isDragging, setIsDragging] = useState(false);

  const center = size / 2;
  const radius = size / 2 - 15;
  const indicatorRadius = radius - 10;

  // 將角度轉換為弧度
  const toRadians = (deg: number) => (deg - 90) * (Math.PI / 180);

  // 計算指示器位置
  const indicatorX = center + indicatorRadius * Math.cos(toRadians(angle));
  const indicatorY = center + indicatorRadius * Math.sin(toRadians(angle));

  // 計算預設位置
  const defaultX = center + (radius + 8) * Math.cos(toRadians(defaultAngle));
  const defaultY = center + (radius + 8) * Math.sin(toRadians(defaultAngle));

  // 處理滑鼠/觸控事件
  const handlePointerEvent = useCallback((e: React.PointerEvent | PointerEvent) => {
    if (!svgRef.current) return;

    const rect = svgRef.current.getBoundingClientRect();
    const x = e.clientX - rect.left - center;
    const y = e.clientY - rect.top - center;

    // 計算角度 (0° 在上方)
    let newAngle = Math.atan2(y, x) * (180 / Math.PI) + 90;
    if (newAngle < 0) newAngle += 360;

    // 四捨五入到整數
    onChange(Math.round(newAngle) % 360);
  }, [center, onChange]);

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

  // 重置到預設角度
  const handleReset = () => {
    onChange(defaultAngle);
  };

  // 角度差異計算
  const angleDiff = angle - defaultAngle;
  const normalizedDiff = angleDiff > 180 ? angleDiff - 360 : (angleDiff < -180 ? angleDiff + 360 : angleDiff);

  return (
    <div className="flex flex-col items-center">
      <div className="text-sm font-medium text-gray-700 mb-1">{label}</div>

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
        {/* 背景圓 */}
        <circle
          cx={center}
          cy={center}
          r={radius}
          fill="#f3f4f6"
          stroke="#d1d5db"
          strokeWidth={2}
        />

        {/* 網格線 */}
        {[0, 45, 90, 135, 180, 225, 270, 315].map((deg) => {
          const x = center + radius * Math.cos(toRadians(deg));
          const y = center + radius * Math.sin(toRadians(deg));
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

        {/* 方位標記 */}
        <text x={center} y={8} textAnchor="middle" fontSize={10} fill="#6b7280">0°</text>
        <text x={size - 5} y={center + 4} textAnchor="end" fontSize={10} fill="#6b7280">90°</text>
        <text x={center} y={size - 3} textAnchor="middle" fontSize={10} fill="#6b7280">180°</text>
        <text x={5} y={center + 4} textAnchor="start" fontSize={10} fill="#6b7280">270°</text>

        {/* 人物位置 (中心) */}
        <circle cx={center} cy={center} r={8} fill="#3b82f6" />
        <text x={center} y={center + 3} textAnchor="middle" fontSize={8} fill="white">人</text>

        {/* 預設位置標記 */}
        <circle
          cx={defaultX}
          cy={defaultY}
          r={4}
          fill="none"
          stroke="#9ca3af"
          strokeWidth={2}
          strokeDasharray="2,2"
        />

        {/* 角度弧線 (顯示與預設的差異) */}
        {normalizedDiff !== 0 && (
          <path
            d={describeArc(center, center, indicatorRadius / 2, defaultAngle, angle)}
            fill="none"
            stroke={Math.abs(normalizedDiff) > 30 ? "#ef4444" : "#22c55e"}
            strokeWidth={2}
            opacity={0.5}
          />
        )}

        {/* 當前位置指示器 */}
        <line
          x1={center}
          y1={center}
          x2={indicatorX}
          y2={indicatorY}
          stroke="#2563eb"
          strokeWidth={2}
        />
        <circle
          cx={indicatorX}
          cy={indicatorY}
          r={8}
          fill="#2563eb"
          stroke="white"
          strokeWidth={2}
          className={isDragging ? 'opacity-80' : ''}
        />
        <text
          x={indicatorX}
          y={indicatorY + 3}
          textAnchor="middle"
          fontSize={7}
          fill="white"
          fontWeight="bold"
        >
          📷
        </text>
      </svg>

      {/* 角度顯示和控制 */}
      <div className="mt-2 flex items-center gap-2">
        <input
          type="number"
          value={angle}
          onChange={(e) => onChange(Math.max(0, Math.min(359, parseInt(e.target.value) || 0)))}
          className="w-16 px-2 py-1 text-center text-sm border border-gray-300 rounded focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
          min={0}
          max={359}
        />
        <span className="text-sm text-gray-500">°</span>
        <button
          onClick={handleReset}
          className="px-2 py-1 text-xs bg-gray-200 hover:bg-gray-300 rounded transition-colors"
          title="重置到預設角度"
        >
          重置
        </button>
      </div>

      {/* 差異提示 */}
      {normalizedDiff !== 0 && (
        <div className={`mt-1 text-xs ${Math.abs(normalizedDiff) > 30 ? 'text-red-500' : 'text-green-600'}`}>
          {normalizedDiff > 0 ? '+' : ''}{normalizedDiff}° 偏移
        </div>
      )}
    </div>
  );
};

// 輔助函數：繪製弧線路徑
function describeArc(x: number, y: number, radius: number, startAngle: number, endAngle: number): string {
  const start = polarToCartesian(x, y, radius, endAngle);
  const end = polarToCartesian(x, y, radius, startAngle);

  let diff = endAngle - startAngle;
  if (diff < 0) diff += 360;
  const largeArcFlag = diff <= 180 ? 0 : 1;

  return [
    'M', start.x, start.y,
    'A', radius, radius, 0, largeArcFlag, 0, end.x, end.y
  ].join(' ');
}

function polarToCartesian(cx: number, cy: number, radius: number, angleDeg: number) {
  const angleRad = (angleDeg - 90) * Math.PI / 180;
  return {
    x: cx + radius * Math.cos(angleRad),
    y: cy + radius * Math.sin(angleRad)
  };
}

export default CameraAngleSelector;
