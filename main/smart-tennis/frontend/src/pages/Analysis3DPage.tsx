/**
 * 3D 姿態分析頁面
 * 支援多視角影片上傳和 3D 骨架可視化
 * 包含相機角度、距離、高度微調功能
 */
import React, { useState, useEffect } from 'react';
import { useDropzone } from 'react-dropzone';
import axios from 'axios';
import Skeleton3DViewer from '../components/Skeleton3DViewer';
import CameraConfigSelector from '../components/CameraConfigSelector';

const API_BASE = process.env.REACT_APP_API_URL || 'http://localhost:5000';

interface CameraConfig {
  angle: number;
  distance: number;
  height: number;
}

interface ViewUpload {
  viewName: string;
  label: string;
  defaultAngle: number;
  file: File | null;
  preview: string | null;
}

interface Analysis3DResult {
  batch_id: string;
  poses_3d: any[];
  skeleton_connections: [number, number][];
  keypoint_names: string[];
  video_info: {
    fps: number;
    total_frames: number;
  };
  statistics: {
    total_frames: number;
    valid_frames: number;
    reconstruction_rate: number;
  };
}

const Analysis3DPage: React.FC = () => {
  // 狀態
  const [views, setViews] = useState<ViewUpload[]>([
    { viewName: 'front', label: '正面', defaultAngle: 0, file: null, preview: null },
    { viewName: 'right', label: '右側', defaultAngle: 90, file: null, preview: null },
    { viewName: 'back', label: '背面', defaultAngle: 180, file: null, preview: null },
    { viewName: 'left', label: '左側', defaultAngle: 270, file: null, preview: null },
  ]);
  // 每個視角的相機配置 (角度、距離、高度)
  const [cameraConfigs, setCameraConfigs] = useState<{ [key: string]: CameraConfig }>({
    front: { angle: 0, distance: 5.0, height: 1.5 },
    right: { angle: 90, distance: 5.0, height: 1.5 },
    back: { angle: 180, distance: 5.0, height: 1.5 },
    left: { angle: 270, distance: 5.0, height: 1.5 },
  });
  const [showConfigTuning, setShowConfigTuning] = useState(false);
  const [personName, setPersonName] = useState('');
  const [uploading, setUploading] = useState(false);
  const [analyzing, setAnalyzing] = useState(false);
  const [progress, setProgress] = useState(0);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<Analysis3DResult | null>(null);
  const [is3DAvailable, setIs3DAvailable] = useState<boolean | null>(null);

  // 檢查 3D 功能是否可用
  useEffect(() => {
    const check3DStatus = async () => {
      try {
        const response = await axios.get(`${API_BASE}/api/3d-status`);
        setIs3DAvailable(response.data.available);
      } catch (err) {
        setIs3DAvailable(false);
      }
    };
    check3DStatus();
  }, []);

  // 處理檔案上傳
  const handleFileDrop = (viewName: string) => (acceptedFiles: File[]) => {
    const file = acceptedFiles[0];
    if (!file) return;

    const preview = URL.createObjectURL(file);
    setViews(prev => prev.map(v =>
      v.viewName === viewName
        ? { ...v, file, preview }
        : v
    ));
  };

  // 移除檔案
  const handleRemoveFile = (viewName: string) => {
    setViews(prev => prev.map(v =>
      v.viewName === viewName
        ? { ...v, file: null, preview: null }
        : v
    ));
  };

  // 上傳並分析
  const handleUploadAndAnalyze = async () => {
    const filesToUpload = views.filter(v => v.file);

    if (filesToUpload.length < 2) {
      setError('請至少上傳 2 個視角的影片');
      return;
    }

    setError(null);
    setUploading(true);
    setProgress(10);

    try {
      // 1. 上傳多視角影片
      const formData = new FormData();
      filesToUpload.forEach(v => {
        if (v.file) {
          formData.append(`video_${v.viewName}`, v.file);
        }
      });
      formData.append('person_name', personName || '未命名球員');
      // 傳送每個視角的相機配置 (角度、距離、高度)
      formData.append('camera_configs', JSON.stringify(cameraConfigs));

      const uploadResponse = await axios.post(
        `${API_BASE}/api/upload-multiview`,
        formData,
        {
          headers: { 'Content-Type': 'multipart/form-data' },
          onUploadProgress: (progressEvent) => {
            const percent = Math.round(
              (progressEvent.loaded * 30) / (progressEvent.total || 1)
            );
            setProgress(10 + percent);
          },
        }
      );

      if (!uploadResponse.data.success) {
        throw new Error(uploadResponse.data.error);
      }

      const batchId = uploadResponse.data.batch_id;
      setProgress(40);
      setUploading(false);
      setAnalyzing(true);

      // 2. 執行 3D 分析
      const analyzeResponse = await axios.post(
        `${API_BASE}/api/analyze-3d/${batchId}`
      );

      if (!analyzeResponse.data.success) {
        throw new Error(analyzeResponse.data.error);
      }

      setProgress(80);

      // 3. 獲取結果
      const resultResponse = await axios.get(
        `${API_BASE}/api/results-3d/${batchId}`
      );

      setResult(resultResponse.data);
      setProgress(100);

    } catch (err: any) {
      setError(err.message || '分析過程中發生錯誤');
    } finally {
      setUploading(false);
      setAnalyzing(false);
    }
  };

  // 重置
  const handleReset = () => {
    setViews(views.map(v => ({ ...v, file: null, preview: null })));
    setResult(null);
    setError(null);
    setProgress(0);
  };

  // 單個視角上傳區域
  const ViewUploadZone: React.FC<{ view: ViewUpload }> = ({ view }) => {
    const { getRootProps, getInputProps, isDragActive } = useDropzone({
      onDrop: handleFileDrop(view.viewName),
      accept: { 'video/*': ['.mp4', '.avi', '.mov', '.mkv'] },
      maxSize: 100 * 1024 * 1024,
      multiple: false,
    });

    if (view.file) {
      return (
        <div className="relative bg-gray-800 rounded-lg overflow-hidden aspect-video">
          <video
            src={view.preview || undefined}
            className="w-full h-full object-cover"
            muted
            playsInline
          />
          <div className="absolute inset-0 bg-black/50 flex items-center justify-center opacity-0 hover:opacity-100 transition-opacity">
            <button
              onClick={() => handleRemoveFile(view.viewName)}
              className="px-4 py-2 bg-red-600 text-white rounded-lg hover:bg-red-700"
            >
              移除
            </button>
          </div>
          <div className="absolute bottom-2 left-2 bg-green-500 text-white px-2 py-1 rounded text-sm">
            {view.label} ✓
          </div>
        </div>
      );
    }

    return (
      <div
        {...getRootProps()}
        className={`border-2 border-dashed rounded-lg aspect-video flex flex-col items-center justify-center cursor-pointer transition-colors ${
          isDragActive
            ? 'border-blue-500 bg-blue-50'
            : 'border-gray-300 hover:border-blue-400 bg-gray-50'
        }`}
      >
        <input {...getInputProps()} />
        <div className="text-4xl mb-2">📹</div>
        <p className="text-sm font-medium text-gray-700">{view.label}</p>
        <p className="text-xs text-gray-500">拖放或點擊上傳</p>
      </div>
    );
  };

  // 3D 功能不可用
  if (is3DAvailable === false) {
    return (
      <div className="max-w-4xl mx-auto p-8">
        <div className="bg-yellow-50 border border-yellow-200 rounded-lg p-6 text-center">
          <div className="text-4xl mb-4">⚠️</div>
          <h2 className="text-xl font-bold text-yellow-800 mb-2">
            3D 重建功能不可用
          </h2>
          <p className="text-yellow-700">
            後端 3D 重建模組未正確載入，請檢查服務器配置。
          </p>
        </div>
      </div>
    );
  }

  // 複製批次 ID 到剪貼簿
  const copyBatchId = () => {
    if (result?.batch_id) {
      navigator.clipboard.writeText(result.batch_id);
      alert('批次 ID 已複製到剪貼簿！');
    }
  };

  // 顯示結果
  if (result) {
    return (
      <div className="h-screen flex flex-col">
        {/* 標題欄 */}
        <div className="bg-white shadow px-6 py-4">
          <div className="flex items-center justify-between">
            <div>
              <h1 className="text-2xl font-bold text-gray-800">3D 姿態分析結果</h1>
              <p className="text-sm text-gray-500">
                重建成功率: {(result.statistics.reconstruction_rate * 100).toFixed(1)}%
                ({result.statistics.valid_frames}/{result.statistics.total_frames} 幀)
              </p>
            </div>
            <button
              onClick={handleReset}
              className="px-4 py-2 bg-gray-600 text-white rounded-lg hover:bg-gray-700"
            >
              重新分析
            </button>
          </div>
          {/* 批次 ID 顯示與複製 */}
          <div className="mt-3 flex items-center gap-2 bg-blue-50 px-4 py-2 rounded-lg">
            <span className="text-sm text-blue-700 font-medium">批次 ID:</span>
            <code className="text-sm bg-white px-2 py-1 rounded border flex-1 truncate">
              {result.batch_id}
            </code>
            <button
              onClick={copyBatchId}
              className="px-3 py-1 bg-blue-600 text-white text-sm rounded hover:bg-blue-700"
            >
              複製
            </button>
            <a
              href="/poseaug"
              className="px-3 py-1 bg-green-600 text-white text-sm rounded hover:bg-green-700"
            >
              用於 PoseAug
            </a>
          </div>
        </div>

        {/* 3D 視圖 */}
        <div className="flex-1">
          <Skeleton3DViewer
            poses3D={result.poses_3d}
            skeletonConnections={result.skeleton_connections}
            keypointNames={result.keypoint_names}
            fps={result.video_info.fps}
            autoPlay={false}
            showLabels={false}
            showGrid={true}
          />
        </div>
      </div>
    );
  }

  // 上傳界面
  return (
    <div className="max-w-6xl mx-auto p-8">
      <div className="text-center mb-8">
        <h1 className="text-4xl font-bold text-gray-800 mb-4">
          3D 姿態分析
        </h1>
        <p className="text-lg text-gray-600">
          上傳多角度網球影片，重建 3D 人體骨架
        </p>
      </div>

      {error && (
        <div className="mb-6 bg-red-50 border border-red-200 text-red-700 px-4 py-3 rounded-lg">
          {error}
        </div>
      )}

      <div className="bg-white rounded-lg shadow-lg p-8">
        {/* 基本資訊 */}
        <div className="mb-8">
          <h2 className="text-xl font-semibold mb-4">基本資訊</h2>
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">
              球員姓名
            </label>
            <input
              type="text"
              value={personName}
              onChange={(e) => setPersonName(e.target.value)}
              placeholder="未命名球員"
              className="w-full max-w-md px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
            />
          </div>
        </div>

        {/* 視角上傳區域 */}
        <div className="mb-8">
          <h2 className="text-xl font-semibold mb-4">
            影片上傳 (至少 2 個視角)
          </h2>
          <div className="grid grid-cols-2 gap-4">
            {views.map((view) => (
              <ViewUploadZone key={view.viewName} view={view} />
            ))}
          </div>
          <p className="mt-2 text-sm text-gray-500">
            建議上傳所有 4 個視角以獲得最佳 3D 重建效果
          </p>
        </div>

        {/* 相機配置微調 */}
        <div className="mb-8">
          <div className="flex items-center justify-between mb-4">
            <div>
              <h2 className="text-xl font-semibold">相機配置微調</h2>
              <p className="text-sm text-gray-500">調整每個相機的角度、距離和高度以提高 3D 重建精確度</p>
            </div>
            <button
              onClick={() => setShowConfigTuning(!showConfigTuning)}
              className="px-4 py-2 text-sm bg-blue-100 hover:bg-blue-200 text-blue-700 rounded-lg transition-colors flex items-center gap-2"
            >
              {showConfigTuning ? '收起設定' : '精確校準'}
              <span className={`transform transition-transform ${showConfigTuning ? 'rotate-180' : ''}`}>▼</span>
            </button>
          </div>

          {showConfigTuning && (
            <div className="bg-gray-50 rounded-lg p-6">
              <div className="bg-blue-50 border border-blue-200 rounded-lg p-4 mb-6">
                <h3 className="font-medium text-blue-800 mb-2">使用說明</h3>
                <ul className="text-sm text-blue-700 space-y-1">
                  <li>• <strong>角度</strong>：拖動相機圖標或輸入數值調整相機圍繞球員的角度 (0°=正前方)</li>
                  <li>• <strong>距離</strong>：調整相機到球員的距離 (1-20公尺)</li>
                  <li>• <strong>高度</strong>：調整相機離地的高度 (0.5-5公尺)</li>
                </ul>
              </div>

              {/* 四個相機配置選擇器 */}
              <div className="grid grid-cols-2 lg:grid-cols-4 gap-4 mb-6">
                {views.map((view) => (
                  <CameraConfigSelector
                    key={view.viewName}
                    viewName={view.viewName}
                    label={view.label}
                    defaultAngle={view.defaultAngle}
                    config={cameraConfigs[view.viewName]}
                    onChange={(newConfig) => setCameraConfigs(prev => ({
                      ...prev,
                      [view.viewName]: newConfig
                    }))}
                  />
                ))}
              </div>

              {/* 俯視圖和側視圖預覽 */}
              <div className="flex flex-wrap justify-center gap-8">
                {/* 俯視圖 */}
                <div className="text-center">
                  <div className="relative w-64 h-64 bg-white rounded-lg border border-gray-200 shadow-inner">
                    <svg width="256" height="256" className="absolute inset-0">
                      {/* 距離圓環 */}
                      {[5, 10, 15, 20].map((dist) => {
                        const r = 10 + (dist / 20) * 100;
                        return (
                          <circle
                            key={dist}
                            cx="128" cy="128" r={r}
                            fill="none" stroke="#e5e7eb" strokeWidth={1}
                            strokeDasharray="4,4"
                          />
                        );
                      })}

                      {/* 中心人物 */}
                      <circle cx="128" cy="128" r="8" fill="#3b82f6" />
                      <text x="128" y="131" textAnchor="middle" fontSize="8" fill="white">人</text>

                      {/* 相機位置 */}
                      {views.map((view) => {
                        const config = cameraConfigs[view.viewName];
                        const rad = (config.angle - 90) * Math.PI / 180;
                        const r = 10 + (config.distance / 20) * 100;
                        const x = 128 + r * Math.cos(rad);
                        const y = 128 + r * Math.sin(rad);
                        const hasFile = view.file !== null;

                        return (
                          <g key={view.viewName}>
                            <line
                              x1="128" y1="128" x2={x} y2={y}
                              stroke={hasFile ? "#22c55e" : "#d1d5db"}
                              strokeWidth={hasFile ? 2 : 1}
                              strokeDasharray={hasFile ? "none" : "4,4"}
                            />
                            <circle
                              cx={x} cy={y} r="12"
                              fill={hasFile ? "#22c55e" : "#9ca3af"}
                            />
                            <text x={x} y={y + 3} textAnchor="middle" fontSize="8" fill="white">
                              {view.label.charAt(0)}
                            </text>
                          </g>
                        );
                      })}
                    </svg>
                  </div>
                  <div className="text-xs text-gray-500 mt-2">俯視圖 (角度 + 距離)</div>
                </div>

                {/* 側視圖 */}
                <div className="text-center">
                  <div className="relative w-64 h-64 bg-white rounded-lg border border-gray-200 shadow-inner">
                    <svg width="256" height="256" className="absolute inset-0">
                      {/* 地面線 */}
                      <line x1="20" y1="220" x2="236" y2="220" stroke="#9ca3af" strokeWidth={2} />
                      <text x="128" y="240" textAnchor="middle" fontSize="10" fill="#6b7280">地面</text>

                      {/* 高度刻度 */}
                      {[1, 2, 3, 4, 5].map((h) => {
                        const py = 220 - h * 36;
                        return (
                          <g key={h}>
                            <line x1="20" y1={py} x2="236" y2={py} stroke="#e5e7eb" strokeWidth={1} strokeDasharray="4,4" />
                            <text x="16" y={py + 4} textAnchor="end" fontSize="8" fill="#9ca3af">{h}m</text>
                          </g>
                        );
                      })}

                      {/* 人物 */}
                      <rect x="118" y="120" width="20" height="100" fill="#3b82f6" rx="4" />
                      <circle cx="128" cy="110" r="12" fill="#3b82f6" />
                      <text x="128" y="172" textAnchor="middle" fontSize="10" fill="white">人</text>

                      {/* 相機高度指示 */}
                      {views.map((view, idx) => {
                        const config = cameraConfigs[view.viewName];
                        const px = 40 + idx * 50;
                        const py = 220 - config.height * 36;
                        const hasFile = view.file !== null;

                        return (
                          <g key={view.viewName}>
                            <line x1={px} y1="220" x2={px} y2={py} stroke={hasFile ? "#22c55e" : "#d1d5db"} strokeWidth={2} />
                            <circle cx={px} cy={py} r="10" fill={hasFile ? "#22c55e" : "#9ca3af"} />
                            <text x={px} y={py + 3} textAnchor="middle" fontSize="8" fill="white">
                              {view.label.charAt(0)}
                            </text>
                            <text x={px} y={py - 14} textAnchor="middle" fontSize="8" fill="#6b7280">
                              {config.height}m
                            </text>
                          </g>
                        );
                      })}
                    </svg>
                  </div>
                  <div className="text-xs text-gray-500 mt-2">側視圖 (高度)</div>
                </div>
              </div>
            </div>
          )}
        </div>

        {/* 進度條 */}
        {(uploading || analyzing) && (
          <div className="mb-6">
            <div className="flex items-center justify-between mb-2">
              <span className="text-sm font-medium text-gray-700">
                {uploading ? '上傳中...' : '3D 重建中...'}
              </span>
              <span className="text-sm text-gray-500">{progress}%</span>
            </div>
            <div className="w-full bg-gray-200 rounded-full h-2.5">
              <div
                className="bg-blue-600 h-2.5 rounded-full transition-all duration-300"
                style={{ width: `${progress}%` }}
              />
            </div>
          </div>
        )}

        {/* 提交按鈕 */}
        <button
          onClick={handleUploadAndAnalyze}
          disabled={uploading || analyzing || views.filter(v => v.file).length < 2}
          className={`w-full py-3 rounded-lg text-white font-medium transition-colors ${
            uploading || analyzing || views.filter(v => v.file).length < 2
              ? 'bg-gray-400 cursor-not-allowed'
              : 'bg-blue-600 hover:bg-blue-700'
          }`}
        >
          {uploading ? '上傳中...' : analyzing ? '分析中...' : '開始 3D 分析'}
        </button>
      </div>
    </div>
  );
};

export default Analysis3DPage;
