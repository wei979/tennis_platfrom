import React, { useState, useEffect } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { analyzeVideo } from '../services/api';

const AnalysisPage: React.FC = () => {
  const { fileId } = useParams<{ fileId: string }>();
  const navigate = useNavigate();
  const [analyzing, setAnalyzing] = useState(false);
  const [progress, setProgress] = useState(0);
  const [currentStep, setCurrentStep] = useState('準備中...');
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (fileId) {
      startAnalysis(fileId);
    }
  }, [fileId]);

  const startAnalysis = async (id: string) => {
    setAnalyzing(true);
    setError(null);

    // 模擬分析步驟
    const steps = [
      '初始化模型...',
      '載入影片...',
      '檢測網球...',
      '追蹤軌跡...',
      '分析姿態...',
      '檢測擊球...',
      '計算速度...',
      '生成結果...'
    ];

    try {
      for (let i = 0; i < steps.length; i++) {
        setCurrentStep(steps[i]);
        setProgress((i / steps.length) * 90); // 90% 為分析進度
        await new Promise(resolve => setTimeout(resolve, 1000)); // 模擬處理時間
      }

      // 執行實際分析
      setCurrentStep('正在處理影片...');
      const result = await analyzeVideo(id);

      setProgress(100);
      setCurrentStep('分析完成！');

      // 等待一下然後跳轉到結果頁面
      setTimeout(() => {
        navigate(`/results/${id}`);
      }, 1500);

    } catch (err: any) {
      setError(err.message || '分析過程中發生錯誤');
      setAnalyzing(false);
    }
  };

  if (!fileId) {
    return (
      <div className="text-center">
        <div className="text-red-600 text-xl">❌</div>
        <h2 className="text-xl font-semibold mt-2">錯誤</h2>
        <p className="text-gray-600">無效的檔案ID</p>
      </div>
    );
  }

  return (
    <div className="max-w-4xl mx-auto">
      <div className="text-center mb-8">
        <h1 className="text-4xl font-bold text-gray-800 mb-4">
          🔍 正在分析您的網球影片
        </h1>
        <p className="text-lg text-gray-600">
          AI 正在分析您的影片，請稍候...
        </p>
      </div>

      <div className="bg-white rounded-lg shadow-lg p-8">
        {analyzing && !error ? (
          <div className="text-center">
            {/* 進度環 */}
            <div className="relative inline-flex items-center justify-center mb-6">
              <svg className="w-32 h-32 transform -rotate-90">
                <circle
                  cx="64"
                  cy="64"
                  r="56"
                  stroke="currentColor"
                  strokeWidth="8"
                  fill="none"
                  className="text-gray-200"
                />
                <circle
                  cx="64"
                  cy="64"
                  r="56"
                  stroke="currentColor"
                  strokeWidth="8"
                  fill="none"
                  strokeDasharray={`${2 * Math.PI * 56}`}
                  strokeDashoffset={`${2 * Math.PI * 56 * (1 - progress / 100)}`}
                  className="text-blue-600 transition-all duration-300"
                />
              </svg>
              <div className="absolute inset-0 flex items-center justify-center">
                <span className="text-2xl font-bold text-blue-600">
                  {Math.round(progress)}%
                </span>
              </div>
            </div>

            <h3 className="text-xl font-semibold mb-2">{currentStep}</h3>
            <p className="text-gray-600 mb-6">
              這個過程可能需要幾分鐘時間，取決於影片長度和複雜度
            </p>

            {/* 分析步驟指示器 */}
            <div className="space-y-3">
              <div className="flex items-center justify-between bg-gray-50 p-3 rounded">
                <span>🎯 網球檢測</span>
                <span className={progress > 20 ? 'text-green-600' : 'text-gray-400'}>
                  {progress > 20 ? '✓' : '⏳'}
                </span>
              </div>
              <div className="flex items-center justify-between bg-gray-50 p-3 rounded">
                <span>📍 軌跡追蹤</span>
                <span className={progress > 40 ? 'text-green-600' : 'text-gray-400'}>
                  {progress > 40 ? '✓' : '⏳'}
                </span>
              </div>
              <div className="flex items-center justify-between bg-gray-50 p-3 rounded">
                <span>🏸 正反手檢測</span>
                <span className={progress > 60 ? 'text-green-600' : 'text-gray-400'}>
                  {progress > 60 ? '✓' : '⏳'}
                </span>
              </div>
              <div className="flex items-center justify-between bg-gray-50 p-3 rounded">
                <span>⚡ 速度分析</span>
                <span className={progress > 80 ? 'text-green-600' : 'text-gray-400'}>
                  {progress > 80 ? '✓' : '⏳'}
                </span>
              </div>
            </div>
          </div>
        ) : error ? (
          <div className="text-center">
            <div className="text-red-600 text-6xl mb-4">❌</div>
            <h3 className="text-xl font-semibold mb-4 text-red-800">分析失敗</h3>
            <p className="text-red-600 mb-6">{error}</p>
            <div className="space-x-4">
              <button
                onClick={() => window.location.reload()}
                className="bg-blue-600 text-white px-6 py-2 rounded-lg hover:bg-blue-700"
              >
                重試
              </button>
              <button
                onClick={() => navigate('/upload')}
                className="bg-gray-600 text-white px-6 py-2 rounded-lg hover:bg-gray-700"
              >
                重新上傳
              </button>
            </div>
          </div>
        ) : (
          <div className="text-center">
            <div className="text-green-600 text-6xl mb-4">✅</div>
            <h3 className="text-xl font-semibold mb-4">分析完成！</h3>
            <p className="text-gray-600">正在跳轉到結果頁面...</p>
          </div>
        )}
      </div>

      <div className="mt-8 bg-blue-50 rounded-lg p-6">
        <h2 className="text-xl font-semibold mb-4">💡 分析過程說明</h2>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-sm">
          <div>
            <h4 className="font-medium mb-2">🎯 網球檢測</h4>
            <p className="text-gray-600">使用 YOLOv8 模型識別影片中的網球</p>
          </div>
          <div>
            <h4 className="font-medium mb-2">📍 軌跡追蹤</h4>
            <p className="text-gray-600">追蹤網球在各幀中的位置變化</p>
          </div>
          <div>
            <h4 className="font-medium mb-2">🏸 正反手檢測</h4>
            <p className="text-gray-600">使用姿態檢測技術分析擊球動作</p>
          </div>
          <div>
            <h4 className="font-medium mb-2">⚡ 速度分析</h4>
            <p className="text-gray-600">計算網球的飛行速度和軌跡統計</p>
          </div>
        </div>
      </div>
    </div>
  );
};

export default AnalysisPage;
