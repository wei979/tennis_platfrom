import React from 'react';
import { Link } from 'react-router-dom';

const HomePage: React.FC = () => {
    return (
        // 外層容器：設置最小高度和背景色
        <div className="bg-gray-50 py-12 min-h-screen">
            
            {/* 置中容器：限制最大寬度並置中 */}
            <div className="max-w-6xl mx-auto px-4 sm:px-6 lg:px-8 text-center">
            
                <div className="mb-12">
                    <h1 className="text-5xl font-bold text-gray-800 mb-6">
                        🎾 Smart Tennis
                    </h1>
                    <p className="text-xl text-gray-600 mb-8">
                        使用 AI 技術分析您的網球技巧
                    </p>
                </div>

                {/* 功能介紹區塊 (卡片) */}
                <div className="grid grid-cols-1 md:grid-cols-3 gap-8 mb-12">
                    {/* 卡片 1: 網球追蹤 */}
                    <div className="bg-white p-6 rounded-xl shadow-lg flex flex-col items-center transition-all hover:shadow-xl border border-gray-100">
                        <div className="text-4xl mb-4 text-orange-500">🎯</div>
                        <h3 className="text-xl font-semibold mb-2 text-gray-800">網球追蹤</h3>
                        <p className="text-gray-600">
                            使用 YOLOv8 模型精確追蹤網球軌跡
                        </p>
                    </div>

                    {/* 卡片 2: 正反手檢測 */}
                    <div className="bg-white p-6 rounded-xl shadow-lg flex flex-col items-center transition-all hover:shadow-xl border border-gray-100">
                        <div className="text-4xl mb-4 text-blue-500">🏸</div>
                        <h3 className="text-xl font-semibold mb-2 text-gray-800">正反手檢測</h3>
                        <p className="text-gray-600">
                            自動識別球員的正手和反手擊球
                        </p>
                    </div>

                    {/* 卡片 3: 速度分析 */}
                    <div className="bg-white p-6 rounded-xl shadow-lg flex flex-col items-center transition-all hover:shadow-xl border border-gray-100">
                        <div className="text-4xl mb-4 text-green-500">⚡</div>
                        <h3 className="text-xl font-semibold mb-2 text-gray-800">速度分析</h3>
                        <p className="text-gray-600">
                            計算網球飛行速度和軌跡統計
                        </p>
                    </div>
                </div>

                {/* 主要行動呼籲區塊 (包含所有按鈕) */}
                <div className="bg-blue-50 p-8 rounded-xl shadow-inner border-t-2 border-blue-200">
                    <h2 className="text-3xl font-bold text-gray-800 mb-4">
                        開始分析您的網球影片
                    </h2>
                    <p className="text-gray-600 mb-6">
                        上傳您的網球比賽或練習影片，我們的 AI 將為您提供詳細的技術分析
                    </p>
                    
                    {/* 按鈕區域：新增了查看儀表板的按鈕 */}
                    <div className="flex flex-wrap justify-center space-x-4 space-y-4 md:space-y-0">
                        
                        {/* 1. 上傳單一影片 */}
                        <Link
                            to="/upload"
                            className="bg-blue-600 text-white px-8 py-3 rounded-lg text-lg font-semibold hover:bg-blue-700 transition-colors shadow-md"
                        >
                            立即上傳單一影片
                        </Link>
                        
                        {/* 2. 查看儀表板 (新增) */}
                        <Link
                            to="/dashboard"
                            className="bg-purple-600 text-white px-8 py-3 rounded-lg text-lg font-semibold hover:bg-purple-700 transition-colors shadow-md"
                        >
                             查看球員儀表板
                        </Link>

                        {/* 3. 批量上傳影片 */}
                        <Link
                            to="/multiupload"
                            className="bg-green-500 text-white px-8 py-3 rounded-lg text-lg font-semibold hover:bg-green-600 transition-colors shadow-md"
                        >
                            批量上傳影片 (記錄人名)
                        </Link>
                    </div>
                </div>

                {/* 底部功能特點清單 */}
                <div className="mt-12 text-left">
                    <h2 className="text-2xl font-bold text-gray-800 mb-6">功能特點</h2>
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                        <div className="flex items-start space-x-3">
                            <div className="text-green-600 text-xl">✓</div>
                            <div>
                                <h4 className="font-semibold">高精度追蹤</h4>
                                <p className="text-gray-600">使用最新的 YOLO 模型進行物件檢測</p>
                            </div>
                        </div>
                        <div className="flex items-start space-x-3">
                            <div className="text-green-600 text-xl">✓</div>
                            <div>
                                <h4 className="font-semibold">姿態分析</h4>
                                <p className="text-gray-600">使用 MediaPipe 進行人體姿態檢測</p>
                            </div>
                        </div>
                        <div className="flex items-start space-x-3">
                            <div className="text-green-600 text-xl">✓</div>
                            <div>
                                <h4 className="font-semibold">速度計算</h4>
                                <p className="text-gray-600">精確計算網球飛行速度和軌跡</p>
                            </div>
                        </div>
                        <div className="flex items-start space-x-3">
                            <div className="text-green-600 text-xl">✓</div>
                            <div>
                                <h4 className="font-semibold">統計分析</h4>
                                <p className="text-gray-600">提供詳細的比賽統計和可視化圖表</p>
                            </div>
                        </div>
                    </div>
                </div>
            </div> {/* 結束置中容器 */}
        </div>
    );
};

export default HomePage;