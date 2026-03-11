// frontend/src/App.tsx

import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import Navbar from './components/Navbar'; // 假設 Navbar 在 src/components/
import HomePage from './pages/HomePage';
import UploadPage from './pages/UploadPage';
import AnalysisPage from './pages/AnalysisPage';
import ResultsPage from './pages/ResultsPage';
import MultiUploadPage from './pages/MultiUploadPage';
import MultiResultsPage from './pages/MultiResultsPage'; 
import PlayerDashboardPage from './pages/PlayerDashboardPage'; // 修正後的匯入
import Analysis3DPage from './pages/Analysis3DPage'; // 3D 姿態分析頁面
import PoseAugPage from './pages/PoseAugPage'; // PoseAug 姿態增強測試頁面
import './App.css';

function App() {
  return (
    <Router future={{ v7_startTransition: true, v7_relativeSplatPath: true }}>
      <div className="App min-h-screen bg-gray-50">
        <Navbar />
        <main className="container mx-auto px-4 py-8">
          <Routes>
            <Route path="/" element={<HomePage />} />
            <Route path="/upload" element={<UploadPage />} />
            <Route path="/multiupload" element={<MultiUploadPage />} />
            
            {/* 🌟 新增：球員數據儀表板路由 🌟 */}
            <Route path="/dashboard" element={<PlayerDashboardPage />} />

            {/* 🎯 3D 姿態分析路由 */}
            <Route path="/analysis-3d" element={<Analysis3DPage />} />

            {/* 🧪 PoseAug 姿態增強測試路由 */}
            <Route path="/poseaug" element={<PoseAugPage />} /> 

            {/* 舊的單檔案分析/結果路由 */}
            <Route path="/analysis/:fileId" element={<AnalysisPage />} />
            <Route path="/results/:fileId" element={<ResultsPage />} />
            
            {/* 多檔案結果頁面 */}
            <Route path="/multi-results/:batchId" element={<MultiResultsPage />} />
          </Routes>
        </main>
      </div>
    </Router>
  );
}

export default App;