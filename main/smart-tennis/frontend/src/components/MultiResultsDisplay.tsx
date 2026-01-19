// frontend/src/components/MultiResultsDisplay.tsx
import React from 'react';

// 從 PlayerDashboardPage.tsx 複製過來的介面
interface AnalysisData {
    summary: {
        total_shots?: number;
        avg_speed?: number;
        initial_speed?: number;
        stabilityScore?: number;
        [key: string]: any;
    };
    shots: any;
    speed: any;
}

interface FetchedResult {
    fileId: string;
    name: string; 
    status: '完成' | '處理中' | '失敗';
    data?: AnalysisData; 
    error?: string;
    fileName: string;
    uploadTime: string; 
}

interface MultiResultsDisplayProps {
    fetchedResults: FetchedResult[];
    isBatchAnalysis: boolean; 
    playerSummary: { person_name: string };
}

const MultiResultsDisplay: React.FC<MultiResultsDisplayProps> = ({ fetchedResults, isBatchAnalysis, playerSummary }) => {
    
    // 計算所有影片的總體平均數據 (供總結區塊使用)
    const totalShots = fetchedResults.reduce((sum, item) => sum + (item.data?.summary?.total_shots || 0), 0);
    const validResultsCount = fetchedResults.filter(item => item.data?.summary?.avg_speed !== undefined).length;
    
    const avgSpeed = validResultsCount > 0 ? (
        fetchedResults.reduce((sum, item) => sum + (item.data?.summary?.avg_speed || 0), 0) / validResultsCount
    ) : 0;

    return (
        <div className="multi-results-container space-y-8">
            
            {/* 總體結果區塊 */}
            <div className="p-6 bg-indigo-50 border-l-4 border-indigo-500 rounded-lg shadow-md">
                <h4 className="text-xl font-bold text-indigo-800 mb-2">
                    {isBatchAnalysis ? '批次分析總結' : '歷史數據平均'}
                </h4>
                <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-gray-700">
                    <p>總影片數: <span className="font-semibold text-indigo-600">{fetchedResults.length}</span></p>
                    <p>總擊球數: <span className="font-semibold text-indigo-600">{totalShots}</span></p>
                    <p>平均速度: <span className="font-semibold text-indigo-600">{avgSpeed.toFixed(1)} km/h</span></p>
                    <p>球員: <span className="font-semibold">{playerSummary.person_name}</span></p>
                </div>
            </div>

            {/* 單次分析結果列表 */}
            <h4 className="text-xl font-semibold border-b pb-2">單次分析細節</h4>
            <div className="space-y-6">
                {fetchedResults.map((result, index) => (
                    <div key={result.fileId} className="bg-white p-5 border border-gray-200 rounded-lg shadow-sm hover:shadow-md transition duration-150">
                        <div className="flex justify-between items-start mb-3">
                            <p className="text-lg font-bold text-gray-800">
                                {result.fileName} 
                                <span className="text-sm font-normal text-gray-500 ml-2">({index + 1})</span>
                            </p>
                            <p className="text-sm text-gray-500">
                                上傳於: {new Date(result.uploadTime).toLocaleDateString()}
                            </p>
                        </div>
                        
                        {/* 顯示單筆數據的簡要總結 */}
                        <div className="grid grid-cols-2 sm:grid-cols-3 gap-3 text-sm">
                            <p>擊球數: <span className="font-medium text-blue-600">{result.data?.summary?.total_shots || 'N/A'}</span></p>
                            <p>發球初速: <span className="font-medium text-blue-600">{result.data?.summary?.initial_speed?.toFixed(1) || 'N/A'} km/h</span></p>
                            <p>穩定度: <span className="font-medium text-blue-600">{result.data?.summary?.stabilityScore?.toFixed(2) || 'N/A'}</span></p>
                        </div>
                        
                        {/* 這裡可以添加一個跳轉到詳細結果頁面 (e.g. /results/:fileId) 的連結 */}
                        <div className="mt-4">
                            {/* <Link to={`/results/${result.fileId}`} className="text-indigo-500 hover:text-indigo-700 font-medium text-sm">
                                查看完整報告 →
                            </Link> */}
                        </div>
                    </div>
                ))}
            </div>
        </div>
    );
};

export default MultiResultsDisplay;