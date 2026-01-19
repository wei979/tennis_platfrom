import React, { useState, useEffect } from 'react';
import { 
  Radar, 
  RadarChart, 
  PolarGrid, 
  PolarAngleAxis, 
  PolarRadiusAxis, 
  ResponsiveContainer, 
  Legend 
} from 'recharts';
import { Play, BarChart2, X, Zap, Activity, Target, Trophy, Gauge, Search, ChevronRight } from 'lucide-react';
import axios from 'axios';

const API_BASE = "http://localhost:5000/api";

// --- 徹底解決 TSX 類型檢查報錯：將組件轉型為 any ---
const AnyRadarChart = RadarChart as any;
const AnyPolarGrid = PolarGrid as any;
const AnyPolarAngleAxis = PolarAngleAxis as any;
const AnyPolarRadiusAxis = PolarRadiusAxis as any;
const AnyRadar = Radar as any;
const AnyLegend = Legend as any;

// --- TypeScript 介面定義 ---
interface Summary {
    max_speed: number;
    initial_speed: number;
    stabilityScore: number;
    hitHeight: number;
    hitAngle: number;
    total_shots: number;
}

interface AnalysisResult {
    fileId: string;
    fileName: string;
    uploadTime: string;
    data: {
        summary: Summary;
    };
}

const PlayerDashboardPage: React.FC = () => {
    const [players, setPlayers] = useState<string[]>([]);
    const [selectedPlayer, setSelectedPlayer] = useState<string>('');
    const [history, setHistory] = useState<AnalysisResult[]>([]);
    const [selectedIds, setSelectedIds] = useState<string[]>([]);
    const [showComparison, setShowComparison] = useState<boolean>(false);

    // 獲取球員名單
    useEffect(() => {
        axios.get<string[]>(`${API_BASE}/players`)
            .then(res => setPlayers(res.data))
            .catch(err => console.error("API Error:", err));
    }, []);

    // 獲取選中球員的歷史數據
    useEffect(() => {
        if (selectedPlayer) {
            axios.get<AnalysisResult[]>(`${API_BASE}/analysis/player/${selectedPlayer}`)
                .then(res => {
                    setHistory(res.data);
                    setSelectedIds([]);
                })
                .catch(() => setHistory([]));
        }
    }, [selectedPlayer]);

    const toggleSelect = (id: string) => {
        setSelectedIds(prev => 
            prev.includes(id) ? prev.filter(i => i !== id) : [...prev, id]
        );
    };

    // --- 符合「多重結果頁面」的雷達圖數據邏輯 ---
    const formatRadarData = (summary: Summary) => [
        { 
            subject: '穩定度', 
            A: summary.stabilityScore > 100 ? 100 : summary.stabilityScore, 
            fullMark: 100 
        },
        { 
            subject: '平均速度', 
            A: summary.max_speed || 0, 
            fullMark: 200 
        },
        { 
            subject: '初速度(力量)', 
            A: summary.initial_speed || summary.max_speed || 0, 
            fullMark: 200 
        },
        { 
            subject: '高度控制', 
            A: Math.min((summary.hitHeight || 0) * 40, 100), 
            fullMark: 100 
        },
        { 
            subject: '角度控制', 
            A: Math.max(100 - Math.abs(summary.hitAngle || 0) * 5, 0), 
            fullMark: 100 
        },
    ];

    const getComparisonData = () => {
        const subjects = ['穩定度', '平均速度', '初速度(力量)', '高度控制', '角度控制'];
        const selectedRecords = history.filter(item => selectedIds.includes(item.fileId));
        return subjects.map(sub => {
            const row: any = { subject: sub, fullMark: 100 };
            selectedRecords.forEach((item, idx) => {
                const data = formatRadarData(item.data.summary);
                const found = data.find(d => d.subject === sub);
                row[`match_${idx}`] = found ? found.A : 0;
            });
            return row;
        });
    };

    return (
        <div className="min-h-screen bg-gray-50 text-gray-800 p-8 font-sans">
            {/* Header 頂部導覽 */}
            <header className="max-w-7xl mx-auto flex flex-col md:flex-row justify-between items-center mb-10 gap-6">
                <div className="flex items-center gap-4">
                    <div className="bg-blue-600 p-3 rounded-2xl shadow-lg shadow-blue-200">
                        <Trophy className="text-white" size={28} />
                    </div>
                    <div>
                        <h1 className="text-3xl font-bold text-gray-900 tracking-tight">球員數據儀表板</h1>
                        <p className="text-gray-500 font-medium">Player Performance Analytics</p>
                    </div>
                </div>

                <div className="flex items-center gap-3 bg-white p-2 rounded-2xl shadow-sm border border-gray-100">
                    <div className="flex items-center px-3 text-gray-400">
                        <Search size={18} />
                    </div>
                    <select 
                        className="bg-transparent border-none text-gray-700 font-semibold pr-8 py-2 outline-none cursor-pointer"
                        onChange={(e) => setSelectedPlayer(e.target.value)}
                        value={selectedPlayer}
                    >
                        <option value="">選擇測試球員</option>
                        {players.map(p => <option key={p} value={p}>{p}</option>)}
                    </select>

                    {selectedIds.length >= 2 && (
                        <button 
                            onClick={() => setShowComparison(true)}
                            className="bg-blue-600 hover:bg-blue-700 text-white px-6 py-2.5 rounded-xl font-bold transition-all shadow-md flex items-center gap-2"
                        >
                            <BarChart2 size={18} /> 數據對比 ({selectedIds.length})
                        </button>
                    )}
                </div>
            </header>

            {/* 歷史列表卡片 */}
            <div className="max-w-7xl mx-auto grid grid-cols-1 xl:grid-cols-2 gap-8">
                {history.map((record) => (
                    <div key={record.fileId} className={`bg-white border transition-all duration-300 rounded-[2.5rem] shadow-sm hover:shadow-xl ${selectedIds.includes(record.fileId) ? 'border-blue-400 ring-4 ring-blue-50' : 'border-gray-100'}`}>
                        <div className="p-6 flex justify-between items-center border-b border-gray-50">
                            <div className="flex items-center gap-3">
                                <div className="w-2 h-10 bg-blue-500 rounded-full"></div>
                                <div>
                                    <h3 className="font-bold text-lg text-gray-900">{record.fileName}</h3>
                                    <p className="text-xs font-semibold text-gray-400 uppercase tracking-widest">{new Date(record.uploadTime).toLocaleString()}</p>
                                </div>
                            </div>
                            <input 
                                type="checkbox" 
                                checked={selectedIds.includes(record.fileId)} 
                                onChange={() => toggleSelect(record.fileId)} 
                                className="w-6 h-6 rounded-lg border-gray-300 text-blue-600 cursor-pointer" 
                            />
                        </div>

                        <div className="p-8 flex flex-col lg:flex-row gap-8">
                            <div className="flex-1 space-y-6">
                                <div className="aspect-video rounded-3xl overflow-hidden bg-gray-900 shadow-inner">
                                    <video controls className="w-full h-full object-contain" src={`${API_BASE}/processed-video/${record.fileId}`} />
                                </div>
                                <div className="grid grid-cols-3 gap-3">
                                    <WhiteStat label="最高速" value={`${record.data.summary.max_speed}`} unit="km/h" color="text-orange-500" />
                                    <WhiteStat label="擊球數" value={record.data.summary.total_shots} unit="shots" color="text-blue-500" />
                                    <WhiteStat label="穩定度" value={`${record.data.summary.stabilityScore.toFixed(1)}`} unit="%" color="text-emerald-500" />
                                </div>
                            </div>

                            <div className="w-full lg:w-72 h-72 bg-gray-50 rounded-[2rem] p-4 border border-gray-100">
                                <ResponsiveContainer width="100%" height="100%">
                                    <AnyRadarChart data={formatRadarData(record.data.summary)}>
                                        <AnyPolarGrid stroke="#e2e8f0" />
                                        <AnyPolarAngleAxis dataKey="subject" tick={{ fill: '#64748b', fontSize: 11, fontWeight: 700 }} />
                                        <AnyRadar dataKey="A" stroke="#3b82f6" fill="#3b82f6" fillOpacity={0.3} strokeWidth={3} />
                                    </AnyRadarChart>
                                </ResponsiveContainer>
                            </div>
                        </div>
                    </div>
                ))}
            </div>

            {/* 深度數據對比跳窗 (Overlay) */}
            {showComparison && (
                <div className="fixed inset-0 z-50 flex items-center justify-center bg-gray-900/60 backdrop-blur-md p-6">
                    <div className="bg-white rounded-[3.5rem] w-full max-w-7xl p-12 relative shadow-2xl animate-in fade-in zoom-in duration-300 overflow-y-auto max-h-[95vh]">
                        <button onClick={() => setShowComparison(false)} className="absolute top-10 right-10 text-gray-300 hover:text-gray-900 transition-colors">
                            <X size={44} />
                        </button>
                        
                        <div className="flex flex-col md:flex-row justify-between items-end mb-12 gap-4">
                            <div className="flex items-center gap-4">
                                <div className="bg-blue-100 p-4 rounded-3xl text-blue-600">
                                    <BarChart2 size={32} />
                                </div>
                                <div>
                                    <h2 className="text-4xl font-extrabold text-gray-900 tracking-tight">深度數據對比報告</h2>
                                    <p className="text-gray-500 font-medium">針對選定之 {selectedIds.length} 場訓練進行多維度分析</p>
                                </div>
                            </div>
                            <div className="bg-gray-50 px-6 py-3 rounded-2xl border border-gray-100">
                                <p className="text-xs font-bold text-gray-400 uppercase tracking-widest">平均穩定度</p>
                                <p className="text-2xl font-black text-blue-600">
                                    {(history.filter(i => selectedIds.includes(i.fileId)).reduce((acc, curr) => acc + curr.data.summary.stabilityScore, 0) / selectedIds.length).toFixed(1)}%
                                </p>
                            </div>
                        </div>

                        <div className="grid grid-cols-1 lg:grid-cols-12 gap-12">
                            {/* 左側：大型雷達圖 */}
                            <div className="lg:col-span-5 bg-gray-50 rounded-[3rem] p-10 border border-gray-100 flex flex-col items-center">
                                <h4 className="text-center font-bold text-gray-400 uppercase text-sm mb-6 tracking-tighter">技術指標覆蓋圖表</h4>
                                <div className="w-full h-[450px]">
                                    <ResponsiveContainer width="100%" height="100%">
                                        <AnyRadarChart data={getComparisonData()}>
                                            <AnyPolarGrid stroke="#cbd5e1" />
                                            <AnyPolarAngleAxis dataKey="subject" tick={{ fill: '#334155', fontSize: 13, fontWeight: 800 }} />
                                            <AnyPolarRadiusAxis angle={30} domain={[0, 100]} tick={false} axisLine={false} />
                                            {selectedIds.map((id, idx) => (
                                                <AnyRadar 
                                                    key={id} 
                                                    name={`對比場次 ${idx + 1}`} 
                                                    dataKey={`match_${idx}`} 
                                                    stroke={idx === 0 ? "#2563eb" : idx === 1 ? "#dc2626" : "#059669"} 
                                                    fill={idx === 0 ? "#3b82f6" : idx === 1 ? "#ef4444" : "#10b981"} 
                                                    fillOpacity={0.1} 
                                                    strokeWidth={4}
                                                />
                                            ))}
                                            <AnyLegend wrapperStyle={{ paddingTop: '30px' }} />
                                        </AnyRadarChart>
                                    </ResponsiveContainer>
                                </div>
                            </div>

                            {/* 右側：數據對比表格 */}
                            <div className="lg:col-span-7 space-y-6">
                                <div className="bg-white border border-gray-100 rounded-[2.5rem] shadow-sm overflow-hidden">
                                    <table className="w-full text-left border-collapse">
                                        <thead>
                                            <tr className="bg-gray-50/50">
                                                <th className="p-6 text-sm font-bold text-gray-400 uppercase">指標項目</th>
                                                {selectedIds.map((id, idx) => (
                                                    <th key={id} className="p-6 text-sm font-bold text-center" style={{ color: idx === 0 ? "#2563eb" : idx === 1 ? "#dc2626" : "#059669" }}>
                                                        場次 {idx + 1}
                                                    </th>
                                                ))}
                                            </tr>
                                        </thead>
                                        <tbody className="divide-y divide-gray-50">
                                            <ComparisonRow label="穩定度 (%)" field="stabilityScore" data={history.filter(i => selectedIds.includes(i.fileId))} isPercent />
                                            <ComparisonRow label="平均速度 (km/h)" field="max_speed" data={history.filter(i => selectedIds.includes(i.fileId))} />
                                            <ComparisonRow label="初速度 (km/h)" field="initial_speed" data={history.filter(i => selectedIds.includes(i.fileId))} />
                                            <ComparisonRow label="高度控制 (m)" field="hitHeight" data={history.filter(i => selectedIds.includes(i.fileId))} />
                                            <ComparisonRow label="總擊球次數" field="total_shots" data={history.filter(i => selectedIds.includes(i.fileId))} />
                                        </tbody>
                                    </table>
                                </div>

                                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                                    {history.filter(i => selectedIds.includes(i.fileId)).map((item, idx) => (
                                        <div key={item.fileId} className="p-6 rounded-[2rem] border border-gray-100 bg-white flex items-center justify-between group hover:border-blue-200 transition-all">
                                            <div className="flex items-center gap-4">
                                                <div className="w-10 h-10 rounded-xl flex items-center justify-center font-black text-white shadow-md" style={{ backgroundColor: idx === 0 ? "#2563eb" : idx === 1 ? "#dc2626" : "#059669" }}>
                                                    {idx + 1}
                                                </div>
                                                <div>
                                                    <p className="text-xs font-bold text-gray-400 uppercase leading-none mb-1">{new Date(item.uploadTime).toLocaleDateString()}</p>
                                                    <p className="font-bold text-gray-800 leading-none">{item.fileName}</p>
                                                </div>
                                            </div>
                                            <ChevronRight className="text-gray-200 group-hover:text-blue-400 transition-colors" />
                                        </div>
                                    ))}
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            )}
        </div>
    );
};

// --- 輔助子組件 ---

const WhiteStat = ({ label, value, unit, color }: { label: string, value: string | number, unit: string, color: string }) => (
    <div className="bg-white p-4 rounded-2xl border border-gray-100 shadow-sm text-center">
        <p className="text-[10px] font-bold text-gray-400 uppercase tracking-tighter mb-1">{label}</p>
        <div className="flex items-baseline justify-center gap-0.5">
            <span className={`text-xl font-black ${color}`}>{value}</span>
            <span className="text-[10px] font-bold text-gray-300">{unit}</span>
        </div>
    </div>
);

const ComparisonRow = ({ label, field, data, isPercent = false }: { label: string, field: string, data: any[], isPercent?: boolean }) => {
    const values = data.map(d => d.data.summary[field] || 0);
    const maxVal = Math.max(...values);

    return (
        <tr className="hover:bg-gray-50/30 transition-colors">
            <td className="p-6 font-bold text-gray-600">{label}</td>
            {data.map((item) => {
                const val = item.data.summary[field] || 0;
                const isBest = val === maxVal && values.length > 1;
                return (
                    <td key={item.fileId} className="p-6 text-center">
                        <span className={`text-lg font-black ${isBest ? 'text-blue-600' : 'text-gray-900'}`}>
                            {isPercent ? val.toFixed(1) : val}
                        </span>
                        {isBest && (
                            <span className="ml-2 text-[10px] bg-blue-100 text-blue-600 px-2 py-1 rounded-full font-bold uppercase">Best</span>
                        )}
                    </td>
                );
            })}
        </tr>
    );
};

export default PlayerDashboardPage;