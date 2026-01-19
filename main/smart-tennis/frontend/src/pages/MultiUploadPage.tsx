import React, { useState, useCallback, useMemo } from 'react';
import { useDropzone } from 'react-dropzone';
import { useNavigate } from 'react-router-dom';
import { uploadVideo } from '../services/api'; 
// 假設您在 '../services/api' 中有 uploadVideo 函數的定義

// 定義一個用於記錄上傳檔案的介面
interface UploadFileWithMeta {
    file: File; 
    personName?: string; 
    uploadStatus?: 'pending' | 'uploading' | 'success' | 'failed' | 'analyzing' | 'analysis_failed';
    uploadProgress?: number;
    fileId?: string; 
    errorMessage?: string; 
    originalFileName: string; // 💥 新增: 原始檔案名稱
}

const MultiUploadPage: React.FC = () => {
    const [personName, setPersonName] = useState<string>('');
    const [filesToUpload, setFilesToUpload] = useState<UploadFileWithMeta[]>([]);
    const [isUploading, setIsUploading] = useState(false);
    const [globalError, setGlobalError] = useState<string | null>(null);
    const navigate = useNavigate();

    // 處理檔案拖放或選擇
    const onDrop = useCallback((acceptedFiles: File[]) => {
        const newFiles: UploadFileWithMeta[] = acceptedFiles.map(file => ({
            file: file, // 儲存原始檔案
            personName: personName || '未命名', 
            uploadStatus: 'pending',
            uploadProgress: 0,
            originalFileName: file.name, // 💥 記錄原始檔案名稱
        }));
        
        setFilesToUpload(prevFiles => [...prevFiles, ...newFiles]);
        setGlobalError(null); 
    }, [personName]);

    // 處理人名輸入框的變化
    const handleNameChange = (e: React.ChangeEvent<HTMLInputElement>) => {
        const newName = e.target.value;
        setPersonName(newName);
        
        // 更新所有待處理檔案的 personName
        setFilesToUpload(prevFiles => prevFiles.map(metaFile => 
            metaFile.uploadStatus === 'pending' ? { ...metaFile, personName: newName || '未命名' } : metaFile
        ));
    };

    // 處理移除檔案
    const handleRemoveFile = (fileName: string) => {
        setFilesToUpload(prevFiles => prevFiles.filter(metaFile => metaFile.file.name !== fileName));
    };

    // 處理開始上傳與分析 (依序處理)
    const handleUploadAll = async () => {
        if (isUploading) return;

        const filesToProcess = filesToUpload.filter(f => f.uploadStatus === 'pending' || f.uploadStatus === 'failed' || f.uploadStatus === 'analysis_failed');
        if (filesToProcess.length === 0) {
            setGlobalError("未找到可處理的檔案。");
            return;
        }

        setIsUploading(true);
        setGlobalError(null);
        // 💥 收集成功檔案的中繼資料 (用於導航傳遞新名稱)
        let successfulFileMetas: { fileId: string; fileName: string; originalIndex: number }[] = [];
        let totalFailedCount = 0;
        let successfulCount = 0; // 用於影片編號計數

        const updatedFiles = [...filesToUpload];

        for (let i = 0; i < updatedFiles.length; i++) {
            let metaFile = updatedFiles[i]; // 從副本中獲取檔案資訊
            
            if (metaFile.uploadStatus !== 'pending' && metaFile.uploadStatus !== 'failed' && metaFile.uploadStatus !== 'analysis_failed') continue;

            const fileToUpload = metaFile.file;

            // --- 嚴格 Blob 檢查 ---
            if (!fileToUpload || !(fileToUpload instanceof File)) {
                totalFailedCount++;
                metaFile = { 
                    ...metaFile, 
                    uploadStatus: 'failed', 
                    errorMessage: `檔案 ${metaFile.file.name || '未知'} 在上傳時遺失或無效。`
                };
                updatedFiles[i] = metaFile;
                continue; // 跳過此無效檔案
            }

            // --- 步驟 1: 上傳準備 ---
            setFilesToUpload(prev => prev.map((f, index) => 
                index === i ? { ...f, uploadStatus: 'uploading', uploadProgress: 0, errorMessage: undefined } : f
            ));
            
            let currentFileId: string | undefined;

            try {
                // 傳遞已檢查的 fileToUpload (Blob) 和人名
                const result = await uploadVideo(fileToUpload, metaFile.personName || '未命名', (progress) => {
                    // 使用 setFilesToUpload 函數式更新確保進度條同步
                    setFilesToUpload(prev => prev.map((f, index) => 
                        index === i ? { ...f, uploadProgress: progress } : f
                    ));
                });

                if (!result.success) {
                    throw new Error(result.error || '上傳失敗');
                }
                currentFileId = result.file_id;

                // --- 步驟 2: 觸發分析 (強化錯誤處理) ---
                setFilesToUpload(prev => prev.map((f, index) => 
                    index === i ? { ...f, uploadStatus: 'analyzing', fileId: currentFileId } : f
                ));

                const analyzeRes = await fetch(`/api/analyze/${currentFileId!}`, { method: 'POST' });
                let analyzeData;

                try {
                    analyzeData = await analyzeRes.json();
                } catch (jsonError) {
                    throw new Error(`分析API回應格式錯誤 (HTTP ${analyzeRes.status})`);
                }

                if (!analyzeRes.ok || !analyzeData.success) {
                    throw new Error(`分析觸發失敗: ${analyzeData.error || '未知錯誤'}`);
                }

                // 成功
                successfulCount++; // 💥 成功計數器增加
                setFilesToUpload(prev => prev.map((f, index) => 
                    index === i ? { ...f, uploadStatus: 'success', fileId: currentFileId } : f
                ));
                
                // 💥 收集成功檔案的中繼資料，並使用人名+影片編號N作為顯示名稱
                successfulFileMetas.push({ 
                    fileId: currentFileId!, 
                    fileName: `${metaFile.personName || '未命名'} 影片 ${successfulCount}`,
                    originalIndex: i 
                });

            } catch (err: any) {
                // 處理上傳或分析失敗並區分狀態
                totalFailedCount++;

                const isAnalysisFailure = currentFileId !== undefined;
                const errorDetails = err.message || (isAnalysisFailure ? '無法取得伺服器分析錯誤' : '無法取得上傳錯誤');
                const specificErrorMessage = isAnalysisFailure ? 
                    `分析失敗: ${errorDetails}` : 
                    `上傳失敗: ${errorDetails}`;

                setFilesToUpload(prev => prev.map((f, index) => 
                    index === i ? { 
                        ...f, 
                        uploadStatus: isAnalysisFailure ? 'analysis_failed' : 'failed', 
                        uploadProgress: 0, 
                        errorMessage: specificErrorMessage
                    } : f
                ));
            }
        } // 結束 for 循環

        setIsUploading(false);

        // --- 步驟 3: 導航至結果頁面 ---
        if (successfulFileMetas.length > 0) {
            const batchIdString = successfulFileMetas.map(m => m.fileId).join(',');
            // 💥 傳遞包含新檔案名稱和人名的 state
            const stateToPass = { 
                personName: personName || '未命名球員',
                fileMetaList: successfulFileMetas 
            };
            
            navigate(`/multi-results/${batchIdString}`, { state: stateToPass, replace: true }); 
        } else if (totalFailedCount > 0) {
            setGlobalError(`所有嘗試上傳/分析的檔案均失敗 (${totalFailedCount} 個)。請檢查錯誤日誌。`);
        } else {
            setGlobalError("未找到可處理的檔案。");
        }
    };
    
    // 檢查是否有檔案處於待處理或失敗狀態
    const hasPendingOrFailedFiles = useMemo(() => 
        filesToUpload.some(f => f.uploadStatus === 'pending' || f.uploadStatus === 'failed' || f.uploadStatus === 'analysis_failed'), 
        [filesToUpload]
    );
    
    // Dropzone 配置
    const { getRootProps, getInputProps, isDragActive } = useDropzone({
        onDrop,
        accept: {
            'video/*': ['.mp4', '.avi', '.mov', '.mkv']
        },
        maxSize: 100 * 1024 * 1024,
        multiple: true,
        disabled: isUploading 
    });

    // 確定上傳按鈕的文字
    const uploadButtonText = useMemo(() => {
        if (isUploading) {
            const currentFile = filesToUpload.find(f => f.uploadStatus === 'uploading' || f.uploadStatus === 'analyzing');
            if (currentFile) {
                if (currentFile.uploadStatus === 'uploading') {
                    return `📤 正在上傳 ${currentFile.file.name}: ${currentFile.uploadProgress?.toFixed(1)}%...`;
                }
                return `⚙️ 正在分析 ${currentFile.file.name}...`;
            }
            return '⚙️ 正在處理檔案...';
        }
        const pendingCount = filesToUpload.filter(f => f.uploadStatus === 'pending' || f.uploadStatus === 'failed' || f.uploadStatus === 'analysis_failed').length;
        return `🚀 開始上傳與分析所有待處理檔案 (${pendingCount})`;
    }, [isUploading, filesToUpload]);

    return (
        <div className="max-w-4xl mx-auto">
            <div className="text-center mb-8">
                <h1 className="text-4xl font-bold text-gray-800 mb-4">
                    🎾 上傳多個網球影片
                </h1>
                <p className="text-lg text-gray-600">
                    支援 MP4、AVI、MOV、MKV 格式，檔案大小限制 100MB / 單個檔案
                </p>
            </div>

            <div className="bg-white rounded-lg shadow-lg p-8 mb-6">
                {/* 1. 人名輸入欄位 */}
                <div className="mb-6">
                    <label 
                        htmlFor="person-name-input" 
                        className="block text-lg font-medium text-gray-700 mb-2"
                    >
                        🧑 欲分析球員姓名
                    </label>
                    <input
                        id="person-name-input" 
                        type="text"
                        value={personName}
                        onChange={handleNameChange}
                        placeholder="請輸入球員姓名 (e.g., 王小明)"
                        className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-blue-500 focus:border-blue-500 text-gray-900"
                        disabled={isUploading}
                    />
                </div>

                {/* 2. 拖放區域 */}
                <div
                    {...getRootProps()}
                    className={`border-2 border-dashed rounded-lg p-12 text-center transition-colors ${
                        isUploading ? 'cursor-not-allowed opacity-70' : 'cursor-pointer'
                    } ${
                        isDragActive
                            ? 'border-blue-500 bg-blue-50'
                            : 'border-gray-300 hover:border-blue-400 hover:bg-gray-50'
                    }`}
                >
                    <input {...getInputProps()} />
                    <div className="text-6xl mb-4">🎥</div>
                    {isDragActive ? (
                        <p className="text-xl text-blue-600">放開以上傳多個檔案...</p>
                    ) : (
                        <div>
                            <p className="text-xl text-gray-600 mb-2">
                                拖拽多個影片檔案到此處，或點擊選擇檔案
                            </p>
                            <p className="text-sm text-gray-500">
                                支援的格式：MP4, AVI, MOV, MKV（最多可選）
                            </p>
                        </div>
                    )}
                </div>

                {/* 3. 檔案列表 */}
                {filesToUpload.length > 0 && (
                    <div className="mt-6 border-t pt-4">
                        <h3 className="text-xl font-semibold mb-4 text-gray-800">
                            📂 待上傳影片 ({filesToUpload.length} 個)
                        </h3>
                        <ul className="space-y-3">
                            {filesToUpload.map((metaFile, index) => (
                                <li 
                                    key={metaFile.file.name + index} 
                                    className="flex items-start justify-between p-3 bg-gray-50 rounded-lg border"
                                >
                                    
                                    <div className="flex-1 min-w-0 mr-4"> 
                                        <p className="text-sm font-medium text-gray-800 truncate">
                                            {metaFile.file.name} 
                                            <span className="ml-2 text-xs text-blue-600 bg-blue-100 px-2 py-0.5 rounded-full">
                                                {metaFile.personName}
                                            </span>
                                        </p>
                                        <p className="text-xs text-gray-500 mt-1">
                                            大小: {(metaFile.file.size / (1024 * 1024)).toFixed(2)} MB
                                        </p>
                                        
                                        {/* 上傳進度/狀態顯示區 */}
                                        <div className="mt-2">
                                            {metaFile.uploadStatus === 'uploading' && (
                                                <>
                                                    <div className="w-full bg-gray-200 rounded-full h-2.5">
                                                        <div
                                                            className="bg-blue-600 h-2.5 rounded-full transition-all duration-300"
                                                            style={{ width: `${metaFile.uploadProgress}%` }}
                                                        ></div>
                                                    </div>
                                                    <span className="text-xs text-blue-600 block mt-1">
                                                        正在上傳: {metaFile.uploadProgress?.toFixed(1)}%
                                                    </span>
                                                </>
                                            )}
                                            {metaFile.uploadStatus === 'analyzing' && (
                                                <span className="text-sm text-indigo-600 mt-1 flex items-center animate-pulse">
                                                    ⏳ 正在觸發分析...
                                                </span>
                                            )}
                                            {metaFile.uploadStatus === 'success' && (
                                                <span className="text-sm text-green-600 mt-1 flex items-center">
                                                    ✅ 上傳並分析成功
                                                </span>
                                            )}
                                            {metaFile.uploadStatus === 'analysis_failed' && (
                                                <span className="text-sm text-yellow-700 mt-1 flex items-center">
                                                    ⚠️ 上傳成功，但分析失敗
                                                </span>
                                            )}
                                            {(metaFile.uploadStatus === 'failed' || metaFile.uploadStatus === 'pending') && (
                                                <span className={`text-sm mt-1 flex items-center ${metaFile.uploadStatus === 'failed' ? 'text-red-600' : 'text-gray-500'}`}>
                                                    {metaFile.uploadStatus === 'failed' ? '❌ 上傳失敗' : '待處理'}
                                                </span>
                                            )}
                                            {metaFile.errorMessage && metaFile.uploadStatus !== 'success' && (
                                                <p className="text-xs text-red-500 mt-1 italic truncate">
                                                    錯誤: {metaFile.errorMessage}
                                                </p>
                                            )}
                                        </div>
                                    </div>
                                    <button
                                        onClick={() => handleRemoveFile(metaFile.file.name)}
                                        disabled={isUploading}
                                        className="p-1 text-red-500 hover:text-red-700 disabled:opacity-50"
                                    >
                                        🗑️
                                    </button>
                                </li>
                            ))}
                        </ul>

                        {/* 上傳按鈕 */}
                        <button
                            onClick={handleUploadAll}
                            disabled={!hasPendingOrFailedFiles || isUploading}
                            className={`w-full mt-6 py-3 px-4 text-white font-semibold rounded-lg shadow-md transition-colors ${
                                !hasPendingOrFailedFiles || isUploading
                                    ? 'bg-gray-400 cursor-not-allowed'
                                    : 'bg-blue-600 hover:bg-blue-700 focus:outline-none focus:ring-4 focus:ring-blue-300'
                            }`}
                        >
                            {uploadButtonText}
                        </button>
                    </div>
                )}

                {/* 4. 全局錯誤訊息 */}
                {globalError && (
                    <div className="mt-6 p-4 bg-red-50 border border-red-200 rounded-lg">
                        <div className="flex">
                            <div className="text-red-600 text-xl mr-3">❌</div>
                            <div>
                                <h4 className="text-red-800 font-semibold">操作錯誤</h4>
                                <p className="text-red-600">{globalError}</p>
                            </div>
                        </div>
                    </div>
                )}
            </div>

            {/* 5. 注意事項 */}
            <div className="mt-8 bg-blue-50 rounded-lg p-6 border border-blue-200">
                <h2 className="text-xl font-semibold mb-4">📋 上傳注意事項</h2>
                <ul className="space-y-2 text-gray-700">
                    <li className="flex items-start">
                        <span className="text-blue-600 mr-2">•</span>
                        在拖拽或選擇檔案前，請先在上方**輸入球員姓名**，該人名將會自動套用至所有新加入的影片。
                    </li>
                    <li className="flex items-start">
                        <span className="text-blue-600 mr-2">•</span>
                        點擊 **「開始上傳與分析」** 後，系統會依序處理所有待上傳或上傳失敗的檔案。
                    </li>
                    <li className="flex items-start">
                        <span className="text-blue-600 mr-2">•</span>
                        觸發分析是一個長時間操作。如果分析完成，頁面將會自動跳轉至**綜合分析報告頁面**。
                    </li>
                    <li className="flex items-start">
                        <span className="text-blue-600 mr-2">•</span>
                        在處理大量或較大檔案時，建議使用高解析度、影片長度在 **30 秒到 5 分鐘** 之間的影片。
                    </li>
                </ul>
            </div>
        </div>
    );
};

export default MultiUploadPage;