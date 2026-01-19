import axios from 'axios';

// 使用環境變數或預設值設置 API 基礎 URL
const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:5000/api';

const api = axios.create({
  baseURL: API_BASE_URL,
  timeout: 30000, // 30 seconds
});

export interface VideoInfo {
  duration: number;
  fps: number;
  width: number;
  height: number;
  frame_count: number;
}

export interface UploadResponse {
  success: boolean;
  file_id?: string;
  filename?: string;
  video_info?: VideoInfo;
  error?: string;
}

// 擊球分析的詳細結構
export interface Shot {
  frame: number;
  timestamp: number;
  type: 'forehand' | 'backhand';
  side: 'left' | 'right';
  confidence: number;
  ball_contact_frame: number;
  swing_velocity: number;
}

export interface SpeedDistribution {
  range: string;
  count: number;
  percentage: number;
}

export interface TrajectorySpeed {
  trajectory_id: number;
  speeds: number[];
  max_speed: number;
  avg_speed: number;
}

// 單一檔案分析結果的完整結構
export interface AnalysisResults {
  file_id: string;
  timestamp: string;
  tracking: any;
  shots: {
    shots: Shot[];
    total_shots: number;
    forehand_count: number;
    backhand_count: number;
  };
  speed: {
    max_speed: number;
    avg_speed: number;
    max_speed_kmh: number;
    avg_speed_kmh: number;
    speed_distribution: SpeedDistribution[];
    trajectory_speeds: TrajectorySpeed[];
  };
  summary: {
    total_shots: number;
    forehand_count: number;
    backhand_count: number;
    max_speed: number;
    avg_speed: number;
  };
}


//批次上傳時，單個檔案上傳成功的返回結果
export interface BatchUploadFileResult extends UploadResponse {
    personName: string;
}

//擴展 AnalysisResults，納入新的指標
export interface SingleVideoResult extends AnalysisResults {
    initialVelocity: number; 
    hitHeight: number;      
    hitAngle: number;        
    stabilityScore: number;  
    swingSpeed: number;      
    
    fileName: string; 
    personName: string; 
}

// 獲取整個批次結果的頂層結構
export interface AnalysisBatchData {
    personName: string; 
    batchId: string;    
    results: SingleVideoResult[]; 
}

// 【移除】用於前端傳遞帶有 personName 的 File 物件，改為傳遞純 File
// interface UploadFileWithPersonName extends File {
//     personName?: string; 
// }

// 上傳影片 (單檔案，【修正】傳入純 File 和 personName)
export const uploadVideo = async (
  file: File, // 👈 接受純淨的 File 類型
  personName: string, // 👈 額外傳遞人名
  onProgress?: (progress: number) => void
): Promise<UploadResponse> => {
  
  const formData = new FormData();
  
  //確保 FormData 正確地包含檔案本身和其名稱 (file 必須是 Blob 實例)
  formData.append('video', file, file.name); 

  if (personName) {
      formData.append('person_name', personName); // 傳遞人名給後端
  }
  
  try {
    const response = await api.post('/upload', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
      onUploadProgress: (progressEvent) => {
        if (progressEvent.total && onProgress) {
          const progress = (progressEvent.loaded / progressEvent.total) * 100;
          onProgress(progress);
        }
      },
    });

    // 成功時返回後端響應
    return response.data;

  } catch (error: any) {
    // 處理 Axios 拋出的錯誤 (4xx/5xx 狀態碼)
    const errorMessage = error.response?.data?.error || '網路連線或伺服器上傳處理失敗';

    //返回 UploadResponse 類型，供前端 MultiUploadPage 檢查 result.success
    return { 
        success: false, 
        error: errorMessage
    };
  }
};

//多檔案上傳函數 (保持不變)
export const uploadBatchVideos = async (
    files: { file: File; personName: string }[],
    onProgress?: (progress: number) => void
): Promise<{ success: boolean; batchId: string; results: BatchUploadFileResult[] }> => {

    if (files.length === 0) {
        throw new Error('沒有影片檔案可以上傳');
    }

    const formData = new FormData();
    formData.append('person_name', files[0].personName || '未命名'); 
    
    files.forEach((item, index) => {
        if (item.file && item.file.name) {
            formData.append(`video_${index}`, item.file, item.file.name);
        }
    });

    try {
        const response = await api.post('/multi-upload', formData, {
            headers: { 'Content-Type': 'multipart/form-data' },
            onUploadProgress: (progressEvent) => {
                if (progressEvent.total && onProgress) {
                    const progress = (progressEvent.loaded / progressEvent.total) * 100;
                    onProgress(progress);
                }
            },
        });
        
        return response.data; 
    } catch (error: any) {
        throw new Error(error.response?.data?.error || '批量上傳失敗');
    }
};


// 開始分析 (單檔案，保持原始函數)
export const analyzeVideo = async (fileId: string): Promise<AnalysisResults> => {
  try {
    const response = await api.post(`/analyze/${fileId}`, null, { timeout: 0 }); 
    return response.data.results;
  } catch (error: any) {
    throw new Error(error.response?.data?.error || '分析失敗');
  }
};

//開始批次分析 (保持不變)
export const analyzeBatchVideos = async (batchId: string): Promise<{ success: boolean }> => {
    try {
        const response = await api.post(`/analyze-batch/${batchId}`, null, { timeout: 0 });
        return response.data;
    } catch (error: any) {
        throw new Error(error.response?.data?.error || '批次分析失敗');
    }
};


// 獲取分析結果 (單檔案，保持不變)
export const getResults = async (fileId: string): Promise<AnalysisResults> => {
  try {
    const response = await api.get(`/results/${fileId}`);
    return response.data;
  } catch (error: any) {
    throw new Error(error.response?.data?.error || '獲取結果失敗');
  }
};

//獲取批次分析結果 (保持不變)
export const getBatchResults = async (batchId: string): Promise<AnalysisBatchData> => {
    try {
        const response = await api.get(`/batch-results/${batchId}`);
        return response.data;
    } catch (error: any) {
        throw new Error(error.response?.data?.error || '獲取批次結果失敗');
    }
};


// 獲取原始影片 URL (保持不變)
export const getVideoUrl = (fileId: string): string => {
  return `${API_BASE_URL}/video/${fileId}`;
};

// 獲取處理後影片 URL (保持不變)
export const getProcessedVideoUrl = (fileId: string): string => {
  const ts = Date.now();
  return `${API_BASE_URL}/processed-video/${fileId}?t=${ts}`;
};

// 健康檢查 (保持不變)
export const healthCheck = async (): Promise<{ status: string; timestamp: string }> => {
  try {
    const response = await api.get('/health');
    return response.data;
  } catch (error: any) {
    throw new Error('無法連接到後端服務');
  }
};

export default api;