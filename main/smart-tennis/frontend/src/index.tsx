import React from 'react';
import ReactDOM from 'react-dom/client';
import './index.css';
import App from './App';

const root = ReactDOM.createRoot(
  document.getElementById('root') as HTMLElement
);
root.render(
  // 注意：StrictMode 會在開發環境觸發 useEffect 兩次
  // 這會導致分析 API 被呼叫兩次。為避免重複分析，我們暫時移除 StrictMode。
  <App />
);
