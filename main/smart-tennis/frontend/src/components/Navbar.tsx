import React from 'react';
import { Link, useLocation } from 'react-router-dom';

const Navbar: React.FC = () => {
  const location = useLocation();

  const isActive = (path: string) => {
    return location.pathname === path;
  };

  return (
    <nav className="bg-blue-600 text-white shadow-lg">
      <div className="container mx-auto px-4">
        <div className="flex justify-between items-center py-4">
          <Link to="/" className="text-2xl font-bold flex items-center">
            🎾 Smart Tennis
          </Link>
          
          <div className="flex space-x-6">
            <Link
              to="/"
              className={`px-4 py-2 rounded-lg transition-colors ${
                isActive('/') ? 'bg-blue-700' : 'hover:bg-blue-500'
              }`}
            >
              首頁
            </Link>
            <Link
              to="/upload"
              className={`px-4 py-2 rounded-lg transition-colors ${
                isActive('/upload') ? 'bg-blue-700' : 'hover:bg-blue-500'
              }`}
            >
              上傳影片
            </Link>
          </div>
        </div>
      </div>
    </nav>
  );
};

export default Navbar;
