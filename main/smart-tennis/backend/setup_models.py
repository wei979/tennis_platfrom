#!/usr/bin/env python3
"""
下載和設置必要的 AI 模型
"""

import os
import urllib.request
from ultralytics import YOLO

def download_yolo_model():
    """下載 YOLOv8 模型"""
    models_dir = '../models'
    os.makedirs(models_dir, exist_ok=True)
    
    model_path = os.path.join(models_dir, 'yolov8n.pt')
    
    if not os.path.exists(model_path):
        print("正在下載 YOLOv8 nano 模型...")
        try:
            # 使用 ultralytics 自動下載
            model = YOLO('yolov8n.pt')
            model.save(model_path)
            print(f"✅ 模型已下載到: {model_path}")
        except Exception as e:
            print(f"❌ 下載失敗: {e}")
    else:
        print(f"✅ 模型已存在: {model_path}")

def setup_models():
    """設置所有必要的模型"""
    print("🤖 設置 AI 模型...")
    
    # 下載 YOLO 模型
    download_yolo_model()
    
    print("🎉 模型設置完成！")

if __name__ == "__main__":
    setup_models()
