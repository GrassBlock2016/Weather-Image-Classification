import os
import argparse
import numpy as np
from PIL import Image
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# 配置参数
IMG_SIZE = (64, 64)
GRAYSCALE = True

def load_model(model_path):
    """加载CPU模型"""
    pipeline = joblib.load(model_path)
    return (
        pipeline['model'], 
        pipeline['scaler'],
        pipeline['pca'],
        pipeline['classes']
    )

def preprocess_image(image_path):
    """图像预处理（与训练一致）"""
    try:
        img = Image.open(image_path)
        if GRAYSCALE:
            img = img.convert('L')
        else:
            img = img.convert('RGB')
        img = img.resize(IMG_SIZE)
        return np.array(img).flatten().astype(np.float32)
    except Exception as e:
        print(f"处理 {image_path} 出错: {str(e)}")
        return None

def predict_single(image_path, model, scaler, pca):
    """单张图像预测"""
    features = preprocess_image(image_path)
    if features is None:
        return None
    
    # 标准化
    scaled = scaler.transform([features])
    
    # PCA转换
    if pca is not None:
        transformed = pca.transform(scaled)
    else:
        transformed = scaled
    
    # 预测
    pred = model.predict(transformed)
    return pred[0]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='SVM图像分类预测(CPU版)')
    parser.add_argument('input', help='输入图像路径或目录')
    parser.add_argument('--model', default='./model/SVM_CPU_model.pkl', 
                      help='模型路径')
    args = parser.parse_args()

    # 加载模型
    model, scaler, pca, classes = load_model(args.model)
    print(f"已加载模型，类别数: {len(classes)}")

    # 执行预测（仅支持单张预测）
    if os.path.isfile(args.input):
        pred_id = predict_single(args.input, model, scaler, pca)
        if pred_id is not None:
            print(f"\n预测结果: {classes[pred_id]}")
    else:
        print("CPU版本暂不支持批量预测") 