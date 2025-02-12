import os
import argparse
import numpy as np
from PIL import Image
import joblib
import cudf
from cuml.preprocessing import StandardScaler

# 配置参数
IMG_SIZE = (64, 64)
GRAYSCALE = True

def load_model(model_path):
    """加载模型及预处理管道"""
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
    
    # 转换为GPU数据结构
    features_gpu = cudf.DataFrame([features])
    
    # 标准化
    scaled = scaler.transform(features_gpu)
    
    # PCA转换
    if pca is not None:
        transformed = pca.transform(scaled)
    else:
        transformed = scaled
    
    # 预测
    pred = model.predict(transformed)
    return pred[0]

def predict_batch(folder_path, model, scaler, pca, class_names):
    """批量预测"""
    results = []
    for fname in os.listdir(folder_path):
        if fname.lower().endswith(('.png', '.jpg', '.jpeg')):
            path = os.path.join(folder_path, fname)
            pred_idx = predict_single(path, model, scaler, pca)
            if pred_idx is not None:
                results.append({
                    'filename': fname,
                    'class': class_names[pred_idx],
                    'class_id': pred_idx
                })
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='SVM图像分类预测')
    parser.add_argument('input', help='输入图像路径或目录')
    parser.add_argument('--model', default='./model/SVM_GPU_model.pkl', 
                      help='模型路径')
    args = parser.parse_args()

    # 加载模型
    model, scaler, pca, classes = load_model(args.model)
    print(f"已加载模型，类别数: {len(classes)}")

    # 执行预测
    if os.path.isfile(args.input):
        pred_id = predict_single(args.input, model, scaler, pca)
        if pred_id is not None:
            print(f"\n预测结果: {classes[pred_id]}")
    elif os.path.isdir(args.input):
        results = predict_batch(args.input, model, scaler, pca, classes)
        print(f"\n批量预测完成，共 {len(results)} 个结果:")
        for res in results:
            print(f"{res['filename']} → {res['class']}")
    else:
        print("无效的输入路径") 