import os
import time
import numpy as np
from PIL import Image
import cudf
import cuml
# from cuml.svm import SVC
from cuml.svm import LinearSVC
from cuml.preprocessing import StandardScaler
from cuml.decomposition import PCA
from sklearn.metrics import accuracy_score, classification_report

# 配置参数
IMG_SIZE = (64, 64)
GRAYSCALE = True
DATA_DIR = './data/train'
MODEL_PATH = './model/svm_gpu_model.pkl'

# --------------------------
# 1. GPU数据加载与预处理
# --------------------------
def load_gpu_data(folder_path):
    """使用cuDF加速数据加载"""
    images = []
    labels = []
    class_names = sorted([d for d in os.listdir(folder_path) 
                        if os.path.isdir(os.path.join(folder_path, d))])
    
    for label, cls in enumerate(class_names):
        cls_dir = os.path.join(folder_path, cls)
        print(f"Loading {cls}...", end=' ')
        
        for fname in os.listdir(cls_dir):
            if fname.startswith('.'): continue
            try:
                img = Image.open(os.path.join(cls_dir, fname))
                if GRAYSCALE:
                    img = img.convert('L')
                else:
                    img = img.convert('RGB')
                img = img.resize(IMG_SIZE)
                images.append(np.array(img).flatten().astype(np.float32))
                labels.append(label)
            except Exception as e:
                print(f"Error loading {fname}: {str(e)}")
        
        print(f"Loaded {len(os.listdir(cls_dir))} samples")
    
    # 转换为cuDF数据结构
    return cudf.DataFrame(images), cudf.Series(labels), class_names

# --------------------------
# 2. 主流程
# --------------------------
def main():
    # 加载数据
    print("Loading training data...")
    X_train, y_train, class_names = load_gpu_data(DATA_DIR)
    
    # 数据标准化
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    # 添加维度验证
    if X_train_scaled.shape[1] < 2:
        print("特征维度不足，跳过PCA")
        X_train_pca = X_train_scaled
    else:
        # 计算最优组件数
        max_components = min(X_train_scaled.shape[1], 300)
        pca = PCA(n_components=max_components)
        pca.fit(X_train_scaled)
        
        # 处理可能的空值情况
        explained_variance = np.cumsum(pca.explained_variance_ratio_)
        optimal_components = max(1, np.argmax(explained_variance >= 0.95) + 1)
        print(f"Optimal components: {optimal_components}")
        
        # 重新拟合PCA
        pca = PCA(n_components=optimal_components)
        X_train_pca = pca.fit_transform(X_train_scaled)
    
    # 修改后的维度处理
    if X_train_pca.ndim == 1:
        X_train_pca = X_train_pca.to_frame()  # 将Series转换为DataFrame
    
    # 确保数据类型统一
    X_train_pca = cudf.DataFrame(X_train_pca)
    
    print(f"最终特征维度: {X_train_pca.shape[1]}")
    
    # 在训练前添加类型验证
    if isinstance(X_train_pca, cudf.DataFrame):
        print(f"特征矩阵类型: {type(X_train_pca)}, 形状: {X_train_pca.shape}")
    else:
        raise TypeError("特征矩阵必须是cuDF DataFrame")
    
    # svm_model = SVC(
    #     kernel='linear', 
    #     C=0.5,
    #     cache_size=4096,  # 使用更大缓存
    #     probability=False
    # )
    
    # 训练SVM（GPU加速）
    svm_model = LinearSVC(
        penalty='l2',          # 正则化类型
        loss='squared_hinge', # 损失函数
        C=0.5,                # 正则化强度
        max_iter=1000,        # 最大迭代次数
        tol=1e-3,             # 停止阈值
        linesearch_max_iter=50, # 线性搜索次数
        verbose=1             # 显示训练日志
    )
    
    print("Training GPU SVM...")
    start = time.time()
    svm_model.fit(
        X_train_pca.astype(np.float32),  # 确保数据类型一致
        y_train.astype(np.int32)
    )
    print(f"Training time: {time.time()-start:.2f}s")
    
    # 修改数据转换部分（LinearSVC需要数组输入）
    X_train_pca = X_train_pca.values if isinstance(X_train_pca, cudf.DataFrame) else X_train_pca
    
    # 保存完整pipeline
    from joblib import dump
    dump({
        'model': svm_model,
        'scaler': scaler,
        'pca': pca,
        'classes': class_names
    }, MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")

if __name__ == "__main__":
    main() 