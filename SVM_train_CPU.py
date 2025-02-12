import numpy as np
import os
from PIL import Image
from sklearn import svm
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV
from tqdm import tqdm
import time
import joblib
from sklearn.decomposition import PCA

# 定义图像处理参数
IMG_SIZE = (64, 64)  # 统一调整的图像尺寸
GRAYSCALE = True     # 是否转换为灰度图

# --------------------------
# 1. 数据加载与预处理函数
# --------------------------
def load_images_from_folder(base_folder):
    images = []
    labels = []
    class_names = sorted([d for d in os.listdir(base_folder) 
                         if os.path.isdir(os.path.join(base_folder, d))])
    
    for label, class_name in enumerate(class_names):
        class_dir = os.path.join(base_folder, class_name)
        print(f"Loading {class_name}...", end=' ')
        
        # 使用列表推导式提高加载速度
        class_images = []
        for filename in os.listdir(class_dir):
            img_path = os.path.join(class_dir, filename)
            try:
                img = Image.open(img_path)
                if GRAYSCALE:
                    img = img.convert('L')
                else:
                    img = img.convert('RGB')
                img = img.resize(IMG_SIZE)
                class_images.append(np.array(img).flatten())
            except Exception as e:
                print(f"\nError loading {img_path}: {str(e)}")
                continue
                
        images.extend(class_images)
        labels.extend([label]*len(class_images))
        print(f"Loaded {len(class_images)} samples")
    
    return np.array(images), np.array(labels), class_names

# --------------------------
# 2. 加载数据集（添加进度显示）
# --------------------------
def load_dataset(path, desc):
    print(f"\n{desc} data loading:")
    X, y, class_names = load_images_from_folder(path)
    print(f"{desc} data shape: {X.shape}")
    return X, y, class_names

print("="*40)
X_train, y_train, class_names = load_dataset("./data/train", "Training")
X_val, y_val, _ = load_dataset("./data/val", "Validation")
# X_test, y_test = load_dataset("./data/test", "Test")
print("="*40)

# --------------------------
# 3. 数据标准化与降维
# --------------------------
scaler = StandardScaler()
print("\n数据标准化...")
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)

# 添加PCA降维（保留95%方差）
pca = PCA(n_components=0.95)
X_train = pca.fit_transform(X_train)
X_val = pca.transform(X_val)
print(f"降维后特征维度：{X_train.shape[1]}")

# --------------------------
# 4. 创建优化后的SVM分类器
# --------------------------
svm_clf = svm.SVC(
    kernel='linear',
    C=0.5,  # 调整正则化强度
    cache_size=1000,  # 增大缓存
    random_state=42
)

# --------------------------
# 5. 高效训练
# --------------------------
print("\n开始训练SVM分类器...")
start_time = time.time()
svm_clf.fit(X_train, y_train)
print(f"训练完成，耗时：{time.time()-start_time:.1f}秒")

# --------------------------
# 6. 评估模型（添加详细分析）
# --------------------------
def detailed_evaluation(model, X, y, set_name, class_names):
    print(f"\n{'='*40}")
    print(f"{set_name} Set Evaluation")
    y_pred = model.predict(X)
    
    print(f"\nAccuracy: {accuracy_score(y, y_pred):.4f}")
    print("\nClassification Report:")
    print(classification_report(y, y_pred, target_names=class_names))
    
    print("Confusion Matrix:")
    print(confusion_matrix(y, y_pred))
    print('='*40)

detailed_evaluation(svm_clf, X_val, y_val, "Validation", class_names)
# detailed_evaluation(svm_clf, X_test, y_test, "Test")

# --------------------------
# 7. 保存模型（完整保存）
# --------------------------
save_path = './model/svm_model.pkl'
print(f"\nSaving model to {save_path}...")
joblib.dump({
    'model': svm_clf,
    'scaler': scaler,
    'classes': class_names,
    'img_size': IMG_SIZE,
    'grayscale': GRAYSCALE,
    'training_date': time.strftime("%Y-%m-%d %H:%M:%S")
}, save_path)
print("Model saved successfully!")