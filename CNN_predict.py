import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import os
import argparse

# 配置参数（需与训练时一致）
IMG_SIZE = 152
COLOR_MODE = 'RGB'
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class CNN(nn.Module):
    def __init__(self, num_classes):
        super(CNN, self).__init__()
        in_channels = 3 if COLOR_MODE == 'RGB' else 1
        
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((19, 19))
        )
        
        with torch.no_grad():
            dummy_input = torch.randn(1, in_channels, IMG_SIZE, IMG_SIZE)
            dummy_output = self.features(dummy_input)
            self._to_linear = dummy_output.size(1) * dummy_output.size(2) * dummy_output.size(3)
            
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(self._to_linear, 512),
            nn.ReLU(),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

def load_model(model_path, num_classes):
    """加载训练好的模型"""
    model = CNN(num_classes=num_classes)
    state_dict = torch.load(model_path, map_location=DEVICE)
    
    # 添加维度验证
    if model.classifier[4].out_features != state_dict['classifier.4.weight'].shape[0]:
        raise ValueError(f"模型类别数不匹配！当前定义：{num_classes}，保存的模型：{state_dict['classifier.4.weight'].shape[0]}")
    
    model.load_state_dict(state_dict)
    model.to(DEVICE)
    model.eval()
    return model

def preprocess_image(image_path):
    """图像预处理"""
    transform = transforms.Compose([
        transforms.Resize(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) 
        if COLOR_MODE == 'RGB' else 
        transforms.Normalize([0.5], [0.5])
    ])
    
    try:
        img = Image.open(image_path)
        if COLOR_MODE == 'RGB' and img.mode != 'RGB':
            img = img.convert('RGB')
        elif COLOR_MODE == 'L' and img.mode != 'L':
            img = img.convert('L')
        return transform(img).unsqueeze(0)
    except Exception as e:
        print(f"Error processing {image_path}: {str(e)}")
        return None

def predict(image_path, model, class_names):
    """单张图像预测"""
    tensor = preprocess_image(image_path)
    if tensor is None:
        return None
    
    with torch.no_grad():
        outputs = model(tensor.to(DEVICE))
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        conf, preds = torch.max(probabilities, 1)
    
    return {
        'filename': os.path.basename(image_path),
        'prediction': class_names[preds.item()],
        'confidence': conf.item()
    }

def batch_predict(folder_path, model, class_names):
    """批量预测"""
    results = []
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            filepath = os.path.join(folder_path, filename)
            result = predict(filepath, model, class_names)
            if result:
                results.append(result)
    return results

def get_class_names(train_dir='./data/train'):
    classes = sorted([d for d in os.listdir(train_dir) 
                     if os.path.isdir(os.path.join(train_dir, d))])
    print(f"自动检测到{len(classes)}个类别：{classes}")
    return classes

if __name__ == "__main__":
    CLASS_NAMES = get_class_names()  # 自动获取
    MODEL_PATH = "./model/CNN_best_model.pth"
    
    # 命令行参数解析
    parser = argparse.ArgumentParser(description='Image Classification Predictor')
    parser.add_argument('input', type=str, help='Image file or directory path')
    parser.add_argument('--model', type=str, default=MODEL_PATH, help='Model path')
    parser.add_argument('--classes', nargs='+', default=CLASS_NAMES, help='List of class names')
    args = parser.parse_args()

    # 初始化模型
    model = load_model(args.model, len(args.classes))
    print(f"Model loaded from {args.model}")
    print(f"Using device: {DEVICE}")

    # 执行预测
    if os.path.isfile(args.input):
        result = predict(args.input, model, args.classes)
        if result:
            print("\nPrediction Result:")
            print(f"Image: {result['filename']}")
            print(f"Class: {result['prediction']}")
            print(f"Confidence: {result['confidence']:.4f}")
    elif os.path.isdir(args.input):
        results = batch_predict(args.input, model, args.classes)
        print("\nBatch Prediction Results:")
        for res in results:
            print(f"{res['filename']}: {res['prediction']} ({res['confidence']:.4f})")
    else:
        print("Invalid input path. Please provide a valid image file or directory.") 