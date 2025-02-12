import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import os
import time
import gc

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'  # 用于更精确的错误定位
os.environ['TORCH_CUDA_ARCH_LIST'] = '7.5'  # Tesla T4的计算能力是7.5

# 配置参数
IMG_SIZE = 152          # 统一图像尺寸
BATCH_SIZE = 16         # 批量大小
EPOCHS = 15             # 训练轮数
class_names = sorted([d for d in os.listdir('./data/train') 
                     if os.path.isdir(os.path.join('./data/train', d)) 
                     and not d.startswith('.')])
NUM_CLASSES = len(class_names)
COLOR_MODE = 'RGB'      # 'RGB' 或 'L' (灰度)
torch.backends.cudnn.enabled = False

# 修改设备检测逻辑
if torch.cuda.is_available():
    try:
        # 显式指定设备
        DEVICE = torch.device("cuda:0")
        # 执行显存预热
        torch.cuda.init()
        torch.cuda.empty_cache()
        _ = torch.zeros(1).to(DEVICE)
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA version: {torch.version.cuda}")
        print(f"PyTorch CUDA version: {torch.version.cuda}")
    except Exception as e:
        print(f"CUDA初始化失败: {str(e)}")
        DEVICE = torch.device("cpu")
else:
    DEVICE = torch.device("cpu")

# 在模型实例化前添加环境验证
print("="*40)
print("环境验证:")
print(f"PyTorch版本: {torch.__version__}")
print(f"CUDA可用: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"当前设备: {torch.cuda.current_device()}")
    print(f"设备名称: {torch.cuda.get_device_name(0)}")
    print(f"CUDA版本: {torch.version.cuda}")
    print(f"cuDNN版本: {torch.backends.cudnn.version()}")
print("="*40)

# --------------------------
# 1. 数据准备（使用PyTorch数据管道）
# --------------------------
# 数据增强和预处理
train_transform = transforms.Compose([
    transforms.Resize(IMG_SIZE),
    transforms.RandomRotation(20),
    transforms.RandomHorizontalFlip(),
    transforms.RandomAffine(0, shear=0.2, scale=(0.8, 1.2)),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
]) if COLOR_MODE == 'RGB' else transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize(IMG_SIZE),
    transforms.RandomRotation(20),
    transforms.RandomHorizontalFlip(),
    transforms.RandomAffine(0, shear=0.2, scale=(0.8, 1.2)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

val_test_transform = transforms.Compose([
    transforms.Resize(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
]) if COLOR_MODE == 'RGB' else transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

# 创建数据集
train_dataset = datasets.ImageFolder(
    './data/train',
    transform=train_transform
)
val_dataset = datasets.ImageFolder(
    './data/val',
    transform=val_test_transform
)
# test_dataset = datasets.ImageFolder(
#     './data/test',
#     transform=val_test_transform
# )

# 创建数据加载器
train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=4
)
val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=4
)
# test_loader = DataLoader(
#     test_dataset,
#     batch_size=BATCH_SIZE,
#     shuffle=False,
#     num_workers=4
# )

# 获取类别标签映射
class_names = train_dataset.classes

# --------------------------
# 2. 构建CNN模型
# --------------------------
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        in_channels = 3 if COLOR_MODE == 'RGB' else 1
        
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 152 → 76
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 76 → 38
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((19, 19))  # 确保输出为19x19
        )
        
        # 更精确的维度计算
        with torch.no_grad():
            dummy_input = torch.randn(1, in_channels, IMG_SIZE, IMG_SIZE)
            dummy_output = self.features(dummy_input)
            print(f"Debug - Feature map shape: {dummy_output.shape}")  # 添加调试输出
            self._to_linear = dummy_output.size(1) * dummy_output.size(2) * dummy_output.size(3)
            
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(self._to_linear, 512),
            nn.ReLU(),
            nn.Linear(512, NUM_CLASSES)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

model = CNN().to(DEVICE)

# 添加输入尺寸验证
print(f"\n验证输入尺寸:")
print(f"设定图像尺寸: {IMG_SIZE}x{IMG_SIZE}")
print(f"实际数据尺寸示例: {next(iter(train_loader))[0].shape[2:]}")

# 修改后的数据验证部分
for images, _ in train_loader:
    print(f"\n实际输入数据尺寸: {images.shape}")
    print(f"理论特征图尺寸应: (batch, 128, 19, 19)")
    
    try:
        with torch.no_grad():
            test_features = model.features(images.to(DEVICE))
            print(f"实际特征图尺寸: {test_features.shape}")
            
            test_output = model(images.to(DEVICE))
            print(f"模型输出尺寸: {test_output.shape}")
        break
    except Exception as e:
        print(f"错误详情: {str(e)}")
        raise

# --------------------------
# 3. 训练模型
# --------------------------
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters())

best_acc = 0.0
train_loss, train_acc = [], []
val_loss, val_acc = [], []

for epoch in range(EPOCHS):
    start_time = time.time()
    
    # 在训练循环开始前添加
    torch.cuda.empty_cache()
    gc.collect()
    
    # 训练阶段
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for images, labels in train_loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        # 在每个batch处理后添加
        if torch.cuda.is_available():
            torch.cuda.synchronize()  # 确保CUDA操作完成
    
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = correct / total
    train_loss.append(epoch_loss)
    train_acc.append(epoch_acc)
    
    # 验证阶段
    model.eval()
    val_running_loss = 0.0
    val_correct = 0
    val_total = 0
    
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            val_running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            val_total += labels.size(0)
            val_correct += (predicted == labels).sum().item()
    
    val_epoch_loss = val_running_loss / len(val_loader)
    val_epoch_acc = val_correct / val_total
    val_loss.append(val_epoch_loss)
    val_acc.append(val_epoch_acc)
    
    # 保存最佳模型
    if val_epoch_acc > best_acc:
        best_acc = val_epoch_acc
        torch.save(model.state_dict(), './model/CNN_best_model.pth')
    
    # 打印进度
    time_elapsed = time.time() - start_time
    print(f'Epoch {epoch+1}/{EPOCHS} | Time: {time_elapsed:.0f}s')
    print(f'Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
    print(f'Val Loss: {val_epoch_loss:.4f} Acc: {val_epoch_acc:.4f}\n')

    # 每个epoch后清理显存
    torch.cuda.empty_cache()
    # 监控显存使用
    print(f"显存使用情况: {torch.cuda.memory_allocated()/1024**2:.2f} MB / "
          f"{torch.cuda.get_device_properties(0).total_memory/1024**3:.2f} GB")

# --------------------------
# 4. 评估模型
# --------------------------
model.load_state_dict(torch.load('./model/CNN_best_model.pth'))
model.eval()

all_labels = []
all_preds = []

with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in val_loader:
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)
        
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(predicted.cpu().numpy())

val_acc = correct / total
print(f'\nValidation Accuracy: {val_acc:.4f}')

print("\nClassification Report:")
print(classification_report(all_labels, all_preds, target_names=class_names))

print("\nConfusion Matrix:")
print(confusion_matrix(all_labels, all_preds))

# --------------------------
# 5. 可视化训练过程
# --------------------------
def plot_training_history():
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_acc, label='Training Accuracy')
    plt.plot(val_acc, label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(train_loss, label='Training Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.legend()

    plt.tight_layout()
    plt.savefig('./images/CNN_train_history.png')

plot_training_history()