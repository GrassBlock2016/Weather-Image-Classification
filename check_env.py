import torch

print(f"PyTorch版本: {torch.__version__}")
print(f"CUDA可用: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"设备数量: {torch.cuda.device_count()}")
    
    for i in range(torch.cuda.device_count()):
        print(f"设备 {i}: {torch.cuda.get_device_name(i)}")
        print(f"计算能力: {torch.cuda.get_device_capability(i)}")
        print(f"显存总量: {torch.cuda.get_device_properties(i).total_memory/1024**3:.2f} GB")
        
    print(f"当前CUDA设备: {torch.cuda.current_device()}")
    print(f"CUDA版本: {torch.version.cuda}")
    print(f"cuDNN版本: {torch.backends.cudnn.version()}") 