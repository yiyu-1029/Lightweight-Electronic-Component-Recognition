import random
import torch
import matplotlib.pyplot as plt
from torchvision import datasets, transforms, models
from PIL import Image

# ========== 配置区 路径和count_classes、训练代码完全一致 ==========
dataset_root = r"D:\archive (1)"
num_classes = 36
model_path = "./best_mobilenetv2.pth"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
sample_num = 6

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("当前使用的设备：", device)  # 新增这行

# 加载数据集
dataset = datasets.ImageFolder(root=dataset_root, transform=transform)
# ========== 调试打印 运行后先看这里 ==========
print("当前读取到的类别列表：")
for name in dataset.classes:
    print("-", name)
print(f"\n类别总数：{len(dataset.classes)}")
# ==========================================

idx2class = {v: k for k, v in dataset.class_to_idx.items()}

# 加载模型
model = models.mobilenet_v2()
in_features = model.classifier[1].in_features
model.classifier[1] = torch.nn.Linear(in_features, num_classes)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()
model.to(device)

samples = random.sample(list(dataset.imgs), sample_num)

plt.figure(figsize=(12, 7))
for i, (img_path, true_label) in enumerate(samples):
    raw_img = Image.open(img_path).convert("RGB")
    img_tensor = transform(raw_img).unsqueeze(0).to(device)

    with torch.no_grad():
        out = model(img_tensor)
        pred_idx = torch.argmax(out, dim=1).item()
        pred_cls = idx2class[pred_idx]
        true_cls = idx2class[true_label]

    plt.subplot(2, sample_num//2, i+1)
    plt.imshow(raw_img)
    color = "green" if pred_cls == true_cls else "red"
    plt.title(f"Pred: {pred_cls}\nTrue: {true_cls}", color=color, fontsize=9)
    plt.axis("off")

plt.tight_layout()
plt.savefig("../../results/inference/03_baseline_inference_result.png", dpi=300)
plt.show()
