import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms, models
from torch.optim.lr_scheduler import CosineAnnealingLR

# ====================== 配置区（只改这里就行） ======================
dataset_root     = r"D:\archive (1)"  # 数据集根目录
num_classes      = 36                 # 清洗后的类别数
batch_size       = 8                  # 显存不足就改成4
num_epochs       = 30                 # 最大训练轮数，早停会自动提前结束
lr               = 1e-4               # 保持你原有的学习率
weight_decay     = 2e-4               # 新增：L2正则权重衰减
val_split_ratio  = 0.2                # 20%数据作为验证集
patience         = 6                  # 连续5轮精度不涨就早停
save_path        = "../../models/baseline/best_mobilenetv2_opt.pth"  # 最优模型保存名

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("当前训练设备：", device)


# ====================== 自定义数据集：拆分后分别应用不同增强 ======================
class SplitDataset:
    def __init__(self, full_dataset, indices, transform=None):
        self.full_dataset = full_dataset
        self.indices = indices
        self.transform = transform

    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        img, label = self.full_dataset[real_idx]
        if self.transform is not None:
            img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.indices)


# ====================== 1. 数据增强（优化项1） ======================
# 训练集：温和数据增强，扩充样本多样性，缓解过拟合
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),      # 随机左右翻转
    transforms.RandomRotation(degrees=10),       # 小角度随机旋转
    transforms.ColorJitter(brightness=0.15, contrast=0.15),  # 亮度对比度微调
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 验证集：只做标准化，绝对不加随机增强
val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 先加载完整数据集（不提前加增强）
full_dataset = datasets.ImageFolder(root=dataset_root, transform=None)

# 按比例拆分索引，固定随机种子保证每次拆分一致
val_size    = int(len(full_dataset) * val_split_ratio)
train_size  = len(full_dataset) - val_size
train_indices, val_indices = random_split(
    range(len(full_dataset)),
    [train_size, val_size],
    generator=torch.Generator().manual_seed(42)
)

# 给训练集、验证集分别绑定各自的transform
train_dataset = SplitDataset(full_dataset, train_indices, transform=train_transform)
val_dataset   = SplitDataset(full_dataset, val_indices,   transform=val_transform)

# Windows系统 num_workers 必须设为0，避免报错
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,  num_workers=0)
val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False, num_workers=0)

print(f"总样本数：{len(full_dataset)} | 训练集：{train_size} | 验证集：{val_size}")


# ====================== 2. 模型定义（优化项2：分类头加Dropout） ======================
# 加载预训练MobileNetV2，主干网络完全保留预训练能力
model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
# 【兼容旧版PyTorch】如果上面一行报错，注释掉，改用下面这句：
# model = models.mobilenet_v2(pretrained=True)

# 仅替换分类头：加入Dropout，精准压制过拟合，不削弱主干特征能力
in_features = model.classifier[1].in_features
model.classifier = nn.Sequential(
    nn.Dropout(p=0.3),  # 0.25安全区间，只压过拟合不会欠拟合
    nn.Linear(in_features, num_classes)
)

model = model.to(device)


# ====================== 3. 优化器与学习率（优化项3：权重衰减） ======================
criterion = nn.CrossEntropyLoss()

# 保持学习率不变，新增权重衰减正则
optimizer = optim.Adam(
    model.parameters(),
    lr=lr,
    weight_decay=weight_decay
)

# 优化项4：余弦退火学习率调度，越往后步长越小，收敛更平稳
scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)


# ====================== 4. 早停初始化（优化项4：自动停在最优泛化点） ======================
best_val_acc     = 0.0
early_stop_count = 0


# ====================== 开始训练 ======================
for epoch in range(1, num_epochs + 1):
    # ---------- 训练阶段 ----------
    model.train()
    train_loss_sum = 0.0
    train_correct  = 0
    train_total    = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss    = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss_sum += loss.item() * images.size(0)
        _, predicted = torch.max(outputs.data, 1)
        train_total  += labels.size(0)
        train_correct += (predicted == labels).sum().item()

    avg_train_loss = train_loss_sum / train_total
    train_acc      = train_correct / train_total

    # ---------- 验证阶段 ----------
    model.eval()
    val_loss_sum = 0.0
    val_correct  = 0
    val_total    = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss    = criterion(outputs, labels)

            val_loss_sum += loss.item() * images.size(0)
            _, predicted = torch.max(outputs.data, 1)
            val_total    += labels.size(0)
            val_correct  += (predicted == labels).sum().item()

    avg_val_loss = val_loss_sum / val_total
    val_acc      = val_correct / val_total

    # 更新学习率
    scheduler.step()

    # ---------- 打印本轮结果 ----------
    print(f"Epoch [{epoch:2d}/{num_epochs}] | "
          f"Train Loss: {avg_train_loss:.4f} | Train Acc: {train_acc:.4f} | "
          f"Val Loss: {avg_val_loss:.4f} | Val Acc: {val_acc:.4f}")

    # ---------- 早停 + 保存最优模型 ----------
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        early_stop_count = 0
        torch.save(model.state_dict(), save_path)
        print(f"✅ 验证精度提升，保存最优模型，当前最高精度：{best_val_acc:.4f}")
    else:
        early_stop_count += 1
        if early_stop_count >= patience:
            print(f"\n⏹️  早停触发！连续{patience}轮验证精度无提升，训练提前结束")
            print(f"🏆 最终最优验证准确率：{best_val_acc:.4f}")
            break

# 训练结束
print("\n" + "="*55)
print(f"训练全部完成！最优验证准确率：{best_val_acc:.4f}")
print(f"最优模型已保存为：{save_path}，后续剪枝就用这个权重")
