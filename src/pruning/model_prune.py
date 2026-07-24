import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms, models
import torch.nn.utils.prune as prune

# ====================== 配置区（和你训练代码保持一致）======================
dataset_root     = r"D:\archive (1)"
num_classes      = 36
batch_size       = 8
lr               = 1e-4
weight_decay     = 2e-4
val_split_ratio  = 0.2
patience         = 5
# 剪枝配置
prune_amount     = 0.3   # 剪枝30%权重，根据实验结果调整剪枝比例
finetune_epochs  = 8     # 剪枝后微调轮数
model_path       = "../../models/baseline/best_mobilenetv2_opt.pth"
pruned_save_path = "../../models/pruning/pruned_best_model.pth"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

# ====================== 复用你之前的数据集拆分类 ======================
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

# 数据增强
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 加载并拆分数据集
full_dataset = datasets.ImageFolder(root=dataset_root, transform=None)
val_size = int(len(full_dataset) * val_split_ratio)
train_size = len(full_dataset) - val_size
train_indices, val_indices = random_split(
    range(len(full_dataset)),
    [train_size, val_size],
    generator=torch.Generator().manual_seed(42)
)
train_dataset = SplitDataset(full_dataset, train_indices, transform=train_transform)
val_dataset = SplitDataset(full_dataset, val_indices, transform=val_transform)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

# ====================== 加载原模型 ======================
def build_model():
    # 适配低版本torchvision
    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(in_features, num_classes)
    )
    return model


model = build_model().to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()
print("原始模型加载完成")

# ====================== 剪枝核心逻辑 ======================
# 对所有Conv2d、Linear层进行全局剪枝
parameters_to_prune = []
for name, module in model.named_modules():
    if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
        parameters_to_prune.append((module, "weight"))

# 全局非结构化剪枝
prune.global_unstructured(
    parameters_to_prune,
    pruning_method=prune.L1Unstructured,
    amount=prune_amount
)

# 移除剪枝参数的掩码，永久固化剪枝结果
for module, param_name in parameters_to_prune:
    prune.remove(module, param_name)

print(f"剪枝完成，全局移除 {prune_amount*100}% 冗余权重")

# ====================== 剪枝后微调恢复精度 ======================
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=lr*0.5, weight_decay=weight_decay)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=finetune_epochs)

best_acc = 0.0
early_stop_count = 0

def validate():
    model.eval()
    total_correct = 0
    total_samples = 0
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            out = model(imgs)
            preds = torch.argmax(out, dim=1)
            total_correct += (preds == labels).sum().item()
            total_samples += labels.size(0)
    return total_correct / total_samples

# 微调循环
for epoch in range(finetune_epochs):
    model.train()
    train_loss = 0.0
    for imgs, labels in train_loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        output = model(imgs)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    train_loss /= len(train_loader)
    val_acc = validate()
    scheduler.step()

    print(f"微调 Epoch [{epoch+1}/{finetune_epochs}] | Loss: {train_loss:.4f} | Val Acc: {val_acc:.4f}")

    if val_acc > best_acc:
        best_acc = val_acc
        torch.save(model.state_dict(), pruned_save_path)
        print(f"✅ 微调最优模型已保存，当前最高精度: {best_acc:.4f}")
        early_stop_count = 0
    else:
        early_stop_count += 1
        if early_stop_count >= patience:
            print("早停触发，微调结束")
            break

# ====================== 最终结果对比 ======================
print("-"*50)
print(f"剪枝前最佳精度：≈51.41%")
print(f"剪枝微调后最佳精度：{best_acc:.4f}")
print(f"剪枝后模型权重文件：{pruned_save_path}")
print("剪枝完成，可使用新模型运行推理脚本")
