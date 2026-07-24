import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms, models
import torch.nn.utils.prune as prune

# ====================== 配置区（和训练完全一致）======================
dataset_root     = r"D:\archive (1)"
num_classes      = 36
batch_size       = 8
lr               = 1e-5
weight_decay     = 3e-4
val_split_ratio  = 0.2
patience         = 5

# 结构化剪枝配置：按通道比例裁剪
prune_amount     = 0.3   # 改成0.5就是剪枝50%
finetune_epochs  = 10     # 微调轮数
origin_model_path = "../../models/baseline/best_mobilenetv2_opt.pth"
pruned_save_path  = f"./structured_pruned_{int(prune_amount*100)}.pth"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

# ====================== 数据集工具类（和训练保持一致）======================
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
val_loader   = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

# ====================== 模型构建（和训练结构完全一致）======================
def build_model():
    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(in_features, num_classes)
    )
    return model

model = build_model().to(device)
model.load_state_dict(torch.load(origin_model_path, map_location=device))
model.eval()
print("原始模型加载完成")

# ====================== 核心：L1范数结构化通道剪枝 ======================
# 严格按教程：对卷积层做结构化剪枝，按输出通道维度裁剪
pruned_layers = 0
for name, module in model.named_modules():
    if isinstance(module, nn.Conv2d):
        # 跳过深度可分离卷积的depthwise层，避免破坏特征结构
        if module.groups == module.in_channels:
            continue
        # L1范数结构化剪枝，dim=0 表示按输出通道维度裁剪
        prune.ln_structured(
            module,
            name="weight",
            amount=prune_amount,
            n=1,       # L1范数
            dim=0      # 裁剪输出通道
        )
        pruned_layers += 1

# 永久移除被剪掉的通道权重（固化剪枝结果）
for name, module in model.named_modules():
    if isinstance(module, nn.Conv2d):
        try:
            prune.remove(module, "weight")
        except:
            pass

print(f"结构化剪枝完成，共处理 {pruned_layers} 个卷积层，每通道裁剪 {prune_amount*100}%")

# ====================== 剪枝后微调（严格按教程5个epoch）======================
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
        torch.save(model, "../../models/pruning/structured_pruned_30.pth")
        print(f"✅ 最优模型已保存，当前精度: {best_acc:.4f}")
        early_stop_count = 0
    else:
        early_stop_count += 1
        if early_stop_count >= patience:
            print("早停触发，微调结束")
            break

print("-"*50)
print(f"结构化剪枝{int(prune_amount*100)}% 微调后最佳精度: {best_acc:.4f}")
print(f"模型保存为: {pruned_save_path}")
