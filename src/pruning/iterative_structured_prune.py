import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms, models
import torch.nn.utils.prune as prune
import math

# ====================== 配置区 ======================
dataset_root     = r"D:\archive (1)"
num_classes      = 36
batch_size       = 8
val_split_ratio  = 0.2
patience         = 3

# ========== 核心剪枝配置：改这里切换30%/50% ==========
prune_total           = 0.5    # 总剪枝比例：0.3=剪30%，0.5=剪50%
n_iterations          = 2      # 分几次剪完：2次
finetune_epochs_each  = 5      # 每次剪完微调几轮：5轮 → 总微调10轮
finetune_lr           = 1e-5   # 微调学习率：小步长恢复精度
weight_decay          = 3e-4

origin_model_path = "../../models/baseline/best_mobilenetv2_opt.pth"
pruned_save_path  = f"./iter_structured_pruned_{int(prune_total*100)}.pth"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")
print(f"迭代式结构化剪枝：总比例{prune_total*100:.0f}%，分{n_iterations}次，每次微调{finetune_epochs_each}轮")

# ====================== 数据集工具类 ======================
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

# ====================== 模型构建 ======================
def build_model():
    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(in_features, num_classes)
    )
    return model

def validate(model):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            out = model(imgs)
            preds = torch.argmax(out, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return correct / total

# ====================== 加载初始模型 ======================
model = build_model().to(device)
model.load_state_dict(torch.load(origin_model_path, map_location=device))
model.eval()
base_acc = validate(model)
print(f"\n原始模型基线精度: {base_acc:.4f}")

# 计算每次迭代的剪枝比例（相对于当前剩余通道）
prune_per_iter = 1 - math.pow(1 - prune_total, 1 / n_iterations)
print(f"每次迭代剪枝比例（相对当前）: {prune_per_iter:.4f}")

criterion = nn.CrossEntropyLoss()

# ====================== 迭代剪枝主循环 ======================
for iter_idx in range(1, n_iterations + 1):
    print(f"\n{'='*50}")
    print(f"第 {iter_idx}/{n_iterations} 次剪枝迭代")
    print('='*50)

    # 1. 结构化剪枝：只剪1x1 pointwise卷积，跳过depthwise核心层
    pruned_count = 0
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            # 只对 1×1、单分组的pointwise卷积剪枝，保护深度可分离卷积核心特征
            if module.kernel_size == (1, 1) and module.groups == 1:
                prune.ln_structured(
                    module,
                    name="weight",
                    amount=prune_per_iter,
                    n=1,
                    dim=0
                )
                pruned_count += 1

    # 永久固化本次剪枝结果，通道真实删除
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            try:
                prune.remove(module, "weight")
            except:
                pass

    print(f"完成第{iter_idx}次剪枝，共裁剪 {pruned_count} 个卷积层")
    acc_after_prune = validate(model)
    print(f"剪枝后（未微调）精度: {acc_after_prune:.4f}")

    # 2. 微调恢复精度
    optimizer = optim.Adam(model.parameters(), lr=finetune_lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=finetune_epochs_each)

    best_iter_acc = 0.0
    early_stop_cnt = 0

    for epoch in range(1, finetune_epochs_each + 1):
        model.train()
        train_loss_sum = 0.0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            output = model(imgs)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
            train_loss_sum += loss.item()

        avg_loss = train_loss_sum / len(train_loader)
        val_acc = validate(model)
        scheduler.step()

        print(f"  微调 Epoch [{epoch}/{finetune_epochs_each}] | Loss: {avg_loss:.4f} | Val Acc: {val_acc:.4f}")

        if val_acc > best_iter_acc:
            best_iter_acc = val_acc
            early_stop_cnt = 0
        else:
            early_stop_cnt += 1
            if early_stop_cnt >= patience:
                print(f"  早停触发，本次微调结束")
                break

    print(f"第{iter_idx}次迭代完成，微调后最佳精度: {best_iter_acc:.4f}")

# ====================== 保存最终模型 ======================
torch.save(model, "../../models/pruning/iter_structured_pruned_50.pth")
final_acc = validate(model)

print("\n" + "="*55)
print(f"✅ 迭代式结构化剪枝全部完成")
print(f"总剪枝比例: {prune_total*100:.0f}%")
print(f"原始基线精度: {base_acc:.4f}")
print(f"最终剪枝+微调后精度: {final_acc:.4f}")
print(f"精度损失: {(base_acc - final_acc)*100:.2f} 个百分点")
print(f"最终模型已保存: {pruned_save_path}")
print("="*55)
