import os
import time
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import transforms, datasets
from torchvision.models import mobilenet_v2

# ============================================================
# ====================== 顶部配置区（可修改） ==================
# ============================================================

# 1. 数据集路径（和训练时保持完全一致）
DATA_ROOT = r"D:\archive (1)"  # 内部是各类别文件夹
VAL_RATIO = 0.2  # 验证集比例，和训练时一致
RANDOM_SEED = 42  # 随机种子，必须和训练时相同，保证验证集一致

# 2. 图片预处理（和训练时验证集transform完全一致）
IMG_SIZE = 224
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
BATCH_SIZE = 32
NUM_WORKERS = 0  # Windows环境建议设为0，避免多进程报错

# 3. 待评估模型列表（名称 + 权重文件路径）
# 注意：结构化剪枝模型若为物理删通道，必须保存完整模型对象；基线/非结构化可用state_dict
MODEL_LIST = [
    {
        "name": "原始基线模型",
        "path": "best_mobilenetv2_opt.pth",
        "load_type": "state_dict"  # state_dict: 仅保存权重；full: 保存完整模型
    },
    {
        "name": "非结构化剪枝30%",
        "path": "pruned_best_model.pth",
        "load_type": "state_dict"
    },
    {
        "name": "一次性结构化剪枝30%",
        "path": "structured_pruned_30.pth",
        "load_type": "full"  # 结构化剪枝删了通道，结构改变，建议保存完整模型
    },
    {
        "name": "迭代式结构化剪枝30%",
        "path": "iter_structured_pruned_30.pth",
        "load_type": "full"
    },
    {
    "name": "迭代式结构化剪枝50%",
    "path": "iter_structured_pruned_50.pth",
    "load_type": "full"
    },




]

# 4. 输出文件配置
OUTPUT_MD = "剪枝模型评估对比表.md"
OUTPUT_PLOT = "剪枝模型评估对比图.png"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 5. 推理速度测试配置
INFER_TEST_TIMES = 100  # 推理测试次数，取平均值
INFER_WARMUP_TIMES = 20  # 预热次数，消除GPU启动误差


# ============================================================
# ====================== 工具函数区 ===========================
# ============================================================

def set_seed(seed):
    """固定所有随机种子，保证验证集划分和训练完全一致"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def count_params(model):
    """统计模型总参数量 和 非零参数量（核心指标，体现结构化剪枝收益）"""
    total_params = 0
    nonzero_params = 0
    for param in model.parameters():
        total_params += param.numel()
        nonzero_params += torch.count_nonzero(param).item()
    total_m = total_params / 1e6  # 转成百万单位M
    nonzero_m = nonzero_params / 1e6
    return total_m, nonzero_m


def get_model_size_mb(file_path):
    """获取模型文件物理体积（MB）"""
    size_bytes = os.path.getsize(file_path)
    size_mb = size_bytes / (1024 * 1024)
    return size_mb


def evaluate_accuracy(model, val_loader):
    """评估模型在验证集上的准确率"""
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    return accuracy


def measure_inference_time(model, input_shape=(1, 3, 224, 224)):
    """测试单张图片平均推理耗时（毫秒ms），含GPU预热+同步"""
    model.eval()
    dummy_input = torch.randn(input_shape).to(DEVICE)

    # 预热，消除GPU启动延迟
    with torch.no_grad():
        for _ in range(INFER_WARMUP_TIMES):
            _ = model(dummy_input)

    if DEVICE.type == "cuda":
        torch.cuda.synchronize()

    # 正式计时
    start_time = time.time()
    with torch.no_grad():
        for _ in range(INFER_TEST_TIMES):
            _ = model(dummy_input)

    if DEVICE.type == "cuda":
        torch.cuda.synchronize()
    end_time = time.time()

    avg_time_ms = (end_time - start_time) / INFER_TEST_TIMES * 1000
    return avg_time_ms


def load_model(model_info, num_classes):
    """
    加载模型，兼容两种保存格式：
    - state_dict: 仅保存权重，需重建MobileNetV2结构（基线、非结构化剪枝用）
    - full: 保存完整模型对象（结构化剪枝后结构改变，用此方式）
    """
    path = model_info["path"]
    load_type = model_info["load_type"]

    if load_type == "state_dict":
        # 重建基线模型结构，和训练时完全一致
        model = mobilenet_v2(weights=None)
        # 替换分类头（和训练时一致：Dropout + Linear）
        model.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(model.classifier[1].in_features, num_classes)
        )
        # 加载权重
        state_dict = torch.load(path, map_location=DEVICE)
        model.load_state_dict(state_dict)
    elif load_type == "full":
        # 直接加载完整模型（结构化剪枝后结构已修改）
        model = torch.load(path, map_location=DEVICE,weights_only=False)
    else:
        raise ValueError(f"不支持的加载类型: {load_type}")

    model = model.to(DEVICE)
    model.eval()
    return model


# ============================================================
# ====================== 主执行流程 ===========================
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("开始批量评估模型...")
    print(f"使用设备: {DEVICE}")
    print("=" * 60)

    # 1. 固定种子，加载数据集，划分验证集
    set_seed(RANDOM_SEED)

    # 验证集预处理（无增强，仅Resize+归一化）
    val_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD)
    ])

    full_dataset = datasets.ImageFolder(root=DATA_ROOT, transform=val_transform)
    num_classes = len(full_dataset.classes)
    val_size = int(VAL_RATIO * len(full_dataset))
    train_size = len(full_dataset) - val_size

    # 和训练时完全一致的划分方式
    train_dataset, val_dataset = random_split(
        full_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(RANDOM_SEED)
    )

    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE,
        shuffle=False, num_workers=NUM_WORKERS
    )
    print(f"验证集加载完成，共 {val_size} 张图片，{num_classes} 个类别\n")

    # 2. 遍历所有模型，逐个评估
    results = []
    for idx, model_info in enumerate(MODEL_LIST):
        model_name = model_info["name"]
        print(f"[{idx + 1}/{len(MODEL_LIST)}] 正在评估: {model_name}")

        try:
            # 加载模型
            model = load_model(model_info, num_classes)

            # 统计参数量
            total_params_m, nonzero_params_m = count_params(model)

            # 获取模型文件体积
            model_size_mb = get_model_size_mb(model_info["path"])

            # 评估准确率
            acc = evaluate_accuracy(model, val_loader)

            # 测试推理速度
            infer_time_ms = measure_inference_time(model)

            # 保存结果
            results.append({
                "实验方案": model_name,
                "验证准确率(%)": round(acc, 2),
                "总参数量(M)": round(total_params_m, 2),
                "非零参数量(M)": round(nonzero_params_m, 2),
                "模型体积(MB)": round(model_size_mb, 2),
                "单张推理耗时(ms)": round(infer_time_ms, 2)
            })

            print(f"  准确率: {acc:.2f}% | 非零参数量: {nonzero_params_m:.2f}M | 推理耗时: {infer_time_ms:.2f}ms\n")

            # 释放显存
            del model
            torch.cuda.empty_cache()

        except Exception as e:
            print(f"  ❌ 评估失败: {str(e)}\n")
            continue

    # 3. 生成DataFrame，输出Markdown表格
    if not results:
        print("没有成功评估的模型，程序退出")
        exit()

    df = pd.DataFrame(results)

    # 打印到控制台
    print("=" * 60)
    print("评估结果汇总：")
    print(df.to_string(index=False))
    print("=" * 60)

    # 保存Markdown文件
    md_table = df.to_markdown(index=False, tablefmt="pipe")
    with open(OUTPUT_MD, "w", encoding="utf-8") as f:
        f.write("# 剪枝模型评估对比表\n\n")
        f.write(f"测试环境：{DEVICE.type.upper()} | 验证集大小：{val_size}张\n\n")
        f.write(md_table)
        f.write("\n\n> 注：非零参数量为结构化剪枝核心收益指标，模型体积为权重文件物理大小。")
    print(f"\n✅ Markdown对比表已保存至: {OUTPUT_MD}")

    # 4. 生成可视化对比柱状图
    # 设置中文显示
    plt.rcParams["font.sans-serif"] = ["SimHei"]  # Windows默认黑体
    plt.rcParams["axes.unicode_minus"] = False

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    model_names = [r["实验方案"] for r in results]

    # 子图1：验证准确率
    acc_values = [r["验证准确率(%)"] for r in results]
    axes[0].bar(model_names, acc_values, color=["#4e79a7", "#f28e2b", "#e15759", "#76b7b2"])
    axes[0].set_title("验证准确率对比 (%)", fontsize=13, fontweight="bold")
    axes[0].set_ylabel("准确率 (%)")
    axes[0].set_ylim(min(acc_values) - 2, max(acc_values) + 1)
    for i, v in enumerate(acc_values):
        axes[0].text(i, v + 0.1, f"{v}%", ha="center", fontsize=10)
    plt.setp(axes[0].get_xticklabels(), rotation=15, ha="right")

    # 子图2：非零参数量
    nonzero_values = [r["非零参数量(M)"] for r in results]
    axes[1].bar(model_names, nonzero_values, color=["#4e79a7", "#f28e2b", "#e15759", "#76b7b2"])
    axes[1].set_title("非零参数量对比 (M)", fontsize=13, fontweight="bold")
    axes[1].set_ylabel("参数量 (百万)")
    for i, v in enumerate(nonzero_values):
        axes[1].text(i, v + 0.1, f"{v}M", ha="center", fontsize=10)
    plt.setp(axes[1].get_xticklabels(), rotation=15, ha="right")

    # 子图3：单张推理耗时
    time_values = [r["单张推理耗时(ms)"] for r in results]
    axes[2].bar(model_names, time_values, color=["#4e79a7", "#f28e2b", "#e15759", "#76b7b2"])
    axes[2].set_title("单张推理耗时对比 (ms)", fontsize=13, fontweight="bold")
    axes[2].set_ylabel("耗时 (毫秒)")
    for i, v in enumerate(time_values):
        axes[2].text(i, v + 0.1, f"{v}ms", ha="center", fontsize=10)
    plt.setp(axes[2].get_xticklabels(), rotation=15, ha="right")

    plt.tight_layout()
    plt.savefig(OUTPUT_PLOT, dpi=300, bbox_inches="tight")
    print(f"✅ 可视化对比图已保存至: {OUTPUT_PLOT}")
    print("\n全部评估完成！")
