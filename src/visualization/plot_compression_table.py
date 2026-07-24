import os
import time
import torch
import torch.nn as nn
from torchvision import transforms, datasets
from torchvision.models import mobilenet_v2
import matplotlib.pyplot as plt
import numpy as np

# ====================== 全局配置（和你项目统一） ======================
DATA_ROOT = r"D:\archive (1)"
DEVICE = torch.device("cpu")  # 量化必须CPU测速，保证公平对比
IMG_SIZE = 224
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

# 四个模型路径
MODEL_INFO_LIST = [
    {
        "name": "原始基线模型",
        "file_path": "best_mobilenetv2_opt.pth",
        "net_type": "baseline"
    },
    {
        "name": "迭代式剪枝30%",
        "file_path": "iter_structured_pruned_30.pth",
        "net_type": "full"
    },
    {
        "name": "迭代式剪枝50%",
        "file_path": "iter_structured_pruned_50.pth",
        "net_type": "full"
    },
    {
        "name": "剪枝30% + INT8动态量化",
        "file_path": "quantized_iter_30.pth",
        "net_type": "quant"
    }
]

# 测速参数
TEST_IMG_COUNT = 50  # 用50张图测平均推理延迟，降低波动
WARM_UP_COUNT = 10   # 预热消除CPU启动波动

# ====================== 工具函数 ======================
# 1. 加载模型
def load_model(model_cfg):
    path = model_cfg["file_path"]
    m_type = model_cfg["net_type"]
    if m_type == "baseline":
        state_dict = torch.load(path, map_location=DEVICE)
        num_cls = state_dict["classifier.1.weight"].shape[0]
        model = mobilenet_v2(weights=None)
        model.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(model.classifier[1].in_features, num_cls)
        )
        model.load_state_dict(state_dict)
    else:
        model = torch.load(path, map_location=DEVICE, weights_only=False)
    model = model.to(DEVICE)
    model.eval()
    return model

# 2. 计算模型文件大小 MB
def get_model_file_mb(file_path):
    size_byte = os.path.getsize(file_path)
    return round(size_byte / 1024 / 1024, 2)

# 3. 统计模型总参数量
def count_total_params(model):
    total = sum(p.numel() for p in model.parameters())
    return total

# 4. CPU单图推理平均耗时(ms)
transform_test = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD)
])
def get_avg_infer_latency(model, sample_img):
    img_tensor = transform_test(sample_img).unsqueeze(0).to(DEVICE)
    # 预热
    for _ in range(WARM_UP_COUNT):
        with torch.no_grad():
            _ = model(img_tensor)
    # 正式计时
    start = time.perf_counter()
    for _ in range(TEST_IMG_COUNT):
        with torch.no_grad():
            _ = model(img_tensor)
    total_ms = (time.perf_counter() - start) * 1000
    avg_ms = total_ms / TEST_IMG_COUNT
    return round(avg_ms, 3)

# 5. 简易验证集Top1准确率（可选，真实数据集可替换完整val acc）
def calc_val_top1_acc(model, val_dataset, sample_limit=800):
    correct = 0
    total = min(sample_limit, len(val_dataset))
    model.eval()
    with torch.no_grad():
        for i in range(total):
            img, label = val_dataset[i]
            tensor = transform_test(img).unsqueeze(0).to(DEVICE)
            out = model(tensor)
            pred = torch.argmax(out, dim=1).item()
            if pred == label:
                correct += 1
    acc = correct / total * 100
    return round(acc, 2)

# ====================== 批量计算全部指标 ======================
if __name__ == "__main__":
    plt.rcParams["font.sans-serif"] = ["SimHei"]
    plt.rcParams["axes.unicode_minus"] = False

    # 加载验证集，用于测速与准确率
    full_ds = datasets.ImageFolder(DATA_ROOT)
    val_ds, _ = torch.utils.data.random_split(full_ds, [int(0.2*len(full_ds)), int(0.8*len(full_ds))],
                                             generator=torch.Generator().manual_seed(42))
    sample_test_img = val_ds[0][0]  # 随便取一张图用来测速

    table_rows = []
    baseline_file_size = 0.0
    baseline_latency = 0.0
    baseline_params = 0

    print("开始批量计算各模型全部性能指标...\n")
    for cfg in MODEL_INFO_LIST:
        name = cfg["name"]
        print(f"正在处理：{name}")
        # 加载模型
        net = load_model(cfg)
        # 1 文件体积 MB
        file_mb = get_model_file_mb(cfg["file_path"])
        # 2 参数量
        param_num = count_total_params(net)
        # 3 平均推理延迟 ms
        avg_lat = get_avg_infer_latency(net, sample_test_img)
        # 4 验证集Top1准确率
        top1_acc = calc_val_top1_acc(net, val_ds)

        # 基线保存基准值，用于计算压缩率、加速比
        if "原始基线" in name:
            baseline_file_size = file_mb
            baseline_latency = avg_lat
            baseline_params = param_num
            compress_ratio = 100.0
            speed_up = 1.0
        else:
            # 体积压缩率 = 当前体积/基线体积 *100%
            compress_ratio = round((file_mb / baseline_file_size) * 100, 2)
            # 推理加速比 = 基线延迟 / 当前延迟
            speed_up = round(baseline_latency / avg_lat, 2)

        # 格式化参数量显示
        param_str = f"{param_num / 1e6:.2f} M"
        table_rows.append([
            name,
            param_str,
            f"{file_mb} MB",
            f"{compress_ratio} %",
            f"{avg_lat} ms",
            f"{speed_up} x",
            f"{top1_acc} %"
        ])
        print(f"{name} 计算完成：体积{file_mb}MB，单图延迟{avg_lat}ms，准确率{top1_acc}%\n")
        del net
        torch.cuda.empty_cache()

    # ====================== 绘制性能对比表格图 ======================
    col_labels = [
        "模型名称",
        "总参数量",
        "模型文件体积",
        "体积压缩率",
        "CPU单图平均延迟",
        "推理加速倍数",
        "验证集Top1准确率"
    ]
    fig, ax = plt.subplots(figsize=(16, 4), dpi=150)
    ax.axis("tight")
    ax.axis("off")
    table = ax.table(
        cellText=table_rows,
        colLabels=col_labels,
        loc="center",
        cellLoc="center"
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.8)
    fig.suptitle("各轻量化模型性能指标完整对比表（CPU环境统一测试）", fontsize=13, weight="bold")
    plt.tight_layout()
    plt.savefig("../../results/pruning/model_compression_performance_table.png", bbox_inches="tight", dpi=200)
    print("✅ 性能指标表格图已保存 model_compression_performance_table.png")
    plt.show()
