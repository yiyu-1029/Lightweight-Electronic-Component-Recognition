import torch
import torch.nn as nn
from torchvision import transforms, datasets
from torchvision.models import mobilenet_v2
from PIL import Image
import matplotlib.pyplot as plt

# ============================================================
# 配置区
# ============================================================
TEST_IMG_PATH = r"/val_sample_export/microchip_val_test.jpg"
DATA_ROOT = r"D:\archive (1)"

IMG_SIZE = 224
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
DEVICE = torch.device("cpu")

# 待对比模型列表（已替换为你现有的动态量化模型）
MODEL_LIST = [
    {"name": "原始基线模型", "path": "best_mobilenetv2_opt.pth", "type": "baseline"},
    {"name": "迭代式剪枝30%", "path": "iter_structured_pruned_30.pth", "type": "full"},
    {"name": "迭代式剪枝50%", "path": "iter_structured_pruned_50.pth", "type": "full"},
    {"name": "剪枝30% + INT8动态量化", "path": "quantized_iter_30.pth", "type": "full"}
]

# ============================================================
# 类别名兜底（杜绝索引越界）
# ============================================================
try:
    full_dataset = datasets.ImageFolder(root=DATA_ROOT)
    CLASS_NAMES = full_dataset.classes
except:
    CLASS_NAMES = []

def get_safe_class_name(idx):
    if 0 <= idx < len(CLASS_NAMES):
        return CLASS_NAMES[idx]
    else:
        return f"类别{idx+1}"

# ============================================================
# 推理函数
# ============================================================
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=MEAN, std=STD)
])

def predict(model, img_tensor):
    model.eval()
    with torch.no_grad():
        output = model(img_tensor.unsqueeze(0).to(DEVICE))
        prob = torch.nn.functional.softmax(output, dim=1)
        confidence, pred_idx = torch.max(prob, 1)
        idx = pred_idx.item()
        class_name = get_safe_class_name(idx)
    return class_name, confidence.item() * 100

def load_model(model_info):
    path = model_info["path"]
    m_type = model_info["type"]

    if m_type == "baseline":
        state_dict = torch.load(path, map_location=DEVICE)
        num_classes = state_dict["classifier.1.weight"].shape[0]
        model = mobilenet_v2(weights=None)
        model.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(model.classifier[1].in_features, num_classes)
        )
        model.load_state_dict(state_dict)
    else:
        # 剪枝模型、动态量化模型都用完整模型加载方式
        model = torch.load(path, map_location=DEVICE, weights_only=False)

    model = model.to(DEVICE)
    model.eval()
    return model

# ============================================================
# 主流程
# ============================================================
if __name__ == "__main__":
    plt.rcParams["font.sans-serif"] = ["SimHei"]
    plt.rcParams["axes.unicode_minus"] = False

    img = Image.open(TEST_IMG_PATH).convert("RGB")
    img_tensor = transform(img)

    results = []
    for m_info in MODEL_LIST:
        print(f"正在推理：{m_info['name']}")
        model = load_model(m_info)
        pred_class, conf = predict(model, img_tensor)
        results.append({
            "name": m_info["name"],
            "class": pred_class,
            "confidence": conf
        })
        print(f"  预测：{pred_class} | 置信度：{conf:.2f}%\n")
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # 绘图：1张原图 + 4个模型结果
    fig, axes = plt.subplots(1, 5, figsize=(20, 5), dpi=150)

    axes[0].imshow(img)
    axes[0].set_title("测试原图", fontsize=11, fontweight="bold")
    axes[0].axis("off")

    for i, res in enumerate(results):
        ax = axes[i+1]
        ax.imshow(img)
        ax.set_title(res["name"], fontsize=10, fontweight="bold")
        ax.axis("off")
        text = f"预测类别：{res['class']}\n置信度：{res['confidence']:.2f}%"
        ax.text(
            0.5, -0.18, text,
            transform=ax.transAxes, ha="center", va="top",
            fontsize=9, bbox=dict(boxstyle="round", facecolor="white", alpha=0.92)
        )

    plt.suptitle("基线 / 剪枝 / INT8量化 推理效果对比", fontsize=14, fontweight="bold", y=0.98)
    plt.tight_layout()
    plt.savefig("../../results/inference/06_inference_comparison.png", bbox_inches="tight", dpi=200)
    print("✅ 推理对比图已保存：06_inference_comparison.png")
    plt.show()
