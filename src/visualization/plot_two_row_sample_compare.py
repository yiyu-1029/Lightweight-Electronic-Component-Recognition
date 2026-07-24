import torch
import torch.nn as nn
from torchvision import transforms, datasets
from torchvision.models import mobilenet_v2
from PIL import Image
import matplotlib.pyplot as plt

# ====================== 配置区 ======================
IMG_EASY_PATH = r"/test_component.jpg"
IMG_HARD_PATH = r"/val_sample_export/microchip_val_test.jpg"
DATA_ROOT = r"D:\archive (1)"

IMG_SIZE = 224
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
DEVICE = torch.device("cpu")

MODEL_LIST = [
    {"name": "原始基线模型", "path": "best_mobilenetv2_opt.pth", "type": "baseline"},
    {"name": "迭代式剪枝30%", "path": "iter_structured_pruned_30.pth", "type": "full"},
    {"name": "迭代式剪枝50%", "path": "iter_structured_pruned_50.pth", "type": "full"},
    {"name": "剪枝30% + INT8动态量化", "path": "quantized_iter_30.pth", "type": "full"}
]

# ====================== 类别名称加载 ======================
try:
    full_dataset = datasets.ImageFolder(root=DATA_ROOT)
    CLASS_NAMES = full_dataset.classes
except:
    CLASS_NAMES = []

def get_safe_class_name(idx):
    if 0 <= idx < len(CLASS_NAMES):
        return CLASS_NAMES[idx]
    return f"类别{idx+1}"

# ====================== 预处理、推理、加载模型 ======================
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=MEAN, std=STD)
])

def predict_single_img(model, pil_img):
    tensor = transform(pil_img).unsqueeze(0).to(DEVICE)
    model.eval()
    with torch.no_grad():
        out = model(tensor)
        prob = torch.nn.functional.softmax(out, dim=1)
        conf, pred_idx = torch.max(prob, dim=1)
    cls_name = get_safe_class_name(pred_idx.item())
    return cls_name, conf.item() * 100

def load_single_model(model_info):
    path = model_info["path"]
    m_type = model_info["type"]
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

def run_all_models(pil_img):
    res = []
    for m_info in MODEL_LIST:
        print(f"推理 {m_info['name']}")
        m = load_single_model(m_info)
        cls, conf = predict_single_img(m, pil_img)
        res.append({"name": m_info["name"], "cls": cls, "conf": conf})
        del m
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return res

# ====================== 主绘图：2行×5列布局 ======================
if __name__ == "__main__":
    plt.rcParams["font.sans-serif"] = ["SimHei"]
    plt.rcParams["axes.unicode_minus"] = False

    img_easy = Image.open(IMG_EASY_PATH).convert("RGB")
    img_hard = Image.open(IMG_HARD_PATH).convert("RGB")

    print("===== 推理简单样本（灯丝） =====")
    res_easy = run_all_models(img_easy)
    print("\n===== 推理困难样本（芯片） =====")
    res_hard = run_all_models(img_hard)

    # 布局：2行5列
    fig, axes = plt.subplots(2, 5, figsize=(18, 8), dpi=150)

    # 第一行：灯丝简单样本
    axes[0, 0].imshow(img_easy)
    axes[0, 0].set_title("简单样本原图\n(灯丝，高区分度)", fontsize=9, weight="bold")
    axes[0, 0].axis("off")
    for idx, item in enumerate(res_easy):
        ax = axes[0, idx+1]
        ax.imshow(img_easy)
        ax.set_title(item["name"], fontsize=8)
        ax.axis("off")
        txt = f"预测：{item['cls']}\n置信度：{item['conf']:.2f}%"
        ax.text(0.5, -0.22, txt, ha="center", transform=ax.transAxes, fontsize=7,
                bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.9))

    # 第二行：芯片困难样本
    axes[1, 0].imshow(img_hard)
    axes[1, 0].set_title("困难样本原图\n(芯片，特征模糊)", fontsize=9, weight="bold")
    axes[1, 0].axis("off")
    for idx, item in enumerate(res_hard):
        ax = axes[1, idx+1]
        ax.imshow(img_hard)
        ax.set_title(item["name"], fontsize=8)
        ax.axis("off")
        txt = f"预测：{item['cls']}\n置信度：{item['conf']:.2f}%"
        ax.text(0.5, -0.22, txt, ha="center", transform=ax.transAxes, fontsize=7,
                bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.9))

    fig.suptitle("模型压缩推理效果对比：简单清晰样本 VS 复杂模糊样本", fontsize=14, weight="bold", y=0.97)
    plt.tight_layout()
    plt.savefig("../../results/inference/double_sample_2row_compare.png", bbox_inches="tight", dpi=200)
    print("\n✅ 两行布局样本对比图已保存：double_sample_2row_compare.png")
    plt.show()
