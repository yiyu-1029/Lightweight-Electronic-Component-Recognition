print("RUNNING CLEAN PROJECT")
import os
import streamlit as st
import torch
import torchvision.transforms as transforms
from PIL import Image


# ==================== Path Configuration ====================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "quantized",
    "quantized_iter_30_script.pt"
)

CLASS_PATH = os.path.join(
    BASE_DIR,
    "data",
    "class_names.txt"
)


# ==================== Model Parameters ====================

IMG_SIZE = 224

MEAN = [0.485, 0.456, 0.406]

STD = [0.229, 0.224, 0.225]


# ==================== Load Classes ====================

with open(CLASS_PATH, "r", encoding="utf-8") as f:
    CLASS_NAMES = [
        line.strip()
        for line in f.readlines()
        if line.strip()
    ]

# 输出类别数量用于检查
print("当前读取到的类别总数：", len(CLASS_NAMES))

# =================================================

# 页面配置
st.set_page_config(page_title="电子元器件识别系统", page_icon="🔌", layout="centered")


# 缓存加载模型，只加载一次提升速度
@st.cache_resource
def load_model():
    model = torch.jit.load(MODEL_PATH, map_location="cpu")
    model.eval()
    return model


# 图像预处理
def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD)
    ])
    return transform(image).unsqueeze(0)


# 推理函数
def predict(model, image):
    img_tensor = preprocess_image(image)
    with torch.no_grad():
        output = model(img_tensor)
        prob = torch.nn.functional.softmax(output, dim=1)
        confidence, pred_idx = torch.max(prob, dim=1)
    pred_class = CLASS_NAMES[pred_idx.item()]
    conf_percent = confidence.item() * 100
    return pred_class, conf_percent


# ===================== 页面主体 =====================
st.title("🔌 电子元器件智能识别系统")
st.markdown("基于 MobileNetV2 + 结构化剪枝 + INT8动态量化 的轻量化识别模型")
st.divider()

# 图片上传
uploaded_file = st.file_uploader("上传一张电子元器件图片", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # 显示图片
    image = Image.open(uploaded_file).convert("RGB")
    col1, col2 = st.columns([1, 1])
    with col1:
        st.image(image, caption="上传的图片", use_column_width=True)

    # 加载模型 + 推理
    model = load_model()
    pred_class, confidence = predict(model, image)

    # 显示结果
    with col2:
        st.subheader("识别结果")
        st.success(f"**类别：{pred_class}**")
        st.metric(label="置信度", value=f"{confidence:.2f} %")
        st.info("模型已完成剪枝+INT8量化，体积压缩3倍，精度损失小于1%")

st.divider()
st.caption("项目说明：基于PyTorch实现的元器件分类系统，支持端侧轻量化部署")
