import os
import matplotlib.pyplot as plt


dataset_root = r"D:\component-recognition\archive (1)\images"

counts = {}
for cls in os.listdir(dataset_root):
    cls_path = os.path.join(dataset_root, cls)
    if not os.path.isdir(cls_path):
        continue
    imgs = [f for f in os.listdir(cls_path)
            if f.lower().endswith((".jpg", ".png", ".jpeg"))]
    counts[cls] = len(imgs)

# 打印数量
print("各类别样本数：")
for k, v in counts.items():
    print(f"{k}: {v}")

# 绘图
plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False
plt.figure(figsize=(12,5))
plt.bar(counts.keys(), counts.values())
plt.xticks(rotation=45, ha="right")
plt.title("元器件数据集分布")
plt.tight_layout()
plt.show()
