from torchvision import datasets

DATA_ROOT = r"D:\archive (1)"
full_dataset = datasets.ImageFolder(root=DATA_ROOT)
class_names = full_dataset.classes

print("数据集类别列表（按索引顺序）：")
print(class_names)
print(f"\n共 {len(class_names)} 个类别")

# 类别名称按行写入文件
with open("../data/class_names.txt", "w", encoding="utf-8") as f:
    for name in class_names:
        f.write(name + "\n")

print("\n已保存到 class_names.txt")
