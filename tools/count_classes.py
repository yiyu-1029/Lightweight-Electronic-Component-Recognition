import os

dataset_root = r"D:\archive (1)"

# 列出所有文件夹（每个文件夹就是一个类别）
class_list = [cls for cls in os.listdir(dataset_root) if os.path.isdir(os.path.join(dataset_root, cls))]

print("所有类别：")
for c in class_list:
    print("-", c)

print(f"\n总类别数量 num_classes = {len(class_list)}")
