import matplotlib.pyplot as plt

# 本轮所有训练数据
epochs = list(range(1, 14))
train_loss = [2.3976, 1.8515, 1.6426, 1.4623, 1.3349, 1.2069, 1.0810, 0.9750, 0.8872, 0.7815, 0.6924, 0.6451, 0.5471]
train_acc = [0.3116, 0.4320, 0.4929, 0.5424, 0.5725, 0.6132, 0.6466, 0.6733, 0.7042, 0.7373, 0.7663, 0.7822, 0.8124]
val_loss  = [1.8665, 1.7580, 1.6634, 1.7037, 1.6387, 1.7277, 1.7340, 1.7417, 1.7907, 1.8400, 1.8862, 1.9432, 1.9717]
val_acc   = [0.4281, 0.4627, 0.5045, 0.4986, 0.5082, 0.5018, 0.5141, 0.5068, 0.5064, 0.5109, 0.5077, 0.5045, 0.5082]

# 解决中文乱码
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

plt.figure(figsize=(12, 5))

# 损失曲线
plt.subplot(1, 2, 1)
plt.plot(epochs, train_loss, label='Train Loss', linewidth=2)
plt.plot(epochs, val_loss, label='Val Loss', linewidth=2)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('训练/验证损失变化曲线')
plt.legend()
plt.grid(alpha=0.3)

# 准确率曲线
plt.subplot(1, 2, 2)
plt.plot(epochs, train_acc, label='Train Acc', linewidth=2)
plt.plot(epochs, val_acc, label='Val Acc', linewidth=2)
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('训练/验证准确率变化曲线')
plt.legend()
plt.grid(alpha=0.3)

plt.tight_layout()
plt.savefig("../../results/training_curves/opt_train_curve.png", dpi=300)
plt.show()
