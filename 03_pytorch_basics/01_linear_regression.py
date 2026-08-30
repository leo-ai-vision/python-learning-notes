import torch
import torch.nn as nn
import numpy as np
from torch.optim import SGD


# ==================== 准备训练数据 ====================

# 生成 x：0～10
x_values = [i for i in range(11)]

# 列表 -> float32 NumPy 数组
x_train = np.array(x_values, dtype=np.float32)

# 从一维 (11,) 变为二维 (11, 1)
# 11 个样本，每个样本有 1 个特征
x_train = x_train.reshape(-1, 1)

print("x_train 形状：", x_train.shape)


# 根据 y = 2x + 1 生成标签
y_values = [2 * i + 1 for i in x_values]

# 列表 -> float32 NumPy 数组
y_train = np.array(y_values, dtype=np.float32)

# 变为 (11, 1)
y_train = y_train.reshape(-1, 1)

print("y_train 形状：", y_train.shape)


# ==================== 定义线性回归模型 ====================

class LinearRegressionModel(nn.Module):

    def __init__(self, input_dim, output_dim):
        super().__init__()

        # 输入形状：[样本数, input_dim]
        # 输出形状：[样本数, output_dim]
        # 内部执行：y = wx + b
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        # 输入：特征 Tensor
        # 输出：预测结果 Tensor
        return self.linear(x)


# 输入和输出维度都是 1
input_dim = 1
output_dim = 1

# 创建模型
model = LinearRegressionModel(input_dim, output_dim)

# 选择 GPU 或 CPU
device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

# 将模型移动到对应设备
model = model.to(device)

print("使用设备：", device)


# ==================== 转换训练数据 ====================

# NumPy 数组 -> PyTorch Tensor -> 移动到 GPU/CPU
# 这些数据不会变化，所以放在循环外转换一次即可
inputs = torch.from_numpy(x_train).to(device)
labels = torch.from_numpy(y_train).to(device)


# ==================== 损失函数和优化器 ====================

# 均方误差损失
criterion = nn.MSELoss()

learning_rate = 0.01

# 正确写法：创建 SGD 优化器
optimizer = SGD(
    model.parameters(),
    lr=learning_rate
)


# ==================== 开始训练 ====================

epochs = 1000

for epoch in range(1, epochs + 1):

    # 清除上一轮保存的梯度
    # PyTorch 默认会累加梯度，因此必须调用
    optimizer.zero_grad()

    # 前向传播
    predictions = model(inputs)

    # 计算预测值与真实标签之间的损失
    loss = criterion(predictions, labels)

    # 反向传播，计算梯度
    loss.backward()

    # 根据梯度更新权重和偏置
    optimizer.step()

    if epoch % 50 == 0:
        print(
            "epoch: {}, loss: {:.8f}".format(
                epoch,
                loss.item()
            )
        )


# ==================== 查看训练结果 ====================

# 训练结果应该接近 weight=2、bias=1
print("权重：", model.linear.weight.item())
print("偏置：", model.linear.bias.item())