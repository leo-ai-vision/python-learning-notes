"""
MNIST 手写数字神经网络分类任务

主要流程：
1. 下载并读取 MNIST 数据集
2. 查看 MNIST 数据
3. 将 NumPy 数据转换成 PyTorch Tensor
4. 创建 TensorDataset 和 DataLoader
5. 使用 nn.Module 定义神经网络
6. 定义损失函数和优化器
7. 训练神经网络
8. 在验证集上计算准确率
"""

# ============================================================
# 一、导入需要使用的库
# ============================================================

from pathlib import Path
import requests
import pickle
import gzip

import numpy as np

import torch
from torch import nn
from torch import optim
import torch.nn.functional as F

# PyTorch 数据集和数据加载器
from torch.utils.data import TensorDataset, DataLoader

# 用于显示 MNIST 图片
from matplotlib import pyplot as plt


# ============================================================
# 二、设置 MNIST 数据集保存路径
# ============================================================

# 创建 Path 对象，表示 data 文件夹
DATA_PATH = Path("data")

# data/mnist
PATH = DATA_PATH / "mnist"

# 创建 data/mnist 文件夹
# parents=True：
# 如果上一级 data 文件夹不存在，也一起创建
#
# exist_ok=True：
# 如果文件夹已经存在，不报错
PATH.mkdir(parents=True, exist_ok=True)


# MNIST 数据集下载地址
URL = "http://deeplearning.net/data/mnist/"

# 数据集文件名称
FILENAME = "mnist.pkl.gz"


# ============================================================
# 三、读取 MNIST 数据集
# ============================================================

def load_mnist():
    """
    读取 MNIST 数据集。

    如果本地没有 mnist.pkl.gz：
        先自动下载

    如果已经存在：
        直接读取

    返回：
        x_train：训练图片
        y_train：训练标签
        x_valid：验证图片
        y_valid：验证标签
    """

    # 得到完整文件路径：
    # data/mnist/mnist.pkl.gz
    file_path = PATH / FILENAME

    # 判断文件是否存在
    if not file_path.exists():

        print("本地没有 MNIST 数据集，开始下载...")

        # 向服务器发送 GET 请求
        response = requests.get(
            URL + FILENAME,
            timeout=30
        )

        # 如果下载失败，直接抛出异常
        response.raise_for_status()

        # 把下载得到的二进制数据写入文件
        file_path.write_bytes(response.content)

        print("MNIST 数据集下载完成。")

    else:
        print("检测到本地 MNIST 数据集，直接读取。")

    # gzip.open：
    # 打开 .gz 压缩文件
    #
    # "rb"：
    # read binary，二进制读取模式
    with gzip.open(file_path.as_posix(), "rb") as f:

        # pickle.load：
        # 从 pickle 文件中恢复保存的数据
        #
        # 数据结构大致为：
        #
        # (
        #     (x_train, y_train),
        #     (x_valid, y_valid),
        #     (x_test, y_test)
        # )
        #
        # 最后的测试集这里暂时不用，所以使用 _ 接收
        (x_train, y_train), (x_valid, y_valid), _ = pickle.load(
            f,
            encoding="latin-1"
        )

    return x_train, y_train, x_valid, y_valid


# ============================================================
# 四、定义神经网络
# ============================================================

class Mnist_NN(nn.Module):
    """
    创建一个简单的全连接神经网络。

    网络结构：

    784
     ↓
    128
     ↓
    256
     ↓
     10

    MNIST 图片大小：
        28 × 28 = 784

    最后的 10 个输出分别对应：
        0、1、2、3、4、5、6、7、8、9
    """

    def __init__(self):

        # 调用父类 nn.Module 的初始化方法
        super().__init__()

        # ----------------------------------------------------
        # 第一层全连接层
        #
        # 输入：
        # 784 个像素
        #
        # 输出：
        # 128 个神经元
        # ----------------------------------------------------
        self.hidden1 = nn.Linear(
            784,
            128
        )

        # ----------------------------------------------------
        # 第二层全连接层
        #
        # 输入：
        # 128
        #
        # 输出：
        # 256
        # ----------------------------------------------------
        self.hidden2 = nn.Linear(
            128,
            256
        )

        # ----------------------------------------------------
        # 输出层
        #
        # 输入：
        # 256
        #
        # 输出：
        # 10
        #
        # 10 个数字类别：0～9
        # ----------------------------------------------------
        self.out = nn.Linear(
            256,
            10
        )

    def forward(self, x):
        """
        定义数据经过神经网络的前向传播过程。
        """

        # 第一层：
        # Linear + ReLU
        x = F.relu(
            self.hidden1(x)
        )

        # 第二层：
        # Linear + ReLU
        x = F.relu(
            self.hidden2(x)
        )

        # 输出层
        #
        # 注意：
        # 这里不用 softmax
        #
        # 因为后面使用的 cross_entropy
        # 内部已经包含了相关计算
        x = self.out(x)

        return x


# ============================================================
# 五、定义损失函数
# ============================================================

# cross_entropy：
# 交叉熵损失函数
#
# 多分类任务最常用的损失函数之一
#
# MNIST 属于：
# 10 分类任务
loss_func = F.cross_entropy


# ============================================================
# 六、创建 DataLoader
# ============================================================

def get_data(train_ds, valid_ds, bs):
    """
    根据 Dataset 创建 DataLoader。

    train_ds：
        训练集

    valid_ds：
        验证集

    bs：
        batch size
    """

    # 训练集：
    # batch_size = bs
    #
    # shuffle=True：
    # 每个 epoch 开始前打乱数据
    train_dl = DataLoader(
        train_ds,
        batch_size=bs,
        shuffle=True
    )

    # 验证集：
    # batch_size 使用训练集的 2 倍
    #
    # 验证阶段不需要反向传播，
    # 因此可以一次处理更多样本
    valid_dl = DataLoader(
        valid_ds,
        batch_size=bs * 2
    )

    return train_dl, valid_dl


# ============================================================
# 七、定义单个 Batch 的训练过程
# ============================================================

def loss_batch(model, loss_func, xb, yb, opt=None):
    """
    计算一个 batch 的 loss。

    参数：

    model：
        神经网络模型

    loss_func：
        损失函数

    xb：
        一个 batch 的图片

    yb：
        一个 batch 的真实标签

    opt：
        优化器

    如果 opt 不为 None：
        表示训练阶段
        需要反向传播并更新参数

    如果 opt 为 None：
        表示验证阶段
        只计算 loss，不更新模型
    """

    # --------------------------------------------------------
    # 前向传播
    #
    # model(xb)：
    # 把图片输入神经网络
    #
    # 得到预测结果以后，
    # 与真实标签 yb 计算损失
    # --------------------------------------------------------
    loss = loss_func(
        model(xb),
        yb
    )

    # 如果传入了优化器，
    # 说明当前正在训练
    if opt is not None:

        # ----------------------------------------------------
        # 1. 反向传播
        #
        # 根据 loss 自动计算每个参数的梯度
        # ----------------------------------------------------
        loss.backward()

        # ----------------------------------------------------
        # 2. 更新参数
        #
        # 优化器根据梯度调整
        # weights 和 bias
        # ----------------------------------------------------
        opt.step()

        # ----------------------------------------------------
        # 3. 清空梯度
        #
        # PyTorch 中梯度默认会累加，
        # 所以每次更新完参数以后必须清零
        # ----------------------------------------------------
        opt.zero_grad()

    # loss.item()
    # 把 Tensor 类型的 loss 转换成普通 Python 数值
    #
    # len(xb)
    # 当前 batch 有多少个样本
    return loss.item(), len(xb)


# ============================================================
# 八、创建模型和优化器
# ============================================================

def get_model():
    """
    创建神经网络模型和优化器。
    """

    # 创建 Mnist_NN 模型对象
    model = Mnist_NN()

    # --------------------------------------------------------
    # SGD：
    # 随机梯度下降优化器
    #
    # model.parameters()：
    # 获取神经网络所有需要训练的参数
    #
    # lr：
    # learning rate，学习率
    # --------------------------------------------------------
    optimizer = optim.SGD(
        model.parameters(),
        lr=0.001
    )

    return model, optimizer


# ============================================================
# 九、训练模型
# ============================================================

def fit(
        steps,
        model,
        loss_func,
        opt,
        train_dl,
        valid_dl
):
    """
    训练神经网络。

    steps：
        训练多少轮

    model：
        神经网络

    loss_func：
        损失函数

    opt：
        优化器

    train_dl：
        训练集 DataLoader

    valid_dl：
        验证集 DataLoader
    """

    # 训练 steps 轮
    for step in range(steps):

        # ====================================================
        # 1. 训练阶段
        # ====================================================

        # 切换到训练模式
        model.train()

        # 每次从 DataLoader 中取出一个 batch
        for xb, yb in train_dl:

            # 对当前 batch：
            # 前向传播
            # ↓
            # 计算 loss
            # ↓
            # 反向传播
            # ↓
            # 更新参数
            loss_batch(
                model,
                loss_func,
                xb,
                yb,
                opt
            )

        # ====================================================
        # 2. 验证阶段
        # ====================================================

        # 切换到验证模式
        model.eval()

        # 验证的时候不需要计算梯度
        # 可以减少内存消耗，提高运行速度
        with torch.no_grad():

            # 对整个验证集逐个 batch 计算 loss
            losses, nums = zip(
                *[
                    loss_batch(
                        model,
                        loss_func,
                        xb,
                        yb
                    )

                    for xb, yb in valid_dl
                ]
            )

        # ----------------------------------------------------
        # 计算整个验证集的平均 loss
        #
        # losses：
        # 每个 batch 的 loss
        #
        # nums：
        # 每个 batch 的样本数量
        # ----------------------------------------------------
        val_loss = (
                np.sum(
                    np.multiply(
                        losses,
                        nums
                    )
                )
                /
                np.sum(nums)
        )

        # 每完成一轮训练，打印验证集损失
        print(
            "当前 step:",
            step,
            "验证集损失:",
            val_loss
        )


# ============================================================
# 十、计算模型准确率
# ============================================================

def evaluate_accuracy(model, valid_dl):
    """
    在验证集上计算模型分类准确率。
    """

    # correct：
    # 预测正确的样本数量
    correct = 0

    # total：
    # 总样本数量
    total = 0

    # 切换到验证模式
    model.eval()

    # 验证时不需要梯度
    with torch.no_grad():

        # 每次读取一个 batch
        for xb, yb in valid_dl:

            # ------------------------------------------------
            # 将图片输入神经网络
            #
            # 假设：
            #
            # xb.shape
            # =
            # [128, 784]
            #
            # 那么：
            #
            # outputs.shape
            # =
            # [128, 10]
            #
            # 表示：
            # 128 张图片
            # 每张图片都有 10 个类别分数
            # ------------------------------------------------
            outputs = model(xb)

            # ------------------------------------------------
            # torch.max(outputs, dim=1)
            #
            # 在每一行的 10 个类别分数中寻找最大值
            #
            # 第一个返回值：
            # 最大分数
            #
            # 第二个返回值：
            # 最大分数对应的位置，也就是预测类别
            #
            # 因为我们不需要最大分数本身，
            # 所以使用 _ 接收
            # ------------------------------------------------
            _, predicted = torch.max(
                outputs,
                dim=1
            )

            # 当前 batch 的样本数量
            total += yb.size(0)

            # ------------------------------------------------
            # predicted == yb
            #
            # 比较：
            #
            # 模型预测结果
            # 和
            # 真实标签
            #
            # 相等 → True
            # 不相等 → False
            #
            # .sum()
            # 统计 True 的数量
            #
            # .item()
            # 转换成普通 Python 数字
            # ------------------------------------------------
            correct += (
                predicted == yb
            ).sum().item()

    # 计算准确率
    accuracy = (
            100
            * correct
            / total
    )

    print(
        "Accuracy of the network on the validation images: %.2f %%"
        % accuracy
    )

    return accuracy


# ============================================================
# 十一、主程序
# ============================================================

def main():

    # ========================================================
    # 1. 读取 MNIST 数据
    # ========================================================

    x_train, y_train, x_valid, y_valid = load_mnist()

    # 查看训练集前 10 个标签
    print("\n前 10 个训练标签：")
    print(
        y_train[:10]
    )

    # 查看训练集形状
    print("\n训练集原始形状：")
    print(
        x_train.shape
    )

    # 一般输出：
    #
    # (50000, 784)
    #
    # 表示：
    #
    # 50000 张训练图片
    #
    # 每张图片：
    # 28 × 28 = 784 个像素


    # ========================================================
    # 2. 显示第一张 MNIST 图片
    # ========================================================

    # x_train[0]
    # 取训练集第一张图片
    #
    # 原本形状：
    # (784,)
    #
    # reshape(28, 28)
    # 恢复成 28 × 28 图片
    plt.imshow(
        x_train[0].reshape(28, 28),
        cmap="gray"
    )

    # 图片标题显示真实标签
    plt.title(
        f"Label: {y_train[0]}"
    )

    # PyCharm 中需要写 plt.show()
    # 才会真正弹出图片窗口
    plt.show()


    # ========================================================
    # 3. NumPy → PyTorch Tensor
    # ========================================================

    # 原来的 MNIST 数据是 NumPy 数组
    #
    # 使用 torch.tensor
    # 转换成 PyTorch Tensor
    x_train, y_train, x_valid, y_valid = map(

        torch.tensor,

        (
            x_train,
            y_train,
            x_valid,
            y_valid
        )
    )

    # --------------------------------------------------------
    # x_train.shape：
    #
    # (50000, 784)
    #
    # 因此：
    #
    # n = 50000
    # c = 784
    # --------------------------------------------------------
    n, c = x_train.shape

    print("\n转换成 Tensor 后：")

    print(
        "x_train.shape =",
        x_train.shape
    )

    print(
        "y_train 最小值 =",
        y_train.min()
    )

    print(
        "y_train 最大值 =",
        y_train.max()
    )


    # ========================================================
    # 4. 设置 Batch Size
    # ========================================================

    # 每次训练使用 64 张图片
    bs = 64


    # ========================================================
    # 5. 演示一个 Batch
    # ========================================================

    # 取前 64 张训练图片
    xb = x_train[0:bs]

    # 取对应的 64 个标签
    yb = y_train[0:bs]

    print("\n一个 batch 的图片形状：")
    print(
        xb.shape
    )

    print("\n一个 batch 的标签形状：")
    print(
        yb.shape
    )


    # ========================================================
    # 6. 演示手动创建权重和偏置
    # ========================================================

    # --------------------------------------------------------
    # 创建权重矩阵
    #
    # 输入：
    # 784
    #
    # 输出：
    # 10
    #
    # 所以 weights.shape：
    #
    # [784, 10]
    # --------------------------------------------------------
    weights = torch.randn(
        [784, 10],
        dtype=torch.float,
        requires_grad=True
    )

    # --------------------------------------------------------
    # 创建偏置
    #
    # 10 个输出类别
    # 所以需要 10 个 bias
    # --------------------------------------------------------
    bias = torch.zeros(
        10,
        requires_grad=True
    )

    # --------------------------------------------------------
    # xb.mm(weights)
    #
    # 矩阵乘法：
    #
    # [64, 784]
    #
    # ×
    #
    # [784, 10]
    #
    # =
    #
    # [64, 10]
    #
    # 然后加上 bias
    # --------------------------------------------------------
    simple_outputs = (
            xb.mm(weights)
            + bias
    )

    # 计算简单线性模型的 loss
    simple_loss = loss_func(
        simple_outputs,
        yb
    )

    print("\n简单线性模型的初始损失：")
    print(
        simple_loss
    )


    # ========================================================
    # 7. 创建神经网络
    # ========================================================

    net = Mnist_NN()

    print("\n神经网络结构：")

    print(
        net
    )


    # ========================================================
    # 8. 查看网络所有参数
    # ========================================================

    print("\n模型中的权重和偏置参数：")

    # named_parameters()
    #
    # 返回模型中的：
    #
    # 参数名称
    # +
    # 参数 Tensor
    for name, parameter in net.named_parameters():

        print(
            name,
            parameter.size()
        )


    # ========================================================
    # 9. 创建 TensorDataset
    # ========================================================

    # 把图片和标签组合成训练集 Dataset
    train_ds = TensorDataset(
        x_train,
        y_train
    )

    # 创建验证集 Dataset
    valid_ds = TensorDataset(
        x_valid,
        y_valid
    )


    # ========================================================
    # 10. 创建 DataLoader
    # ========================================================

    train_dl, valid_dl = get_data(
        train_ds,
        valid_ds,
        bs
    )


    # ========================================================
    # 11. 创建正式训练模型
    # ========================================================

    model, opt = get_model()


    # ========================================================
    # 12. 开始训练
    # ========================================================

    print("\n开始训练：")

    # 训练 25 轮
    fit(
        25,
        model,
        loss_func,
        opt,
        train_dl,
        valid_dl
    )


    # ========================================================
    # 13. 计算最终准确率
    # ========================================================

    print("\n模型验证结果：")

    evaluate_accuracy(
        model,
        valid_dl
    )


# ============================================================
# 十二、程序入口
# ============================================================

# __name__ == "__main__"
#
# 表示：
#
# 只有直接运行当前这个 .py 文件的时候，
# 才执行 main()
#
# 如果以后这个文件被其他 Python 文件 import，
# main() 不会自动运行
if __name__ == "__main__":
    main()
