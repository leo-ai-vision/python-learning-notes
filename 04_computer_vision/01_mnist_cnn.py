import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms


# =========================
# 1. 定义参数
# =========================

input_size = 28        # MNIST 图片大小：28×28
num_classes = 10       # 一共10个类别：0~9
num_epochs = 3         # 训练3轮
batch_size = 64        # 每次送入64张图片
learning_rate = 0.001  # 学习率


# =========================
# 2. 加载 MNIST 数据集
# =========================

# 将图片转换成 Tensor
transform = transforms.ToTensor()

# 训练集
train_dataset = datasets.MNIST(
    root="./data",       # 数据保存的位置
    train=True,          # True表示训练集
    transform=transform,
    download=True        # 没有数据时自动下载
)

# 测试集
test_dataset = datasets.MNIST(
    root="./data",
    train=False,         # False表示测试集
    transform=transform,
    download=True
)


# =========================
# 3. 创建 DataLoader
# =========================

train_loader = torch.utils.data.DataLoader(
    dataset=train_dataset,
    batch_size=batch_size,   # 每个batch有64张图片
    shuffle=True             # 每轮训练前打乱数据
)

test_loader = torch.utils.data.DataLoader(
    dataset=test_dataset,
    batch_size=batch_size,
    shuffle=False            # 测试集不需要打乱
)


# =========================
# 4. 定义 CNN 网络
# =========================

class CNN(nn.Module):

    def __init__(self):
        super(CNN, self).__init__()

        # -------------------------
        # 第一组卷积
        # 输入：(batch_size, 1, 28, 28)
        # -------------------------
        self.conv1 = nn.Sequential(

            nn.Conv2d(
                in_channels=1,      # 输入通道数：灰度图，所以是1
                out_channels=16,    # 输出16张特征图
                kernel_size=5,      # 卷积核大小：5×5
                stride=1,           # 步长：1
                padding=2           # 补2圈0，使图片大小保持不变
            ),
            # 输出：(batch_size, 16, 28, 28)

            nn.ReLU(),              # ReLU激活函数

            nn.MaxPool2d(
                kernel_size=2       # 2×2最大池化
            )
            # 输出：(batch_size, 16, 14, 14)
        )


        # -------------------------
        # 第二组卷积
        # 输入：(batch_size, 16, 14, 14)
        # -------------------------
        self.conv2 = nn.Sequential(

            # 输入16个通道，输出32个通道
            nn.Conv2d(16, 32, 5, 1, 2),
            # 输出：(batch_size, 32, 14, 14)

            nn.ReLU(),

            # 再进行一次卷积
            nn.Conv2d(32, 32, 5, 1, 2),
            # 输出：(batch_size, 32, 14, 14)

            nn.ReLU(),

            # 最大池化
            nn.MaxPool2d(2)
            # 输出：(batch_size, 32, 7, 7)
        )


        # -------------------------
        # 第三组卷积
        # 输入：(batch_size, 32, 7, 7)
        # -------------------------
        self.conv3 = nn.Sequential(

            nn.Conv2d(32, 64, 5, 1, 2),
            # 输出：(batch_size, 64, 7, 7)

            nn.ReLU()
            # 输出仍然是：(batch_size, 64, 7, 7)
        )


        # -------------------------
        # 全连接层
        # -------------------------

        # conv3输出：
        # 64个通道，每个特征图大小为7×7
        #
        # 每张图片一共有：
        # 64 × 7 × 7 = 3136 个特征
        #
        # 将3136个特征转换成10个类别的预测分数
        self.out = nn.Linear(
            64 * 7 * 7,
            10
        )


    def forward(self, x):
        """
        定义数据在网络中的前向传播过程
        """

        # 输入：
        # (batch_size, 1, 28, 28)


        # 第一组卷积
        x = self.conv1(x)

        # 输出：
        # (batch_size, 16, 14, 14)


        # 第二组卷积
        x = self.conv2(x)

        # 输出：
        # (batch_size, 32, 7, 7)


        # 第三组卷积
        x = self.conv3(x)

        # 输出：
        # (batch_size, 64, 7, 7)


        # ==============================
        # Flatten 拉平操作
        # ==============================

        # 当前 x 的形状：
        #
        # (batch_size, 64, 7, 7)
        #
        # 例如 batch_size = 64：
        #
        # (64, 64, 7, 7)
        #
        #
        # x.size(0)
        #
        # 表示获取第0维大小，
        # 也就是当前batch中有多少张图片。
        #
        # 如果batch_size=64：
        #
        # x.size(0) = 64
        #
        #
        # -1表示：
        #
        # 这一维让PyTorch自动计算。
        #
        # 因为：
        #
        # 64 × 7 × 7 = 3136
        #
        # 所以：
        #
        # (batch_size, 64, 7, 7)
        #
        #           ↓
        #
        # (batch_size, 3136)
        #
        # 也就是每一张图片的所有特征拉成一条长向量

        x = x.view(x.size(0), -1)


        # ==============================
        # 全连接层
        # ==============================

        # 输入：
        # (batch_size, 3136)
        #
        # 输出：
        # (batch_size, 10)
        #
        # 每张图片得到10个预测分数
        # 分别对应数字0~9

        output = self.out(x)


        # 返回模型预测结果
        return output


# =========================
# 5. 定义准确率函数
# =========================

def accuracy(predictions, labels):

    # predictions形状：
    # (batch_size, 10)
    #
    # 每张图片有10个预测分数


    # torch.max(predictions.data, 1)
    #
    # 1表示：
    # 在每张图片的10个类别分数中找最大值
    #
    # torch.max()会返回：
    #
    # 最大值
    # 最大值对应的索引
    #
    # [1]表示只取最大值对应的索引
    #
    # 这个索引就是模型预测的类别

    pred = torch.max(predictions.data, 1)[1]


    # 将预测类别和真实标签进行比较
    #
    # 相同 → True
    # 不同 → False
    #
    # sum()统计预测正确的数量

    rights = pred.eq(
        labels.data.view_as(pred)
    ).sum()


    # 返回：
    #
    # 预测正确数量
    # 当前batch总图片数量

    return rights, len(labels)


# =========================
# 6. 创建网络
# =========================

net = CNN()


# 打印网络结构
print(net)


# =========================
# 7. 定义损失函数
# =========================

# CrossEntropyLoss：
# 多分类任务常用的交叉熵损失函数

criterion = nn.CrossEntropyLoss()


# =========================
# 8. 定义优化器
# =========================

# Adam优化器
#
# net.parameters()
# 表示需要更新CNN网络中的所有参数

optimizer = optim.Adam(
    net.parameters(),
    lr=learning_rate
)


# =========================
# 9. 开始训练
# =========================

# epoch：
# 整个训练集完整训练一次叫一个epoch
#
# num_epochs=3
# 表示完整训练3遍

for epoch in range(num_epochs):


    # 保存当前epoch中
    # 每个batch预测正确数量和总数量

    train_rights = []


    # train_loader每次返回：
    #
    # data   → 图片
    # target → 图片对应的真实标签

    for batch_idx, (data, target) in enumerate(train_loader):


        # =========================
        # 设置为训练模式
        # =========================

        net.train()


        # =========================
        # 前向传播
        # =========================

        # 将图片送入CNN
        #
        # data形状：
        # (batch_size, 1, 28, 28)

        output = net(data)


        # =========================
        # 计算损失
        # =========================

        # output：
        # 模型预测结果
        #
        # target：
        # 真实标签

        loss = criterion(
            output,
            target
        )


        # =========================
        # 清空梯度
        # =========================

        # PyTorch中的梯度默认会累加
        #
        # 所以每次反向传播之前
        # 都要先把上一轮梯度清空

        optimizer.zero_grad()


        # =========================
        # 反向传播
        # =========================

        # 根据loss计算网络参数的梯度

        loss.backward()


        # =========================
        # 更新网络参数
        # =========================

        # optimizer根据计算得到的梯度
        # 更新CNN中的权重和偏置参数

        optimizer.step()


        # =========================
        # 计算训练集准确率
        # =========================

        right = accuracy(
            output,
            target
        )


        # 保存当前batch结果

        train_rights.append(right)


        # =========================
        # 每100个batch测试一次
        # =========================

        if batch_idx % 100 == 0:


            # 设置为测试模式

            net.eval()


            # 保存测试集统计结果

            val_rights = []


            # =========================
            # 测试阶段
            # =========================

            # 测试时不需要计算梯度
            #
            # 可以节省内存
            # 并提高运行速度

            with torch.no_grad():


                for data, target in test_loader:


                    # 将测试图片输入CNN

                    output = net(data)


                    # 计算当前batch
                    # 预测正确的数量

                    right = accuracy(
                        output,
                        target
                    )


                    # 保存结果

                    val_rights.append(right)


            # =========================
            # 计算训练集准确率
            # =========================

            # tup[0]：
            # 预测正确数量
            #
            # tup[1]：
            # 图片总数量

            train_r = (
                sum(
                    tup[0]
                    for tup in train_rights
                ),

                sum(
                    tup[1]
                    for tup in train_rights
                )
            )


            # =========================
            # 计算测试集准确率
            # =========================

            val_r = (
                sum(
                    tup[0]
                    for tup in val_rights
                ),

                sum(
                    tup[1]
                    for tup in val_rights
                )
            )


            # 训练集准确率

            train_acc = (
                100.0
                * train_r[0].item()
                / train_r[1]
            )


            # 测试集准确率

            val_acc = (
                100.0
                * val_r[0].item()
                / val_r[1]
            )


            # =========================
            # 打印训练结果
            # =========================

            print(
                "当前Epoch: {} "
                "[{}/{} ({:.0f}%)]\t"
                "Loss: {:.6f}\t"
                "训练集准确率: {:.2f}%\t"
                "测试集准确率: {:.2f}%".format(

                    epoch + 1,

                    batch_idx * batch_size,

                    len(train_loader.dataset),

                    100.0
                    * batch_idx
                    / len(train_loader),

                    loss.item(),

                    train_acc,

                    val_acc
                )
            )