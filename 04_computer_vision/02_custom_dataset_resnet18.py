# -*- coding: utf-8 -*-

from pathlib import Path
import copy
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models


# =========================================================
# 1. 数据路径和训练参数
# =========================================================

# 当前 .py 文件所在目录
PROJECT_DIR = Path(__file__).resolve().parent

# flower_data 文件夹需要和当前 .py 文件放在同一级目录
DATA_DIR = PROJECT_DIR / "flower_data"

TRAIN_DIR = DATA_DIR / "train_filelist"
VALID_DIR = DATA_DIR / "val_filelist"

TRAIN_TXT = DATA_DIR / "train.txt"
VALID_TXT = DATA_DIR / "val.txt"

BATCH_SIZE = 64
NUM_CLASSES = 102
NUM_EPOCHS = 20
LEARNING_RATE = 1e-3

BEST_MODEL_PATH = PROJECT_DIR / "best.pth"


# =========================================================
# 2. 读取 txt 文件中的图片名称和标签
# =========================================================

def load_annotations(ann_file):
    """
    读取标注文件。

    txt 文件每一行的格式：
    图片名称 标签

    例如：
    image_001.jpg 0
    """

    data_infos = {}

    with open(ann_file, "r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()

            # 跳过空行
            if not line:
                continue

            parts = line.split()

            if len(parts) != 2:
                raise ValueError(
                    f"{ann_file} 第 {line_number} 行格式错误：{line}"
                )

            filename, gt_label = parts

            data_infos[filename] = np.int64(gt_label)

    return data_infos


# =========================================================
# 3. 自定义花卉数据集
# =========================================================

class FlowerDataset(Dataset):

    def __init__(self, root_dir, ann_file, transform=None):

        self.root_dir = Path(root_dir)
        self.ann_file = Path(ann_file)
        self.transform = transform

        # 读取图片名称和标签
        self.img_label = load_annotations(self.ann_file)

        # 保存完整图片路径
        self.images = [
            self.root_dir / image_name
            for image_name in self.img_label.keys()
        ]

        # 保存标签
        self.labels = list(self.img_label.values())

    def __len__(self):
        """返回数据集中的图片数量。"""

        return len(self.images)

    def __getitem__(self, index):
        """根据索引读取一张图片及其标签。"""

        image_path = self.images[index]

        # 读取图片，并统一转换为 RGB 三通道
        with Image.open(image_path) as image_file:
            image = image_file.convert("RGB")

        # 读取标签
        label = self.labels[index]

        # 对图片进行预处理
        if self.transform is not None:
            image = self.transform(image)

        # 标签转换为 LongTensor
        label = torch.tensor(label, dtype=torch.long)

        return image, label


# =========================================================
# 4. 图片预处理
# =========================================================

data_transforms = {

    "train": transforms.Compose([

        # 调整图片大小
        transforms.Resize(64),

        # 随机旋转，角度范围为 -45 到 45 度
        transforms.RandomRotation(45),

        # 从图片中心裁剪为 64×64
        transforms.CenterCrop(64),

        # 以 0.5 的概率水平翻转
        transforms.RandomHorizontalFlip(p=0.5),

        # 以 0.5 的概率垂直翻转
        transforms.RandomVerticalFlip(p=0.5),

        # 随机调整亮度、对比度、饱和度和色相
        transforms.ColorJitter(
            brightness=0.2,
            contrast=0.1,
            saturation=0.1,
            hue=0.1
        ),

        # 以 0.025 的概率转换为灰度图
        transforms.RandomGrayscale(p=0.025),

        # 将图片转换为 Tensor
        transforms.ToTensor(),

        # 图像归一化
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ]),

    "valid": transforms.Compose([

        transforms.Resize(64),

        transforms.CenterCrop(64),

        transforms.ToTensor(),

        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
}


# =========================================================
# 5. 创建 Dataset 和 DataLoader
# =========================================================

def create_dataloaders():

    # 创建训练集
    train_dataset = FlowerDataset(
        root_dir=TRAIN_DIR,
        ann_file=TRAIN_TXT,
        transform=data_transforms["train"]
    )

    # 创建验证集
    valid_dataset = FlowerDataset(
        root_dir=VALID_DIR,
        ann_file=VALID_TXT,
        transform=data_transforms["valid"]
    )

    # 创建训练集 DataLoader
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0
    )

    # 创建验证集 DataLoader
    valid_loader = DataLoader(
        dataset=valid_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )

    return (
        train_dataset,
        valid_dataset,
        train_loader,
        valid_loader
    )


# =========================================================
# 6. 显示 DataLoader 中的一张图片
# =========================================================

def show_one_sample(data_loader, title):

    # 获取一个批次的数据
    images, labels = next(iter(data_loader))

    # 获取批次中的第一张图片
    sample = images[0]

    # 将通道顺序从 C×H×W 改成 H×W×C
    sample = sample.permute(1, 2, 0).numpy()

    # 反归一化：先乘标准差
    sample = sample * np.array(
        [0.229, 0.224, 0.225]
    )

    # 再加均值
    sample = sample + np.array(
        [0.485, 0.456, 0.406]
    )

    # 将像素值限制在 0 到 1 之间
    sample = np.clip(sample, 0, 1)

    # 显示图片
    plt.figure(figsize=(5, 5))
    plt.imshow(sample)
    plt.title(
        f"{title}，标签：{labels[0].item()}"
    )
    plt.axis("off")
    plt.tight_layout()
    plt.show()

    print(
        f"{title}的标签是：{labels[0].item()}"
    )


# =========================================================
# 7. 创建 ResNet18 模型
# =========================================================

def build_model():

    # 不使用预训练权重
    model = models.resnet18(weights=None)

    # 获取原来全连接层的输入特征数量
    num_features = model.fc.in_features

    # 将最后的全连接层修改为 102 类
    model.fc = nn.Linear(
        in_features=num_features,
        out_features=NUM_CLASSES
    )

    return model


# =========================================================
# 8. 训练模型
# =========================================================

def train_model(
        model,
        dataloaders,
        criterion,
        optimizer,
        scheduler,
        device,
        num_epochs=20,
        filename=BEST_MODEL_PATH
):

    start_time = time.time()

    # 保存最佳验证集准确率
    best_acc = 0.0

    # 保存最佳模型参数
    best_model_weights = copy.deepcopy(
        model.state_dict()
    )

    # 保存训练过程
    train_acc_history = []
    val_acc_history = []

    train_losses = []
    valid_losses = []

    learning_rates = [
        optimizer.param_groups[0]["lr"]
    ]

    # 模型放到 CPU 或 GPU
    model = model.to(device)

    for epoch in range(num_epochs):

        print(
            f"Epoch {epoch + 1}/{num_epochs}"
        )

        print("-" * 30)

        # 每个 epoch 都进行训练和验证
        for phase in ["train", "valid"]:

            if phase == "train":

                # 训练模式
                model.train()

            else:

                # 验证模式
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            # 遍历当前数据集
            for inputs, labels in dataloaders[phase]:

                # 数据移动到 CPU 或 GPU
                inputs = inputs.to(device)
                labels = labels.to(device)

                # 清空上一轮梯度
                optimizer.zero_grad()

                # 训练阶段计算梯度，验证阶段不计算梯度
                with torch.set_grad_enabled(
                        phase == "train"
                ):

                    # 模型预测
                    outputs = model(inputs)

                    # 计算损失
                    loss = criterion(
                        outputs,
                        labels
                    )

                    # 获取预测类别
                    predictions = outputs.argmax(
                        dim=1
                    )

                    # 只有训练阶段才更新参数
                    if phase == "train":

                        loss.backward()

                        optimizer.step()

                # 计算整个批次的损失
                running_loss += (
                    loss.item() * inputs.size(0)
                )

                # 统计预测正确的数量
                running_corrects += (
                    predictions == labels
                ).sum().item()

            # 当前数据集样本数量
            dataset_size = len(
                dataloaders[phase].dataset
            )

            # 当前 epoch 平均损失
            epoch_loss = (
                running_loss / dataset_size
            )

            # 当前 epoch 准确率
            epoch_acc = (
                running_corrects / dataset_size
            )

            elapsed_time = (
                time.time() - start_time
            )

            print(
                f"{phase} "
                f"Loss：{epoch_loss:.4f}，"
                f"Acc：{epoch_acc:.4f}，"
                f"用时：{elapsed_time // 60:.0f}分"
                f"{elapsed_time % 60:.0f}秒"
            )

            # 保存验证集上表现最好的模型
            if (
                phase == "valid"
                and epoch_acc > best_acc
            ):

                best_acc = epoch_acc

                best_model_weights = copy.deepcopy(
                    model.state_dict()
                )

                checkpoint = {

                    # 模型参数
                    "state_dict":
                        model.state_dict(),

                    # 最佳准确率
                    "best_acc":
                        best_acc,

                    # 优化器参数
                    "optimizer":
                        optimizer.state_dict()
                }

                torch.save(
                    checkpoint,
                    filename
                )

            # 保存验证集结果
            if phase == "valid":

                val_acc_history.append(
                    epoch_acc
                )

                valid_losses.append(
                    epoch_loss
                )

            # 保存训练集结果
            if phase == "train":

                train_acc_history.append(
                    epoch_acc
                )

                train_losses.append(
                    epoch_loss
                )

        # 每完成一个 epoch 调整一次学习率
        scheduler.step()

        current_learning_rate = (
            optimizer.param_groups[0]["lr"]
        )

        learning_rates.append(
            current_learning_rate
        )

        print(
            f"当前学习率："
            f"{current_learning_rate:.7f}"
        )

        print()

    # 训练总时间
    total_time = (
        time.time() - start_time
    )

    print(
        f"训练完成，总用时："
        f"{total_time // 60:.0f}分"
        f"{total_time % 60:.0f}秒"
    )

    print(
        f"最佳验证集准确率："
        f"{best_acc:.4f}"
    )

    print(
        f"最佳模型保存位置："
        f"{Path(filename).resolve()}"
    )

    # 加载最佳模型参数
    model.load_state_dict(
        best_model_weights
    )

    return (
        model,
        val_acc_history,
        train_acc_history,
        valid_losses,
        train_losses,
        learning_rates
    )


# =========================================================
# 9. 检查数据文件是否存在
# =========================================================

def check_required_files():

    required_paths = [
        TRAIN_DIR,
        VALID_DIR,
        TRAIN_TXT,
        VALID_TXT
    ]

    missing_paths = [
        path
        for path in required_paths
        if not path.exists()
    ]

    if missing_paths:

        print("没有找到以下文件或文件夹：")

        for path in missing_paths:
            print(path)

        raise FileNotFoundError(
            "\n请把 flower_data 文件夹放在"
            "当前 .py 文件旁边，并检查文件夹名称。"
        )


# =========================================================
# 10. 主程序
# =========================================================

def main():

    # 检查数据文件
    check_required_files()

    # 判断是否可以使用 GPU
    if torch.cuda.is_available():

        device = torch.device("cuda:0")

        print("CUDA 可以使用，模型将在 GPU 上训练。")

    else:

        device = torch.device("cpu")

        print("CUDA 不可用，模型将在 CPU 上训练。")

    # 查看前 5 条训练集标注
    annotations = load_annotations(
        TRAIN_TXT
    )

    print("\n训练集标注示例：")

    for image_name, label in list(
            annotations.items()
    )[:5]:

        print(
            f"{image_name} -> {label}"
        )

    # 创建数据集和 DataLoader
    (
        train_dataset,
        valid_dataset,
        train_loader,
        valid_loader
    ) = create_dataloaders()

    print(
        f"\n训练集图片数量："
        f"{len(train_dataset)}"
    )

    print(
        f"验证集图片数量："
        f"{len(valid_dataset)}"
    )

    # 显示训练集和验证集样本
    show_one_sample(
        train_loader,
        "训练集样本"
    )

    show_one_sample(
        valid_loader,
        "验证集样本"
    )

    # 保存 DataLoader
    dataloaders = {

        "train": train_loader,

        "valid": valid_loader
    }

    # 创建 ResNet18
    model = build_model()

    # 设置优化器
    optimizer = optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )

    # 设置学习率衰减
    scheduler = optim.lr_scheduler.StepLR(
        optimizer,
        step_size=7,
        gamma=0.1
    )

    # 设置交叉熵损失函数
    criterion = nn.CrossEntropyLoss()

    # 开始训练
    (
        model,
        val_acc_history,
        train_acc_history,
        valid_losses,
        train_losses,
        learning_rates
    ) = train_model(

        model=model,

        dataloaders=dataloaders,

        criterion=criterion,

        optimizer=optimizer,

        scheduler=scheduler,

        device=device,

        num_epochs=NUM_EPOCHS,

        filename=BEST_MODEL_PATH
    )


# PyCharm 运行程序时从这里开始
if __name__ == "__main__":
    main()