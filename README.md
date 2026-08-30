# Python Learning Notes

这是我的 Python、NumPy、Matplotlib 和 PyTorch 学习记录。仓库只保留经过筛选的代码练习，用于记录从基础语法到计算机视觉模型训练的学习过程。

## 当前内容

| 目录 | 内容 | 能力点 |
| --- | --- | --- |
| `01_numpy` | 数组创建、索引、运算、图像批处理 | 维度、广播、归一化、NHWC/NCHW 转换 |
| `02_matplotlib` | 曲线、散点图、柱状图、图像和子图 | 实验结果可视化 |
| `03_pytorch_basics` | 线性回归、MNIST 多层感知机 | Tensor、`nn.Module`、损失函数、优化器、训练循环 |
| `04_computer_vision` | MNIST CNN、自定义 Dataset、ResNet18 | 卷积网络、DataLoader、训练与验证 |

## 环境安装

建议使用 Python 3.10 或 3.11，并在虚拟环境中安装依赖：

```bash
python -m venv .venv
.venv\Scripts\activate
python -m pip install -r requirements.txt
```

如果需要 GPU 版本的 PyTorch，请根据显卡和 CUDA 环境使用 [PyTorch 官方安装页面](https://pytorch.org/get-started/locally/)生成安装命令。

## 运行示例

```bash
python 01_numpy/05_image_batch_practice.py
python 03_pytorch_basics/01_linear_regression.py
python 04_computer_vision/01_mnist_cnn.py
```

`02_custom_dataset_resnet18.py` 需要自行准备花卉分类数据，目录约定如下：

```text
04_computer_vision/
└── flower_data/
    ├── train_filelist/
    ├── val_filelist/
    ├── train.txt
    └── val.txt
```

标注文件每行格式为 `image_001.jpg 0`。数据集和训练权重不会提交到仓库。

## 下一步

- 使用 GTSRB 完成 ResNet18 交通标志分类基线
- 增加 VGG16 和 MobileNetV2 对比实验
- 实现 FGSM、PGD 攻击与跨模型迁移评估
- 将正式项目整理到独立的求职作品仓库

## 说明

部分代码是在课程学习过程中完成并经过个人注释与整理的练习，不包含课程视频、课件、完整第三方项目、数据集或模型权重。本仓库用于学习记录，不把跟练代码表述为原创研究成果。
