# TransWeather: Multi-Weather Image Restoration with Enhanced Architecture

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.8+-orange.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

一个基于 Transformer 的多天气图像恢复项目，在原始 TransWeather 基础上进行了架构优化和数据增强改进。

## 📋 目录

- [项目简介](#项目简介)
- [项目特色](#项目特色)
- [原始项目](#原始项目)
- [改进内容](#改进内容)
- [项目结构](#项目结构)
- [快速开始](#快速开始)
- [详细使用](#详细使用)
- [实验结果](#实验结果)
- [依赖要求](#依赖要求)
- [许可证](#许可证)

## 🌟 项目简介

本项目实现了一个端到端的天气图像恢复系统，能够有效处理多种天气退化（如雨、雾、雪等）。项目在原始 TransWeather 架构基础上，提出了两种关键改进：

1. **多尺度层级融合**：使用 Selective Kernel Fusion (SK Fusion) 增强多尺度特征融合
2. **智能数据增强**：采用 Mosaic 和 Mixup 增强策略，提升模型泛化能力

## ✨ 项目特色

- 🔧 **模块化设计**：清晰的代码结构，易于理解和扩展
- 🚀 **高效训练**：支持多GPU并行训练，自动保存最佳模型
- 📊 **完整日志**：详细的训练日志记录，方便分析实验结果
- 🎯 **即插即用**：提供一键部署脚本，快速开始实验
- 🔬 **三种实现**：原始版本、多尺度融合、数据增强、以及两者结合版本

## 📖 原始项目

本项目基于 [TransWeather](https://github.com/jel-lambda/new-Transweather) 项目。

**TransWeather** 是一个基于 Transformer 架构的天气图像恢复网络，主要特点：
- 使用 Pyramid Vision Transformer (PVT) 作为编码器
- 采用多尺度特征提取策略
- 结合 Transformer 的长距离依赖建模能力

## 🔬 改进内容

### 1. 多尺度层级融合（Multi-Scale Feature Fusion）

**问题**：原始架构在不同尺度特征融合时采用简单的加法操作，可能无法充分利用多尺度信息。

**解决方案**：
- 引入 **Selective Kernel Fusion (SK Fusion)** 模块
- 在解码器的不同阶段动态融合 encoder 和 decoder 的特征
- 参考 HRNet 的设计思想，添加侧向连接（Lateral Connections）

**技术细节**：
```python
# SK Fusion 模块根据特征内容动态调整融合权重
class SelectiveKernelFusion(nn.Module):
    def forward(self, x_list):
        # 特征自适应融合
        fused = sum(x_resized)
        # 生成注意力权重
        y = self.fc(self.avg_pool(fused))
        return fused * y
```

**效果**：更好地保留细节信息，提升恢复质量。

### 2. 智能数据增强（Data Augmentation）

**问题**：训练数据可能不足以覆盖所有天气类型的组合，模型泛化能力有限。

**解决方案**：
- **Mosaic 增强**：将雨图和雾图拼接，强迫模型同时处理局部不同的天气
- **Mixup 增强**：按比例混合两张图像，增强模型的鲁棒性

**实现特点**：
- 可配置的增强概率（`mosaic_prob`, `mixup_prob`）
- 仅在训练时启用，验证/测试时关闭
- 保持图像和GT的对应关系

**效果**：提升模型对不同天气混合场景的处理能力。

## 📁 项目结构

```
transweather_simple/
├── model.py                  # 原始 TransWeather 模型
├── model_multiscale.py       # 多尺度融合模型（改进版本）
├── dataset.py                # 原始数据集类
├── dataset_aug.py            # 数据增强版本数据集（改进版本）
├── train_new.py              # 原始训练脚本
├── train_multiscale.py       # 多尺度融合训练脚本
├── train_aug.py              # 数据增强训练脚本
├── train_combined.py         # 两者结合训练脚本
├── test.py                   # 模型测试脚本
├── train_utils.py            # 训练工具函数（PSNR, SSIM等）
├── perceptual_loss.py        # 感知损失模块
├── base_networks.py          # 基础网络模块
├── split_dataset.py          # 数据集划分脚本
├── setup.sh                  # 一键部署脚本
├── requirements.txt          # 依赖列表
└── README.md                 # 本文档
```

## 🚀 快速开始

### 方式一：一键部署（推荐）

```bash
# 克隆项目
git clone <repository_url>
cd transweather_simple

# 运行一键部署脚本
chmod +x setup.sh
./setup.sh
```

一键部署脚本会自动：
1. 检查 Python 版本
2. 创建虚拟环境（可选）
3. 安装所有依赖
4. 验证安装是否成功

### 方式二：手动安装

#### 1. 环境要求

- Python >= 3.8
- CUDA >= 10.2 (GPU训练推荐)
- PyTorch >= 1.8.0

#### 2. 安装依赖

```bash
# 使用 pip
pip install -r requirements.txt

# 或使用 conda
conda install pytorch torchvision cudatoolkit=10.2 -c pytorch
pip install timm opencv-python scikit-image pillow numpy
```

#### 3. 准备数据

数据目录结构：
```
data/
├── train/
│   ├── input/          # 输入图像（有天气退化）
│   ├── gt/             # 真实图像（干净图像）
│   └── allfilter.txt   # 训练数据列表
└── test/
    ├── input/
    ├── gt/
    └── allfilter.txt
```

`allfilter.txt` 格式（每行一个相对路径）：
```
input/image1.jpg
input/image2.jpg
input/image3.jpg
```

## 📖 详细使用

### 训练模型

#### 1. 原始 TransWeather 模型

```bash
python train_new.py \
    --exp_name baseline \
    --train_data_dir data/train/ \
    --val_ratio 0.1 \
    --gpu 0 \
    --train_batch_size 128 \
    --num_epochs 200
```

#### 2. 多尺度融合版本

```bash
python train_multiscale.py \
    --exp_name multiscale \
    --train_data_dir data/train/ \
    --val_ratio 0.1 \
    --gpu 0 \
    --train_batch_size 128 \
    --num_epochs 200
```

#### 3. 数据增强版本

```bash
python train_aug.py \
    --exp_name aug \
    --train_data_dir data/train/ \
    --val_ratio 0.1 \
    --gpu 1 \
    --train_batch_size 128 \
    --num_epochs 200 \
    --mosaic_prob 0.5 \
    --mixup_prob 0.5
```

#### 4. 两者结合版本（推荐）

```bash
python train_combined.py \
    --exp_name combined \
    --train_data_dir data/train/ \
    --val_ratio 0.1 \
    --gpu 2 \
    --train_batch_size 128 \
    --num_epochs 200 \
    --mosaic_prob 0.5 \
    --mixup_prob 0.5
```

### 多GPU并行训练

如果有多张GPU，可以同时运行多个实验：

```bash
# GPU 0: 多尺度融合
python train_multiscale.py --exp_name multiscale --gpu 0 --train_data_dir data/train/ --val_ratio 0.1 --train_batch_size 128 --num_epochs 200

# GPU 1: 数据增强
python train_aug.py --exp_name aug --gpu 1 --train_data_dir data/train/ --val_ratio 0.1 --train_batch_size 128 --num_epochs 200 --mosaic_prob 0.5 --mixup_prob 0.5

# GPU 2: 两者结合
python train_combined.py --exp_name combined --gpu 2 --train_data_dir data/train/ --val_ratio 0.1 --train_batch_size 128 --num_epochs 200 --mosaic_prob 0.5 --mixup_prob 0.5
```

### 测试模型

#### 测试整个数据集

```bash
python test.py \
    --model_path ./weights/combined/best \
    --test_data_dir data/test/ \
    --test_filename allfilter.txt \
    --output_dir ./results/combined/
```

#### 测试单张图像

```bash
python test.py \
    --model_path ./weights/combined/best \
    --single_image data/test/input/test.jpg \
    --output_image results/test_restored.jpg
```

### 参数说明

**训练参数**：
- `--exp_name`: 实验名称（必需），用于保存模型和日志
- `--train_data_dir`: 训练数据目录
- `--val_ratio`: 验证集比例（默认 0.1）
- `--gpu`: GPU编号，如 "0", "1", "0,1"（多GPU）
- `--train_batch_size`: 训练批次大小（默认 18）
- `--num_epochs`: 训练轮数（默认 200）
- `--learning_rate`: 学习率（默认 2e-4）
- `--lambda_loss`: 感知损失权重（默认 0.04）

**数据增强参数**（仅 `train_aug.py` 和 `train_combined.py`）：
- `--mosaic_prob`: Mosaic增强概率（默认 0.5）
- `--mixup_prob`: Mixup增强概率（默认 0.5）

## 📊 实验结果

### 模型保存位置

- **模型权重**: `./weights/{exp_name}/best` 和 `./weights/{exp_name}/latest`
- **训练日志**: `./training_log/{exp_name}_log.txt`

### 日志格式

训练日志包含每个epoch的详细信息：
```
================================================================================
实验名称: combined
开始时间: 2024-01-01 10:00:00
================================================================================
Date: 2024-01-01 10:05:30, Time_Cost: 330s, Epoch: [1/200], Train_PSNR: 25.32, Val_PSNR: 26.45, Val_SSIM: 0.8234
Date: 2024-01-01 10:11:00, Time_Cost: 330s, Epoch: [2/200], Train_PSNR: 26.78, Val_PSNR: 27.12, Val_SSIM: 0.8456
...
```

### 性能对比

（根据实际实验结果填写）

| 方法                    | PSNR | SSIM | 参数量 | 训练时间 |
| ----------------------- | ---- | ---- | ------ | -------- |
| Baseline (TransWeather) | -    | -    | -      | -        |
| + Multi-Scale Fusion    | -    | -    | -      | -        |
| + Data Augmentation     | -    | -    | -      | -        |
| Combined                | -    | -    | -      | -        |

## 📦 依赖要求

### 必需依赖

```
torch >= 1.8.0
torchvision >= 0.9.0
timm >= 0.4.0
opencv-python >= 4.5.0
scikit-image >= 0.18.0
numpy >= 1.19.0
pillow >= 8.0.0
```

### 安装方式

创建 `requirements.txt`：
```bash
torch>=1.8.0
torchvision>=0.9.0
timm>=0.4.0
opencv-python>=4.5.0
scikit-image>=0.18.0
numpy>=1.19.0
pillow>=8.0.0
```

安装：
```bash
pip install -r requirements.txt
```

## 🔧 故障排除

### 常见问题

1. **CUDA out of memory**
   - 减小 `train_batch_size`
   - 使用梯度累积

2. **数据加载错误**
   - 检查数据路径是否正确
   - 确认 `allfilter.txt` 格式正确

3. **模型保存失败**
   - 检查磁盘空间
   - 确认 `weights/` 目录权限

## 📝 引用

如果本项目对您的研究有帮助，请引用：

```bibtex
@misc{transweather_improved,
  title={TransWeather: Multi-Weather Image Restoration with Enhanced Architecture},
  author={Your Name},
  year={2024},
  howpublished={\url{https://github.com/yourusername/transweather-improved}}
}
```

原始 TransWeather 项目：
```bibtex
@article{valanarasu2022transweather,
  title={TransWeather: Transformer-based Restoration of Images Degraded by Adverse Weather Conditions},
  author={Valanarasu, Jeya Maria Jose and Yasarla, Rajeev and Patel, Vishal M},
  journal={CVPR},
  year={2022}
}
```

## 📄 许可证

本项目基于 MIT 许可证开源。详见 [LICENSE](LICENSE) 文件。

## 🙏 致谢

- 感谢原始 [TransWeather](https://github.com/jel-lambda/new-Transweather) 项目的开源
- 感谢所有贡献者和用户的支持

## 📮 联系方式

如有问题或建议，欢迎：
- 提交 Issue
- 发送 Pull Request
- 联系作者：your.email@example.com

---

**注意**：本项目为课程作业提交版本，所有改进内容已在代码中详细注释说明。

