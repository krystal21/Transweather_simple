#!/bin/bash

# TransWeather 一键部署脚本
# 功能：自动检查环境、安装依赖、验证安装

set -e  # 遇到错误立即退出

echo "=========================================="
echo "TransWeather 一键部署脚本"
echo "=========================================="
echo ""

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 检查 Python 版本
echo -e "${YELLOW}[1/4] 检查 Python 环境...${NC}"
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}错误: 未找到 Python3，请先安装 Python 3.8 或更高版本${NC}"
    exit 1
fi

PYTHON_VERSION=$(python3 --version | cut -d' ' -f2 | cut -d'.' -f1,2)
REQUIRED_VERSION="3.8"

if [ "$(printf '%s\n' "$REQUIRED_VERSION" "$PYTHON_VERSION" | sort -V | head -n1)" != "$REQUIRED_VERSION" ]; then
    echo -e "${RED}错误: Python 版本需要 >= 3.8，当前版本: $PYTHON_VERSION${NC}"
    exit 1
fi

echo -e "${GREEN}✓ Python 版本: $PYTHON_VERSION${NC}"
echo ""

# 检查 pip
echo -e "${YELLOW}[2/4] 检查 pip...${NC}"
if ! command -v pip3 &> /dev/null; then
    echo -e "${RED}错误: 未找到 pip3，请先安装 pip${NC}"
    exit 1
fi
echo -e "${GREEN}✓ pip 已安装${NC}"
echo ""

# 询问是否创建虚拟环境
echo -e "${YELLOW}[3/4] 是否创建虚拟环境？(推荐)${NC}"
read -p "输入 y 创建虚拟环境，输入 n 跳过 (y/n): " -n 1 -r
echo ""

if [[ $REPLY =~ ^[Yy]$ ]]; then
    VENV_NAME="venv"
    if [ ! -d "$VENV_NAME" ]; then
        echo "创建虚拟环境: $VENV_NAME"
        python3 -m venv $VENV_NAME
        echo -e "${GREEN}✓ 虚拟环境创建成功${NC}"
    else
        echo -e "${YELLOW}虚拟环境已存在，跳过创建${NC}"
    fi
    
    echo "激活虚拟环境..."
    source $VENV_NAME/bin/activate
    echo -e "${GREEN}✓ 虚拟环境已激活${NC}"
    echo ""
fi

# 升级 pip
echo -e "${YELLOW}[4/4] 安装依赖包...${NC}"
echo "升级 pip..."
pip install --upgrade pip -q

# 检查是否有 requirements.txt
if [ -f "requirements.txt" ]; then
    echo "从 requirements.txt 安装依赖..."
    pip install -r requirements.txt
else
    echo "requirements.txt 不存在，安装基础依赖..."
    pip install torch torchvision timm opencv-python scikit-image numpy pillow
fi

echo ""
echo -e "${GREEN}✓ 依赖安装完成${NC}"
echo ""

# 验证安装
echo "=========================================="
echo "验证安装..."
echo "=========================================="

python3 -c "import torch; print(f'✓ PyTorch: {torch.__version__}')" 2>/dev/null || echo -e "${RED}✗ PyTorch 安装失败${NC}"
python3 -c "import torchvision; print(f'✓ TorchVision: {torchvision.__version__}')" 2>/dev/null || echo -e "${RED}✗ TorchVision 安装失败${NC}"
python3 -c "import timm; print(f'✓ timm: {timm.__version__}')" 2>/dev/null || echo -e "${RED}✗ timm 安装失败${NC}"
python3 -c "import cv2; print(f'✓ OpenCV: {cv2.__version__}')" 2>/dev/null || echo -e "${RED}✗ OpenCV 安装失败${NC}"
python3 -c "import skimage; print(f'✓ scikit-image 已安装')" 2>/dev/null || echo -e "${RED}✗ scikit-image 安装失败${NC}"
python3 -c "import numpy; print(f'✓ NumPy: {numpy.__version__}')" 2>/dev/null || echo -e "${RED}✗ NumPy 安装失败${NC}"
python3 -c "import PIL; print(f'✓ Pillow: {PIL.__version__}')" 2>/dev/null || echo -e "${RED}✗ Pillow 安装失败${NC}"

# 检查 CUDA（可选）
echo ""
if python3 -c "import torch; print('CUDA available:', torch.cuda.is_available())" 2>/dev/null; then
    CUDA_AVAILABLE=$(python3 -c "import torch; print(torch.cuda.is_available())" 2>/dev/null)
    if [ "$CUDA_AVAILABLE" = "True" ]; then
        CUDA_VERSION=$(python3 -c "import torch; print(torch.version.cuda)" 2>/dev/null)
        echo -e "${GREEN}✓ CUDA 可用 (版本: $CUDA_VERSION)${NC}"
    else
        echo -e "${YELLOW}⚠ CUDA 不可用，将使用 CPU 训练（会很慢）${NC}"
    fi
fi

echo ""
echo "=========================================="
echo -e "${GREEN}部署完成！${NC}"
echo "=========================================="
echo ""
echo "下一步："
echo "1. 准备数据：将数据放置在 data/train/ 和 data/test/ 目录"
echo "2. 开始训练："
echo "   python train_new.py --exp_name my_experiment --train_data_dir data/train/ --val_ratio 0.1 --gpu 0"
echo ""
if [[ $REPLY =~ ^[Yy]$ ]] && [ -d "$VENV_NAME" ]; then
    echo "注意：虚拟环境已激活。下次使用时请运行："
    echo "   source $VENV_NAME/bin/activate"
    echo ""
fi

