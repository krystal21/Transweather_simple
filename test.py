"""
Transweather测试脚本
用于加载训练好的模型进行图像恢复测试
"""
import os
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from PIL import Image
import numpy as np
from torchvision.transforms import Compose, ToTensor, Normalize

# 兼容新旧版本的PIL/Pillow
try:
    # Pillow 10.0+ 使用 Resampling
    from PIL.Image import Resampling
    ANTIALIAS = Resampling.LANCZOS
except ImportError:
    # 旧版本使用 LANCZOS
    try:
        ANTIALIAS = Image.LANCZOS
    except AttributeError:
        # 更旧的版本使用 ANTIALIAS
        ANTIALIAS = Image.ANTIALIAS

from model import Transweather
from dataset import WeatherDataset
from train_utils import calc_psnr, calc_ssim


def test_single_image(model_path, input_image_path, output_path, device):
    """测试单张图像

    Args:
        model_path: 模型权重路径
        input_image_path: 输入图像路径
        output_path: 输出图像保存路径
        device: 设备
    """
    # 加载模型
    net = Transweather()
    net.load_state_dict(torch.load(model_path, map_location=device))
    net = net.to(device)
    net.eval()

    # 加载和预处理图像
    input_img = Image.open(input_image_path).convert('RGB')

    # 调整图像尺寸为16的倍数
    wd_new, ht_new = input_img.size
    if ht_new > wd_new and ht_new > 1024:
        wd_new = int(np.ceil(wd_new * 1024 / ht_new))
        ht_new = 1024
    elif ht_new <= wd_new and wd_new > 1024:
        ht_new = int(np.ceil(ht_new * 1024 / wd_new))
        wd_new = 1024

    wd_new = int(16 * np.ceil(wd_new / 16.0))
    ht_new = int(16 * np.ceil(ht_new / 16.0))

    input_img = input_img.resize((wd_new, ht_new), ANTIALIAS)

    # 转换为tensor
    transform_input = Compose(
        [ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    input_tensor = transform_input(input_img).unsqueeze(0).to(device)

    # 推理
    with torch.no_grad():
        output_tensor = net(input_tensor)

    # 后处理：从[-1, 1]转换到[0, 1]
    output_tensor = (output_tensor + 1) / 2.0
    output_tensor = torch.clamp(output_tensor, 0, 1)

    # 转换为PIL图像并保存
    output_np = output_tensor[0].cpu().permute(1, 2, 0).numpy()
    output_np = (output_np * 255).astype(np.uint8)
    output_img = Image.fromarray(output_np)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    output_img.save(output_path)
    print(f'结果已保存到: {output_path}')


def test_dataset(model_path, test_data_dir, test_filename, output_dir, device):
    """测试整个数据集

    Args:
        model_path: 模型权重路径
        test_data_dir: 测试数据目录
        test_filename: 测试数据列表文件名
        output_dir: 输出目录
        device: 设备
    """
    # 加载模型
    net = Transweather()
    checkpoint = torch.load(model_path, map_location=device)

    # 处理DataParallel保存的模型
    if isinstance(checkpoint, dict) and 'module.' in str(checkpoint.keys()):
        new_state_dict = {}
        for k, v in checkpoint.items():
            name = k[7:] if k.startswith('module.') else k
            new_state_dict[name] = v
        net.load_state_dict(new_state_dict)
    else:
        net.load_state_dict(checkpoint)

    net = net.to(device)
    net.eval()

    # 创建测试数据集
    test_dataset = WeatherDataset(
        data_dir=test_data_dir,
        filename=test_filename,
        crop_size=None,
        mode='val'
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 计算指标
    psnr_list = []
    ssim_list = []

    print('开始测试...')
    with torch.no_grad():
        for batch_id, test_data in enumerate(test_loader):
            input_im, gt, input_name = test_data
            input_im = input_im.to(device)
            gt = gt.to(device)

            # 推理
            pred_image = net(input_im)

            # 计算PSNR和SSIM
            psnr_list.extend(calc_psnr(pred_image, gt))
            ssim_list.extend(calc_ssim(pred_image, gt))

            # 保存预测结果
            pred_image = (pred_image + 1) / 2.0
            pred_image = torch.clamp(pred_image, 0, 1)

            # 获取图像名称
            if isinstance(input_name, (list, tuple)):
                img_name = input_name[0]
            else:
                img_name = input_name

            # 提取文件名
            if isinstance(img_name, str):
                filename = os.path.basename(img_name)
            else:
                filename = f'result_{batch_id:04d}.jpg'

            # 保存图像
            pred_np = pred_image[0].cpu().permute(1, 2, 0).numpy()
            pred_np = (pred_np * 255).astype(np.uint8)
            pred_img = Image.fromarray(pred_np)
            pred_img.save(os.path.join(output_dir, filename))

            if (batch_id + 1) % 10 == 0:
                print(f'已处理: {batch_id + 1}/{len(test_loader)}')

    # 计算平均指标
    avg_psnr = sum(psnr_list) / len(psnr_list)
    avg_ssim = sum(ssim_list) / len(ssim_list)

    print('=' * 50)
    print('测试结果:')
    print(f'  平均PSNR: {avg_psnr:.2f} dB')
    print(f'  平均SSIM: {avg_ssim:.4f}')
    print(f'  测试样本数: {len(psnr_list)}')
    print(f'  结果保存在: {output_dir}')
    print('=' * 50)


def main():
    parser = argparse.ArgumentParser(description='Transweather测试脚本')
    parser.add_argument('--model_path', type=str, required=True, help='模型权重路径')
    parser.add_argument('--test_data_dir', type=str,
                        default='../data/test/', help='测试数据目录')
    parser.add_argument('--test_filename', type=str,
                        default='allfilter.txt', help='测试数据列表文件名')
    parser.add_argument('--output_dir', type=str,
                        default='./results/', help='输出目录')
    parser.add_argument('--single_image', type=str,
                        default=None, help='单张图像测试模式：输入图像路径')
    parser.add_argument('--output_image', type=str,
                        default=None, help='单张图像输出路径')

    args = parser.parse_args()

    # 设备设置
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f'使用设备: {device}')

    if args.single_image:
        # 单张图像测试模式
        if args.output_image is None:
            args.output_image = args.single_image.replace(
                '.jpg', '_restored.jpg').replace('.png', '_restored.png')
        test_single_image(args.model_path, args.single_image,
                          args.output_image, device)
    else:
        # 数据集测试模式
        test_dataset(args.model_path, args.test_data_dir,
                     args.test_filename, args.output_dir, device)


if __name__ == '__main__':
    main()
