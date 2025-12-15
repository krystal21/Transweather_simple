"""
Transweather训练脚本 - 数据增强版本
使用Mosaic和Mixup数据增强进行训练
"""
import os
import time
import argparse
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torchvision.models import vgg16, VGG16_Weights

from model import Transweather
from dataset_aug import WeatherDatasetAug
from train_utils import to_psnr, print_log, validation, adjust_learning_rate
from perceptual_loss import PerceptualLoss
import re


def set_seed(seed):
    """设置随机种子"""
    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        random.seed(seed)
        print(f'随机种子设置为: {seed}')


def extract_base_id(line):
    """
    从文件路径中提取基础图像ID
    例如: input/im_0402_s90_a06.png -> im_0402
          input/city_read_14216.jpg -> city_read_14216
    """
    filename = os.path.basename(line)

    # 匹配 im_数字_ 格式
    match = re.match(r'^(im_\d+)', filename)
    if match:
        return match.group(1)

    # 其他格式，返回文件名（不含扩展名）
    return os.path.splitext(filename)[0]


def main():
    # 解析参数
    parser = argparse.ArgumentParser(description='Transweather训练脚本 - 数据增强版本')
    parser.add_argument('--learning_rate', type=float,
                        default=2e-4, help='学习率')
    parser.add_argument('--crop_size', type=int, nargs='+',
                        default=[256, 256], help='训练时裁剪尺寸')
    parser.add_argument('--train_batch_size', type=int,
                        default=18, help='训练批次大小')
    parser.add_argument('--val_batch_size', type=int, default=1, help='验证批次大小')
    parser.add_argument('--num_epochs', type=int, default=200, help='训练轮数')
    parser.add_argument('--lambda_loss', type=float,
                        default=0.04, help='感知损失权重')
    parser.add_argument('--exp_name', type=str,
                        required=True, help='实验名称（用于保存模型和日志）')
    parser.add_argument('--seed', type=int, default=19, help='随机种子')
    parser.add_argument('--train_data_dir', type=str,
                        default='../data/train/', help='训练数据目录')
    parser.add_argument('--val_data_dir', type=str,
                        default=None, help='验证数据目录（如果为None，则从训练集中划分）')
    parser.add_argument('--train_filename', type=str,
                        default='allfilter.txt', help='训练数据列表文件名')
    parser.add_argument('--val_filename', type=str, default='allfilter.txt',
                        help='验证数据列表文件名（仅在val_data_dir不为None时使用）')
    parser.add_argument('--val_ratio', type=float,
                        default=0.1, help='从训练集中划分验证集的比例（默认0.1，即10%%）')
    parser.add_argument('--resume', type=str, default=None, help='恢复训练的模型路径')
    parser.add_argument('--gpu', type=str, default=None,
                        help='指定GPU设备，例如: "0" 或 "0,1" 或 "cuda:0"。如果为None，则自动使用所有可用GPU')
    parser.add_argument('--mosaic_prob', type=float, default=0.5, help='Mosaic增强概率')
    parser.add_argument('--mixup_prob', type=float, default=0.5, help='Mixup增强概率')

    args = parser.parse_args()

    # 设置随机种子
    set_seed(args.seed)

    # 打印超参数
    print('=' * 50)
    print('训练超参数 (数据增强版本):')
    print(f'  学习率: {args.learning_rate}')
    print(f'  裁剪尺寸: {args.crop_size}')
    print(f'  训练批次大小: {args.train_batch_size}')
    print(f'  验证批次大小: {args.val_batch_size}')
    print(f'  训练轮数: {args.num_epochs}')
    print(f'  感知损失权重: {args.lambda_loss}')
    print(f'  实验名称: {args.exp_name}')
    print(f'  Mosaic增强概率: {args.mosaic_prob}')
    print(f'  Mixup增强概率: {args.mixup_prob}')
    print('=' * 50)

    # 创建必要的目录
    os.makedirs(f'./weights/{args.exp_name}/', exist_ok=True)
    os.makedirs('./training_log/', exist_ok=True)

    # 设备设置
    if args.gpu is not None:
        if args.gpu.startswith('cuda:'):
            device = torch.device(args.gpu)
            device_ids = [int(args.gpu.split(':')[1])]
        elif ',' in args.gpu:
            device_ids = [int(x.strip()) for x in args.gpu.split(',')]
            device = torch.device(f"cuda:{device_ids[0]}")
        else:
            device_ids = [int(args.gpu)]
            device = torch.device(f"cuda:{device_ids[0]}")
        print(f'使用指定GPU: {device_ids}')
    else:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        device_ids = [Id for Id in range(torch.cuda.device_count())]
        if len(device_ids) > 0:
            print(f'自动检测到 {len(device_ids)} 个GPU: {device_ids}')
        else:
            print('未检测到GPU，使用CPU')

    print(f'使用设备: {device}')

    # 创建模型
    net = Transweather()
    net = net.to(device)
    if len(device_ids) > 1:
        net = nn.DataParallel(net, device_ids=device_ids)
        print(f'使用多GPU训练: {device_ids}')

    # 创建优化器
    optimizer = torch.optim.Adam(net.parameters(), lr=args.learning_rate)

    # 加载预训练权重（如果指定）
    if args.resume:
        if os.path.exists(args.resume):
            if isinstance(net, nn.DataParallel):
                net.module.load_state_dict(torch.load(args.resume))
            else:
                net.load_state_dict(torch.load(args.resume))
            print(f'已加载预训练权重: {args.resume}')
        else:
            print(f'警告: 预训练权重文件不存在: {args.resume}')

    # 创建感知损失网络
    try:
        vgg_model = vgg16(weights=VGG16_Weights.IMAGENET1K_V1).features[:16]
    except TypeError:
        vgg_model = vgg16(pretrained=True).features[:16]
    vgg_model = vgg_model.to(device)
    for param in vgg_model.parameters():
        param.requires_grad = False
    loss_network = PerceptualLoss(vgg_model)
    loss_network.eval()

    # 创建数据加载器（使用数据增强版本）
    if args.val_data_dir is None:
        # 从训练集中划分验证集
        print(f'从训练集中划分 {args.val_ratio*100:.1f}% 作为验证集...')

        train_list_file = os.path.join(
            args.train_data_dir, args.train_filename)
        with open(train_list_file, 'r') as f:
            all_lines = [line.strip()
                         for line in f.readlines() if line.strip()]

        # 按基础图像ID分组，避免数据泄露
        groups = {}
        for line in all_lines:
            base_id = extract_base_id(line)
            if base_id not in groups:
                groups[base_id] = []
            groups[base_id].append(line)

        print(f'基础图像组数: {len(groups)}')
        print(f'平均每组图像数: {len(all_lines) / len(groups):.1f}')

        # 将组打乱并按组划分
        random.seed(args.seed)
        group_list = list(groups.items())
        random.shuffle(group_list)

        total_groups = len(group_list)
        val_groups_count = int(total_groups * args.val_ratio)
        train_groups = group_list[:-
                                  val_groups_count] if val_groups_count > 0 else group_list
        val_groups = group_list[-val_groups_count:] if val_groups_count > 0 else []

        train_lines = []
        for _, images in train_groups:
            train_lines.extend(images)

        val_lines = []
        for _, images in val_groups:
            val_lines.extend(images)

        train_size = len(train_lines)
        val_size = len(val_lines)

        print(f'训练组: {len(train_groups)} 组, 训练图像: {train_size} 样本')
        print(f'验证组: {len(val_groups)} 组, 验证图像: {val_size} 样本')

        train_temp_file = os.path.join(
            args.train_data_dir, f'train_temp_{args.exp_name}.txt')
        val_temp_file = os.path.join(
            args.train_data_dir, f'val_temp_{args.exp_name}.txt')

        with open(train_temp_file, 'w') as f:
            f.write('\n'.join(train_lines))
        with open(val_temp_file, 'w') as f:
            f.write('\n'.join(val_lines))

        train_dataset = WeatherDatasetAug(
            data_dir=args.train_data_dir,
            filename=os.path.basename(train_temp_file),
            crop_size=args.crop_size,
            mode='train',
            use_mosaic=True,
            use_mixup=True,
            mosaic_prob=args.mosaic_prob,
            mixup_prob=args.mixup_prob
        )

        val_dataset = WeatherDatasetAug(
            data_dir=args.train_data_dir,
            filename=os.path.basename(val_temp_file),
            crop_size=None,
            mode='val',
            use_mosaic=False,
            use_mixup=False
        )

    else:
        train_dataset = WeatherDatasetAug(
            data_dir=args.train_data_dir,
            filename=args.train_filename,
            crop_size=args.crop_size,
            mode='train',
            use_mosaic=True,
            use_mixup=True,
            mosaic_prob=args.mosaic_prob,
            mixup_prob=args.mixup_prob
        )

        val_dataset = WeatherDatasetAug(
            data_dir=args.val_data_dir,
            filename=args.val_filename,
            crop_size=None,
            mode='val',
            use_mosaic=False,
            use_mixup=False
        )
        print(f'训练样本数: {len(train_dataset)}, 验证样本数: {len(val_dataset)}')

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        num_workers=8,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.val_batch_size,
        shuffle=False,
        num_workers=8,
        pin_memory=True
    )

    # 初始验证
    print('\n进行初始验证...')
    net.eval()
    old_val_psnr, old_val_ssim = validation(net, val_loader, device)
    print(f'初始验证结果 - PSNR: {old_val_psnr:.2f}, SSIM: {old_val_ssim:.4f}\n')

    # 开始训练
    net.train()
    for epoch in range(args.num_epochs):
        psnr_list = []
        start_time = time.time()

        # 调整学习率
        adjust_learning_rate(optimizer, epoch)

        # 训练一个epoch
        for batch_id, train_data in enumerate(train_loader):
            input_image, gt, _ = train_data
            input_image = input_image.to(device)
            gt = gt.to(device)

            # 前向传播
            optimizer.zero_grad()
            net.train()
            pred_image = net(input_image)

            # 计算损失
            pred_image_normalized = (pred_image + 1) / 2.0
            pred_image_normalized = torch.clamp(pred_image_normalized, 0, 1)

            smooth_loss = F.smooth_l1_loss(pred_image_normalized, gt)
            perceptual_loss = loss_network(pred_image_normalized, gt)
            loss = smooth_loss + args.lambda_loss * perceptual_loss

            # 反向传播
            loss.backward()
            optimizer.step()

            # 计算PSNR
            psnr_list.extend(to_psnr(pred_image, gt))

            # 打印进度
            if (batch_id + 1) % 100 == 0:
                print(f'Epoch: {epoch+1}/{args.num_epochs}, Batch: {batch_id+1}/{len(train_loader)}, '
                      f'Loss: {loss.item():.4f}')

        # 计算平均训练PSNR
        train_psnr = sum(psnr_list) / len(psnr_list)

        # 保存最新模型
        if isinstance(net, nn.DataParallel):
            torch.save(net.module.state_dict(),
                       f'./weights/{args.exp_name}/latest')
        else:
            torch.save(net.state_dict(), f'./weights/{args.exp_name}/latest')

        # 验证
        val_psnr, val_ssim = validation(net, val_loader, device)

        # 打印日志
        one_epoch_time = time.time() - start_time
        print_log(epoch + 1, args.num_epochs, one_epoch_time,
                  train_psnr, val_psnr, val_ssim, args.exp_name)

        # 保存最佳模型
        if val_psnr >= old_val_psnr:
            if isinstance(net, nn.DataParallel):
                torch.save(net.module.state_dict(),
                           f'./weights/{args.exp_name}/best')
            else:
                torch.save(net.state_dict(), f'./weights/{args.exp_name}/best')
            print(f'模型已保存 (PSNR提升: {val_psnr - old_val_psnr:.2f})')
            old_val_psnr = val_psnr

        print('-' * 50)

    print('训练完成!')

    # 清理临时文件
    if args.val_data_dir is None:
        train_temp_file = os.path.join(
            args.train_data_dir, f'train_temp_{args.exp_name}.txt')
        val_temp_file = os.path.join(
            args.train_data_dir, f'val_temp_{args.exp_name}.txt')
        if os.path.exists(train_temp_file):
            os.remove(train_temp_file)
        if os.path.exists(val_temp_file):
            os.remove(val_temp_file)


if __name__ == '__main__':
    main()

