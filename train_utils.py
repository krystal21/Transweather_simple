"""
训练工具函数
包含PSNR、SSIM计算、验证函数等
"""
import os
import time
import torch
import torch.nn.functional as F
from math import log10
import cv2
import numpy as np

# 兼容新旧版本的 scikit-image
try:
    from skimage.measure import compare_psnr, compare_ssim
    # 旧版本可能不支持data_range参数，创建包装函数
    _old_compare_psnr = compare_psnr
    _old_compare_ssim = compare_ssim

    def compare_psnr(im1, im2, data_range=None):
        """包装函数，兼容旧版本API"""
        # 旧版本函数可能不支持data_range参数，忽略它
        return _old_compare_psnr(im1, im2)

    def compare_ssim(im1, im2, data_range=None):
        """包装函数，兼容旧版本API"""
        # 旧版本函数可能不支持data_range参数，忽略它
        return _old_compare_ssim(im1, im2)
except ImportError:
    # 新版本 scikit-image 使用新的函数名
    from skimage.metrics import peak_signal_noise_ratio, structural_similarity

    # 创建兼容的包装函数
    def compare_psnr(im1, im2, data_range=None):
        """包装函数，保持与旧版本API兼容"""
        if data_range is None:
            # 自动检测数据范围（Y通道通常是0-255）
            data_range = max(im1.max() - im1.min(), im2.max() - im2.min())
            if data_range == 0:
                data_range = 255.0  # 默认值
        # 注意：新版本函数参数顺序是 (true, test)，但PSNR是对称的
        return peak_signal_noise_ratio(im1, im2, data_range=data_range)

    def compare_ssim(im1, im2, data_range=None):
        """包装函数，保持与旧版本API兼容"""
        if data_range is None:
            # 自动检测数据范围（Y通道通常是0-255）
            data_range = max(im1.max() - im1.min(), im2.max() - im2.min())
            if data_range == 0:
                data_range = 255.0  # 默认值
        return structural_similarity(im1, im2, data_range=data_range)


def calc_psnr(im1, im2):
    """计算PSNR（使用Y通道，支持batch）

    Args:
        im1: 模型输出，范围[-1, 1]，shape: [B, C, H, W]
        im2: GT图像，范围[0, 1]，shape: [B, C, H, W]

    Returns:
        PSNR列表，每个元素对应batch中的一张图像
    """
    # 将模型输出从[-1, 1]转换到[0, 1]
    im1 = (im1 + 1) / 2.0
    im1 = torch.clamp(im1, 0, 1)

    batch_size = im1.shape[0]
    psnr_list = []

    for i in range(batch_size):
        # 转换为numpy并调整维度 [C, H, W] -> [H, W, C]
        im1_np = im1[i].permute(1, 2, 0).detach().cpu().numpy()
        im2_np = im2[i].permute(1, 2, 0).detach().cpu().numpy()

        # 转换为0-255范围（uint8）
        im1_np = (im1_np * 255.0).astype(np.uint8)
        im2_np = (im2_np * 255.0).astype(np.uint8)

        # 转换为Y通道
        im1_y = cv2.cvtColor(im1_np, cv2.COLOR_RGB2YCR_CB)[:, :, 0]
        im2_y = cv2.cvtColor(im2_np, cv2.COLOR_RGB2YCR_CB)[:, :, 0]

        # 计算PSNR，Y通道值范围是0-255
        psnr = compare_psnr(im2_y, im1_y, data_range=255.0)
        psnr_list.append(psnr)

    return psnr_list


def calc_ssim(im1, im2):
    """计算SSIM（使用Y通道，支持batch）

    Args:
        im1: 模型输出，范围[-1, 1]，shape: [B, C, H, W]
        im2: GT图像，范围[0, 1]，shape: [B, C, H, W]

    Returns:
        SSIM列表，每个元素对应batch中的一张图像
    """
    # 将模型输出从[-1, 1]转换到[0, 1]
    im1 = (im1 + 1) / 2.0
    im1 = torch.clamp(im1, 0, 1)

    batch_size = im1.shape[0]
    ssim_list = []

    for i in range(batch_size):
        # 转换为numpy并调整维度 [C, H, W] -> [H, W, C]
        im1_np = im1[i].permute(1, 2, 0).detach().cpu().numpy()
        im2_np = im2[i].permute(1, 2, 0).detach().cpu().numpy()

        # 转换为0-255范围（uint8）
        im1_np = (im1_np * 255.0).astype(np.uint8)
        im2_np = (im2_np * 255.0).astype(np.uint8)

        # 转换为Y通道
        im1_y = cv2.cvtColor(im1_np, cv2.COLOR_RGB2YCR_CB)[:, :, 0]
        im2_y = cv2.cvtColor(im2_np, cv2.COLOR_RGB2YCR_CB)[:, :, 0]

        # 计算SSIM，Y通道值范围是0-255
        ssim = compare_ssim(im2_y, im1_y, data_range=255.0)
        ssim_list.append(ssim)

    return ssim_list


def to_psnr(pred_image, gt):
    """计算PSNR列表（训练时使用，与calc_psnr逻辑一致）

    Args:
        pred_image: 模型输出，范围[-1, 1]
        gt: GT图像，范围[0, 1]

    Returns:
        PSNR列表，每个元素对应batch中的一张图像
    """
    # 将模型输出从[-1, 1]转换到[0, 1]，与GT范围一致
    pred_image = (pred_image + 1) / 2.0
    pred_image = torch.clamp(pred_image, 0, 1)

    # 使用与calc_psnr相同的计算方式（RGB直接计算，不使用Y通道）
    # 这样可以保持训练和验证的一致性
    batch_size = pred_image.shape[0]
    psnr_list = []

    for i in range(batch_size):
        # 计算单张图像的MSE
        mse = F.mse_loss(pred_image[i], gt[i], reduction='mean').item()

        # 避免除零
        if mse == 0:
            psnr = float('inf')
        else:
            intensity_max = 1.0
            psnr = 10.0 * log10(intensity_max / mse)

        psnr_list.append(psnr)

    return psnr_list


def validation(net, val_data_loader, device):
    """验证函数

    Args:
        net: 网络模型
        val_data_loader: 验证数据加载器
        device: 设备

    Returns:
        avr_psnr: 平均PSNR
        avr_ssim: 平均SSIM
    """
    psnr_list = []
    ssim_list = []

    net.eval()
    with torch.no_grad():
        for batch_id, val_data in enumerate(val_data_loader):
            input_im, gt, _ = val_data
            input_im = input_im.to(device)
            gt = gt.to(device)
            pred_image = net(input_im)

            # 计算PSNR和SSIM
            psnr_list.extend(calc_psnr(pred_image, gt))
            ssim_list.extend(calc_ssim(pred_image, gt))

    avr_psnr = sum(psnr_list) / len(psnr_list)
    avr_ssim = sum(ssim_list) / len(ssim_list)
    return avr_psnr, avr_ssim


def print_log(epoch, num_epochs, one_epoch_time, train_psnr, val_psnr, val_ssim, exp_name):
    """打印训练日志"""
    print('({0:.0f}s) Epoch [{1}/{2}], Train_PSNR:{3:.2f}, Val_PSNR:{4:.2f}, Val_SSIM:{5:.4f}'
          .format(one_epoch_time, epoch, num_epochs, train_psnr, val_psnr, val_ssim))

    # 确保日志目录存在
    os.makedirs('./training_log', exist_ok=True)

    # 写入训练日志文件（追加模式，保存所有训练记录）
    log_file = './training_log/{}_log.txt'.format(exp_name)
    with open(log_file, 'a') as f:
        # 如果是第一个epoch，写入实验信息
        if epoch == 1:
            f.write('=' * 80 + '\n')
            f.write('实验名称: {}\n'.format(exp_name))
            f.write('开始时间: {}\n'.format(time.strftime(
                "%Y-%m-%d %H:%M:%S", time.localtime())))
            f.write('=' * 80 + '\n')

        print('Date: {0}, Time_Cost: {1:.0f}s, Epoch: [{2}/{3}], Train_PSNR: {4:.2f}, Val_PSNR: {5:.2f}, Val_SSIM: {6:.4f}'
              .format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
                      one_epoch_time, epoch, num_epochs, train_psnr, val_psnr, val_ssim), file=f)


def adjust_learning_rate(optimizer, epoch, lr_decay=0.5, step=100):
    """调整学习率

    Args:
        optimizer: 优化器
        epoch: 当前epoch
        lr_decay: 学习率衰减因子
        step: 每多少epoch衰减一次
    """
    if not epoch % step and epoch > 0:
        for param_group in optimizer.param_groups:
            param_group['lr'] *= lr_decay
            print('Learning rate sets to {}.'.format(param_group['lr']))
    else:
        for param_group in optimizer.param_groups:
            print('Learning rate sets to {}.'.format(param_group['lr']))
