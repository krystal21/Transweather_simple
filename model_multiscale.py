"""
Transweather模型定义 - 多尺度层级融合版本
基于Transformer的天气图像恢复网络，增强多尺度特征融合
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
import math
import timm
from timm.models.layers import DropPath, to_2tuple, trunc_normal_

from base_networks import ConvLayer, UpsampleConvLayer, ResidualBlock
# 导入原始模型的所有组件（避免循环导入，使用import model然后引用）
import model as base_model


class SelectiveKernelFusion(nn.Module):
    """选择性核融合模块 (Selective Kernel Fusion)
    用于动态融合不同尺度的特征
    """

    def __init__(self, channels, reduction=16):
        super(SelectiveKernelFusion, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x_list):
        """
        Args:
            x_list: 不同尺度的特征列表，需要先上采样/下采样到相同尺寸
        Returns:
            融合后的特征
        """
        # 将所有特征上采样到最大尺寸
        max_h, max_w = max([x.shape[2] for x in x_list]), max(
            [x.shape[3] for x in x_list])
        x_resized = []
        for x in x_list:
            if x.shape[2] != max_h or x.shape[3] != max_w:
                x = F.interpolate(x, size=(max_h, max_w),
                                  mode='bilinear', align_corners=False)
            x_resized.append(x)

        # 特征求和
        fused = sum(x_resized)

        # 生成注意力权重
        b, c, _, _ = fused.size()
        y = self.avg_pool(fused).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)

        # 应用注意力权重
        return fused * y.expand_as(fused)


class MultiScaleConvProjection(nn.Module):
    """多尺度卷积投影模块，增强特征融合"""

    def __init__(self, path=None, **kwargs):
        super(MultiScaleConvProjection, self).__init__()

        self.convd32x = UpsampleConvLayer(512, 512, kernel_size=4, stride=2)
        self.convd16x = UpsampleConvLayer(512, 320, kernel_size=4, stride=2)
        self.dense_4 = nn.Sequential(ResidualBlock(320))
        self.convd8x = UpsampleConvLayer(320, 128, kernel_size=4, stride=2)
        self.dense_3 = nn.Sequential(ResidualBlock(128))
        self.convd4x = UpsampleConvLayer(128, 64, kernel_size=4, stride=2)
        self.dense_2 = nn.Sequential(ResidualBlock(64))
        self.convd2x = UpsampleConvLayer(64, 16, kernel_size=4, stride=2)
        self.dense_1 = nn.Sequential(ResidualBlock(16))
        self.convd1x = UpsampleConvLayer(16, 8, kernel_size=4, stride=2)

        # 多尺度融合模块 (SK Fusion)
        # 在融合不同stage特征时使用
        # 注意：通道数应该匹配融合时的输入特征通道数
        self.sk_fusion_16x = SelectiveKernelFusion(512)  # res32x和x1[3]都是512通道
        self.sk_fusion_8x = SelectiveKernelFusion(320)   # res8x和lateral_3都是320通道
        self.sk_fusion_4x = SelectiveKernelFusion(128)   # res4x和lateral_2都是128通道
        self.sk_fusion_2x = SelectiveKernelFusion(64)    # res2x和lateral_1都是64通道

        # 侧向连接：从encoder的不同stage直接连接到decoder
        # 使用1x1卷积调整通道数
        self.lateral_conv_3 = ConvLayer(
            320, 320, kernel_size=1, stride=1, padding=0)
        self.lateral_conv_2 = ConvLayer(
            128, 128, kernel_size=1, stride=1, padding=0)
        self.lateral_conv_1 = ConvLayer(
            64, 64, kernel_size=1, stride=1, padding=0)

    def forward(self, x1, x2):
        """
        Args:
            x1: encoder输出的多尺度特征列表 [stage0, stage1, stage2, stage3]
            x2: decoder输出的特征列表
        """
        # Stage 4 -> Stage 3 (16x)
        res32x = self.convd32x(x2[0])

        # 对齐尺寸
        if x1[3].shape[3] != res32x.shape[3] and x1[3].shape[2] != res32x.shape[2]:
            p2d = (0, -1, 0, -1)
            res32x = F.pad(res32x, p2d, "constant", 0)
        elif x1[3].shape[3] != res32x.shape[3] and x1[3].shape[2] == res32x.shape[2]:
            p2d = (0, -1, 0, 0)
            res32x = F.pad(res32x, p2d, "constant", 0)
        elif x1[3].shape[3] == res32x.shape[3] and x1[3].shape[2] != res32x.shape[2]:
            p2d = (0, 0, 0, -1)
            res32x = F.pad(res32x, p2d, "constant", 0)

        # 使用SK Fusion融合decoder输出和encoder stage3特征
        res16x = self.sk_fusion_16x([res32x, x1[3]])
        res16x = self.convd16x(res16x)

        # Stage 3 -> Stage 2 (8x)
        if x1[2].shape[3] != res16x.shape[3] and x1[2].shape[2] != res16x.shape[2]:
            p2d = (0, -1, 0, -1)
            res16x = F.pad(res16x, p2d, "constant", 0)
        elif x1[2].shape[3] != res16x.shape[3] and x1[2].shape[2] == res16x.shape[2]:
            p2d = (0, -1, 0, 0)
            res16x = F.pad(res16x, p2d, "constant", 0)
        elif x1[2].shape[3] == res16x.shape[3] and x1[2].shape[2] != res16x.shape[2]:
            p2d = (0, 0, 0, -1)
            res16x = F.pad(res16x, p2d, "constant", 0)

        # 处理侧向连接
        lateral_3 = self.lateral_conv_3(x1[2])
        res8x = self.dense_4(res16x)
        # SK Fusion融合
        res8x = self.sk_fusion_8x([res8x, lateral_3])
        res8x = self.convd8x(res8x)

        # Stage 2 -> Stage 1 (4x)
        lateral_2 = self.lateral_conv_2(x1[1])
        res4x = self.dense_3(res8x)
        # SK Fusion融合
        res4x = self.sk_fusion_4x([res4x, lateral_2])
        res4x = self.convd4x(res4x)

        # Stage 1 -> Stage 0 (2x)
        lateral_1 = self.lateral_conv_1(x1[0])
        res2x = self.dense_2(res4x)
        # SK Fusion融合
        res2x = self.sk_fusion_2x([res2x, lateral_1])
        res2x = self.convd2x(res2x)

        x = res2x
        x = self.dense_1(x)
        x = self.convd1x(x)

        return x


class TransweatherMultiScale(nn.Module):
    """Transweather主网络 - 多尺度融合版本"""

    def __init__(self, path=None, **kwargs):
        super(TransweatherMultiScale, self).__init__()

        self.Tenc = base_model.Tenc()
        self.Tdec = base_model.Tdec()
        self.convtail = MultiScaleConvProjection()
        self.clean = ConvLayer(8, 3, kernel_size=3, stride=1, padding=1)
        self.active = nn.Tanh()

        if path is not None:
            self.load(path)

    def forward(self, x):
        x1 = self.Tenc(x)
        x2 = self.Tdec(x1)
        x = self.convtail(x1, x2)
        clean = self.active(self.clean(x))
        return clean

    def load(self, path):
        """加载预训练权重"""
        checkpoint = torch.load(
            path, map_location=lambda storage, loc: storage)
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint

        # 移除module.前缀（如果存在）
        new_state_dict = {}
        for k, v in state_dict.items():
            name = k[7:] if k.startswith('module.') else k
            # 适配多尺度版本：跳过新增的SK Fusion和lateral_conv层
            if 'sk_fusion' in name or 'lateral_conv' in name:
                continue
            new_state_dict[name] = v

        self.load_state_dict(new_state_dict, strict=False)
        del checkpoint
        torch.cuda.empty_cache()
