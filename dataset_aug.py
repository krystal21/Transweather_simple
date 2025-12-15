"""
数据集定义 - 数据增强版本
支持Mosaic和Mixup数据增强
"""
import torch.utils.data as data
from PIL import Image
from random import randrange, random, randint
from torchvision.transforms import Compose, ToTensor, Normalize
import re
from PIL import ImageFile
import numpy as np
import os

ImageFile.LOAD_TRUNCATED_IMAGES = True

# 兼容新旧版本的PIL/Pillow
try:
    from PIL.Image import Resampling
    ANTIALIAS = Resampling.LANCZOS
except ImportError:
    try:
        ANTIALIAS = Image.LANCZOS
    except AttributeError:
        ANTIALIAS = Image.ANTIALIAS


class WeatherDatasetAug(data.Dataset):
    """
    天气图像数据集 - 数据增强版本
    
    支持训练和验证模式：
    - 训练模式：随机裁剪图像 + Mosaic/Mixup增强
    - 验证模式：保持原始尺寸，调整到16的倍数
    """
    def __init__(self, data_dir, filename, crop_size=None, mode='train', 
                 use_mosaic=True, use_mixup=True, mosaic_prob=0.5, mixup_prob=0.5):
        """
        Args:
            data_dir: 数据目录路径
            filename: 包含图像列表的文件名
            crop_size: 训练时的裁剪尺寸 [width, height]，验证时为None
            mode: 'train' 或 'val'
            use_mosaic: 是否使用Mosaic增强
            use_mixup: 是否使用Mixup增强
            mosaic_prob: Mosaic增强的概率
            mixup_prob: Mixup增强的概率（在使用增强时）
        """
        super().__init__()
        data_list = data_dir + filename
        
        with open(data_list) as f:
            contents = f.readlines()
            input_names = [i.strip() for i in contents]
            gt_names = [i.strip().replace('input', 'gt') for i in input_names]

        self.input_names = input_names
        self.gt_names = gt_names
        self.crop_size = crop_size
        self.data_dir = data_dir
        self.mode = mode
        self.use_mosaic = use_mosaic and (mode == 'train')
        self.use_mixup = use_mixup and (mode == 'train')
        self.mosaic_prob = mosaic_prob
        self.mixup_prob = mixup_prob

    def __getitem__(self, index):
        if self.mode == 'train':
            return self._get_train_item(index)
        else:
            return self._get_val_item(index)

    def _mosaic_augmentation(self, img1_in, img1_gt, img2_in, img2_gt):
        """
        Mosaic增强：将两张图像拼接成一张
        将一张雨图和一张雾图拼接在一起
        """
        w1, h1 = img1_in.size
        w2, h2 = img2_in.size
        
        # 调整两张图像到相同尺寸（取较大尺寸）
        max_w, max_h = max(w1, w2), max(h1, h2)
        img1_in = img1_in.resize((max_w, max_h), ANTIALIAS)
        img1_gt = img1_gt.resize((max_w, max_h), ANTIALIAS)
        img2_in = img2_in.resize((max_w, max_h), ANTIALIAS)
        img2_gt = img2_gt.resize((max_w, max_h), ANTIALIAS)
        
        # 随机选择拼接方式：水平拼接或垂直拼接
        if random() > 0.5:
            # 水平拼接
            mosaic_in = Image.new('RGB', (max_w * 2, max_h))
            mosaic_gt = Image.new('RGB', (max_w * 2, max_h))
            mosaic_in.paste(img1_in, (0, 0))
            mosaic_in.paste(img2_in, (max_w, 0))
            mosaic_gt.paste(img1_gt, (0, 0))
            mosaic_gt.paste(img2_gt, (max_w, 0))
        else:
            # 垂直拼接
            mosaic_in = Image.new('RGB', (max_w, max_h * 2))
            mosaic_gt = Image.new('RGB', (max_w, max_h * 2))
            mosaic_in.paste(img1_in, (0, 0))
            mosaic_in.paste(img2_in, (0, max_h))
            mosaic_gt.paste(img1_gt, (0, 0))
            mosaic_gt.paste(img2_gt, (0, max_h))
        
        return mosaic_in, mosaic_gt

    def _mixup_augmentation(self, img1_in, img1_gt, img2_in, img2_gt, alpha=0.2):
        """
        Mixup增强：将两张图像按比例混合
        """
        # 调整到相同尺寸
        w1, h1 = img1_in.size
        w2, h2 = img2_in.size
        max_w, max_h = max(w1, w2), max(h1, h2)
        
        img1_in = img1_in.resize((max_w, max_h), ANTIALIAS)
        img1_gt = img1_gt.resize((max_w, max_h), ANTIALIAS)
        img2_in = img2_in.resize((max_w, max_h), ANTIALIAS)
        img2_gt = img2_gt.resize((max_w, max_h), ANTIALIAS)
        
        # 转换为numpy数组
        img1_in_arr = np.array(img1_in, dtype=np.float32)
        img1_gt_arr = np.array(img1_gt, dtype=np.float32)
        img2_in_arr = np.array(img2_in, dtype=np.float32)
        img2_gt_arr = np.array(img2_gt, dtype=np.float32)
        
        # 生成混合系数
        lam = np.random.beta(alpha, alpha)
        
        # 混合
        mixed_in = (lam * img1_in_arr + (1 - lam) * img2_in_arr).astype(np.uint8)
        mixed_gt = (lam * img1_gt_arr + (1 - lam) * img2_gt_arr).astype(np.uint8)
        
        return Image.fromarray(mixed_in), Image.fromarray(mixed_gt)

    def _get_train_item(self, index):
        """获取训练样本"""
        crop_width, crop_height = self.crop_size
        
        # 决定使用哪种增强
        use_aug = False
        aug_type = None
        
        if self.use_mosaic and random() < self.mosaic_prob:
            use_aug = True
            aug_type = 'mosaic'
        elif self.use_mixup and random() < self.mixup_prob:
            use_aug = True
            aug_type = 'mixup'
        
        if use_aug:
            # 随机选择另一张图像
            idx2 = randint(0, len(self.input_names) - 1)
            while idx2 == index:
                idx2 = randint(0, len(self.input_names) - 1)
            
            input_name1 = self.input_names[index]
            gt_name1 = self.gt_names[index]
            input_name2 = self.input_names[idx2]
            gt_name2 = self.gt_names[idx2]
            
            input_img1 = Image.open(self.data_dir + input_name1).convert('RGB')
            gt_img1 = Image.open(self.data_dir + gt_name1).convert('RGB')
            input_img2 = Image.open(self.data_dir + input_name2).convert('RGB')
            gt_img2 = Image.open(self.data_dir + gt_name2).convert('RGB')
            
            if aug_type == 'mosaic':
                input_img, gt_img = self._mosaic_augmentation(
                    input_img1, gt_img1, input_img2, gt_img2)
            else:  # mixup
                input_img, gt_img = self._mixup_augmentation(
                    input_img1, gt_img1, input_img2, gt_img2)
            
            img_id = re.split('/', input_name1)[-1][:-4]
        else:
            # 不使用增强，使用原始逻辑
            input_name = self.input_names[index]
            gt_name = self.gt_names[index]
            img_id = re.split('/', input_name)[-1][:-4]
            input_img = Image.open(self.data_dir + input_name).convert('RGB')
            gt_img = Image.open(self.data_dir + gt_name).convert('RGB')

        width, height = input_img.size

        # 如果图像太小，先resize
        if width < crop_width and height < crop_height:
            input_img = input_img.resize((crop_width, crop_height), ANTIALIAS)
            gt_img = gt_img.resize((crop_width, crop_height), ANTIALIAS)
        elif width < crop_width:
            input_img = input_img.resize((crop_width, height), ANTIALIAS)
            gt_img = gt_img.resize((crop_width, height), ANTIALIAS)
        elif height < crop_height:
            input_img = input_img.resize((width, crop_height), ANTIALIAS)
            gt_img = gt_img.resize((width, crop_height), ANTIALIAS)

        width, height = input_img.size

        # 随机裁剪
        x = randrange(0, width - crop_width + 1)
        y = randrange(0, height - crop_height + 1)
        input_crop_img = input_img.crop((x, y, x + crop_width, y + crop_height))
        gt_crop_img = gt_img.crop((x, y, x + crop_width, y + crop_height))

        # 转换为tensor
        transform_input = Compose([ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        transform_gt = Compose([ToTensor()])
        input_im = transform_input(input_crop_img)
        gt = transform_gt(gt_crop_img)

        # 检查通道数
        if input_im.shape[0] != 3 or gt.shape[0] != 3:
            raise Exception(f'Bad image channel: {img_id}')

        return input_im, gt, img_id

    def _get_val_item(self, index):
        """获取验证样本"""
        input_name = self.input_names[index]
        gt_name = self.gt_names[index]
        
        input_img = Image.open(self.data_dir + input_name).convert('RGB')
        gt_img = Image.open(self.data_dir + gt_name).convert('RGB')

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
        gt_img = gt_img.resize((wd_new, ht_new), ANTIALIAS)

        # 转换为tensor
        transform_input = Compose([ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        transform_gt = Compose([ToTensor()])
        input_im = transform_input(input_img)
        gt = transform_gt(gt_img)

        return input_im, gt, input_name

    def __len__(self):
        return len(self.input_names)

