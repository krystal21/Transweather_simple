"""
数据集定义
支持训练和验证数据集
"""
import torch.utils.data as data
from PIL import Image
from random import randrange
from torchvision.transforms import Compose, ToTensor, Normalize
import re
from PIL import ImageFile
import numpy as np

ImageFile.LOAD_TRUNCATED_IMAGES = True

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


class WeatherDataset(data.Dataset):
    """天气图像数据集
    
    支持训练和验证模式：
    - 训练模式：随机裁剪图像
    - 验证模式：保持原始尺寸，调整到16的倍数
    """
    def __init__(self, data_dir, filename, crop_size=None, mode='train'):
        """
        Args:
            data_dir: 数据目录路径
            filename: 包含图像列表的文件名
            crop_size: 训练时的裁剪尺寸 [width, height]，验证时为None
            mode: 'train' 或 'val'
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

    def __getitem__(self, index):
        if self.mode == 'train':
            return self._get_train_item(index)
        else:
            return self._get_val_item(index)

    def _get_train_item(self, index):
        """获取训练样本"""
        crop_width, crop_height = self.crop_size
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
            raise Exception(f'Bad image channel: {gt_name}')

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

