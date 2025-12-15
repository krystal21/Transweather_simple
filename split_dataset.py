"""
数据集拆分脚本
将allweather.txt拆分为训练集和测试集
"""
import os
import random
import argparse


def extract_base_id(line):
    """
    从文件路径中提取基础图像ID
    例如: input/im_0402_s90_a06.png -> im_0402
          input/city_read_14216.jpg -> city_read_14216
    """
    import re
    filename = os.path.basename(line)
    
    # 匹配 im_数字_ 格式
    match = re.match(r'^(im_\d+)', filename)
    if match:
        return match.group(1)
    
    # 其他格式，返回文件名（不含扩展名）
    return os.path.splitext(filename)[0]


def split_dataset(input_file, train_file, test_file, train_ratio=0.9, seed=19, group_by_base_id=True):
    """
    将数据集文件拆分为训练集和测试集
    
    如果group_by_base_id=True，会按基础图像ID分组，确保同一基础图像的所有变体都在同一集合中

    Args:
        input_file: 输入文件路径（allweather.txt）
        train_file: 训练集输出文件路径
        test_file: 测试集输出文件路径
        train_ratio: 训练集比例（默认0.9，即90%训练，10%测试）
        seed: 随机种子
        group_by_base_id: 是否按基础图像ID分组（默认True）
    """
    # 设置随机种子
    random.seed(seed)

    # 读取所有数据
    with open(input_file, 'r') as f:
        all_lines = [line.strip() for line in f.readlines() if line.strip()]

    print(f'总数据量: {len(all_lines)}')

    if group_by_base_id:
        # 按基础图像ID分组
        groups = {}
        for line in all_lines:
            base_id = extract_base_id(line)
            if base_id not in groups:
                groups[base_id] = []
            groups[base_id].append(line)
        
        print(f'基础图像组数: {len(groups)}')
        print(f'平均每组图像数: {len(all_lines) / len(groups):.1f}')
        
        # 将组打乱
        group_list = list(groups.items())
        random.shuffle(group_list)
        
        # 按组进行拆分
        split_point = int(len(group_list) * train_ratio)
        
        train_groups = group_list[:split_point]
        test_groups = group_list[split_point:]
        
        # 展开组内的所有图像
        train_lines = []
        for _, images in train_groups:
            train_lines.extend(images)
        
        test_lines = []
        for _, images in test_groups:
            test_lines.extend(images)
        
        print(f'训练组: {len(train_groups)} 组, 训练图像: {len(train_lines)} 条')
        print(f'测试组: {len(test_groups)} 组, 测试图像: {len(test_lines)} 条')
    else:
        # 原始方式：直接打乱所有数据
        random.shuffle(all_lines)
        
        # 计算分割点
        split_point = int(len(all_lines) * train_ratio)
        
        # 分割数据
        train_lines = all_lines[:split_point]
        test_lines = all_lines[split_point:]
        
        print(f'训练集: {len(train_lines)} 条')
        print(f'测试集: {len(test_lines)} 条')

    # 确保输出目录存在
    os.makedirs(os.path.dirname(train_file) if os.path.dirname(
        train_file) else '.', exist_ok=True)
    os.makedirs(os.path.dirname(test_file) if os.path.dirname(
        test_file) else '.', exist_ok=True)

    # 写入训练集
    with open(train_file, 'w') as f:
        for line in train_lines:
            f.write(line + '\n')

    # 写入测试集
    with open(test_file, 'w') as f:
        for line in test_lines:
            f.write(line + '\n')

    print(f'\n训练集已保存到: {train_file}')
    print(f'测试集已保存到: {test_file}')


def adjust_path_format(input_file, output_file):
    """
    调整路径格式，使其符合dataset.py的要求
    将 ./allweather/input/xxx.jpg 转换为 input/xxx.jpg

    Args:
        input_file: 输入文件路径
        output_file: 输出文件路径
    """
    with open(input_file, 'r') as f:
        lines = [line.strip() for line in f.readlines() if line.strip()]

    # 调整路径格式
    adjusted_lines = []
    for line in lines:
        # 移除开头的 ./
        if line.startswith('./'):
            line = line[2:]
        # 如果路径包含 allweather/，移除它
        if 'allweather/' in line:
            # 例如: ./allweather/input/xxx.jpg -> input/xxx.jpg
            line = line.replace('allweather/', '')

        adjusted_lines.append(line)

    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_file) if os.path.dirname(
        output_file) else '.', exist_ok=True)

    # 写入调整后的路径
    with open(output_file, 'w') as f:
        for line in adjusted_lines:
            f.write(line + '\n')

    print(f'路径格式已调整，保存到: {output_file}')


def main():
    parser = argparse.ArgumentParser(description='拆分数据集脚本')
    parser.add_argument('--input_file', type=str, default='data/allweather/allweather.txt',
                        help='输入文件路径')
    parser.add_argument('--train_file', type=str, default='data/train/allfilter.txt',
                        help='训练集输出文件路径')
    parser.add_argument('--test_file', type=str, default='data/test/allfilter.txt',
                        help='测试集输出文件路径')
    parser.add_argument('--train_ratio', type=float, default=0.9,
                        help='训练集比例（默认0.9）')
    parser.add_argument('--seed', type=int, default=19,
                        help='随机种子（默认19）')
    parser.add_argument('--adjust_path', action='store_true',
                        help='是否调整路径格式（移除./allweather/前缀）')
    parser.add_argument('--no_group', action='store_true',
                        help='不按基础图像ID分组（默认会分组以避免数据泄露）')

    args = parser.parse_args()

    # 检查输入文件是否存在
    if not os.path.exists(args.input_file):
        print(f'错误: 输入文件不存在: {args.input_file}')
        return

    # 先调整路径格式（如果需要）
    temp_train_file = args.train_file
    temp_test_file = args.test_file

    if args.adjust_path:
        print('=' * 50)
        print('先调整路径格式...')
        print('=' * 50)
        # 创建临时文件用于拆分
        temp_input = args.input_file + '.temp'
        adjust_path_format(args.input_file, temp_input)
        input_file_for_split = temp_input
    else:
        input_file_for_split = args.input_file

    # 拆分数据集
    print('\n' + '=' * 50)
    print('开始拆分数据集...')
    print('=' * 50)
    split_dataset(input_file_for_split, temp_train_file, temp_test_file,
                  args.train_ratio, args.seed, group_by_base_id=not args.no_group)

    # 清理临时文件
    if args.adjust_path and os.path.exists(input_file_for_split):
        os.remove(input_file_for_split)

    print('\n' + '=' * 50)
    print('拆分完成！')
    print('=' * 50)
    print(f'\n使用说明:')
    print(f'1. 训练数据目录: {os.path.dirname(args.train_file)}/')
    print(f'2. 测试数据目录: {os.path.dirname(args.test_file)}/')
    print(f'3. 训练时使用: --train_data_dir {os.path.dirname(args.train_file)}/')
    print(f'4. 测试时使用: --test_data_dir {os.path.dirname(args.test_file)}/')


if __name__ == '__main__':
    main()
