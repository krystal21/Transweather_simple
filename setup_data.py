"""
数据目录设置脚本
帮助用户快速设置训练和测试数据目录（使用符号链接避免复制数据）
"""
import os
import sys
from pathlib import Path


def setup_data_directories(allweather_dir='../data/allweather/', 
                          train_dir='../data/train/',
                          test_dir='../data/test/',
                          use_symlinks=True):
    """
    设置数据目录结构
    
    Args:
        allweather_dir: allweather数据目录路径
        train_dir: 训练数据目录路径
        test_dir: 测试数据目录路径
        use_symlinks: 是否使用符号链接（True）还是复制（False）
    """
    allweather_dir = os.path.abspath(allweather_dir)
    train_dir = os.path.abspath(train_dir)
    test_dir = os.path.abspath(test_dir)
    
    # 检查allweather目录是否存在
    if not os.path.exists(allweather_dir):
        print(f'错误: allweather目录不存在: {allweather_dir}')
        return False
    
    input_dir = os.path.join(allweather_dir, 'input')
    gt_dir = os.path.join(allweather_dir, 'gt')
    
    if not os.path.exists(input_dir):
        print(f'错误: input目录不存在: {input_dir}')
        return False
    
    if not os.path.exists(gt_dir):
        print(f'错误: gt目录不存在: {gt_dir}')
        return False
    
    print(f'数据源目录: {allweather_dir}')
    print(f'训练目录: {train_dir}')
    print(f'测试目录: {test_dir}')
    
    # 创建训练和测试目录
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)
    
    # 设置训练目录
    train_input = os.path.join(train_dir, 'input')
    train_gt = os.path.join(train_dir, 'gt')
    
    # 设置测试目录
    test_input = os.path.join(test_dir, 'input')
    test_gt = os.path.join(test_dir, 'gt')
    
    if use_symlinks:
        print('\n使用符号链接方式（推荐，节省磁盘空间）...')
        
        # 创建符号链接
        for src, dst in [(input_dir, train_input), (gt_dir, train_gt),
                        (input_dir, test_input), (gt_dir, test_gt)]:
            if os.path.exists(dst):
                if os.path.islink(dst):
                    print(f'  符号链接已存在: {dst}')
                else:
                    print(f'  警告: {dst} 已存在但不是符号链接，跳过')
            else:
                try:
                    os.symlink(src, dst)
                    print(f'  ✓ 创建符号链接: {dst} -> {src}')
                except OSError as e:
                    print(f'  ✗ 创建符号链接失败: {dst}, 错误: {e}')
                    return False
    else:
        print('\n使用复制方式（需要更多磁盘空间）...')
        print('  注意: 复制大量文件可能需要较长时间')
        response = input('  是否继续？(y/n): ')
        if response.lower() != 'y':
            print('  已取消')
            return False
        
        import shutil
        for src, dst in [(input_dir, train_input), (gt_dir, train_gt),
                        (input_dir, test_input), (gt_dir, test_gt)]:
            if os.path.exists(dst):
                print(f'  警告: {dst} 已存在，跳过复制')
            else:
                try:
                    shutil.copytree(src, dst)
                    print(f'  ✓ 复制完成: {dst}')
                except Exception as e:
                    print(f'  ✗ 复制失败: {dst}, 错误: {e}')
                    return False
    
    print('\n数据目录设置完成！')
    print(f'\n目录结构:')
    print(f'  {train_dir}/')
    print(f'    ├── input/ -> {input_dir}')
    print(f'    └── gt/ -> {gt_dir}')
    print(f'  {test_dir}/')
    print(f'    ├── input/ -> {input_dir}')
    print(f'    └── gt/ -> {gt_dir}')
    
    return True


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='设置训练和测试数据目录')
    parser.add_argument('--allweather_dir', type=str, default='../data/allweather/',
                       help='allweather数据目录路径')
    parser.add_argument('--train_dir', type=str, default='../data/train/',
                       help='训练数据目录路径')
    parser.add_argument('--test_dir', type=str, default='../data/test/',
                       help='测试数据目录路径')
    parser.add_argument('--copy', action='store_true',
                       help='使用复制而不是符号链接（默认使用符号链接）')
    
    args = parser.parse_args()
    
    success = setup_data_directories(
        args.allweather_dir,
        args.train_dir,
        args.test_dir,
        use_symlinks=not args.copy
    )
    
    if success:
        print('\n下一步: 运行拆分脚本生成训练和测试列表文件')
        print('  python split_dataset.py --input_file data/allweather/allweather.txt')
    else:
        sys.exit(1)


if __name__ == '__main__':
    main()

