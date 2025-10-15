#!/usr/bin/env python3
"""
检查配置文件与基础配置的兼容性
Usage: python check_config.py --config unimod1k_improved
"""
import os
import sys
import yaml
import argparse
from easydict import EasyDict as edict

prj_path = os.path.dirname(__file__)
if prj_path not in sys.path:
    sys.path.insert(0, prj_path)

from lib.config.spt.config import cfg as base_cfg


def check_config_compatibility(config_file):
    """检查配置文件中的所有参数是否在基础配置中定义"""

    print(f"\n{'='*80}")
    print(f"检查配置文件: {config_file}")
    print(f"{'='*80}\n")

    # 加载实验配置
    with open(config_file, 'r') as f:
        exp_cfg = edict(yaml.safe_load(f))

    # 递归检查
    missing_params = []

    def check_params(exp_dict, base_dict, path=""):
        if not isinstance(exp_dict, dict):
            return

        for key, value in exp_dict.items():
            current_path = f"{path}.{key}" if path else key

            if key not in base_dict:
                missing_params.append({
                    'path': current_path,
                    'value': value,
                    'type': type(value).__name__
                })
            elif isinstance(value, dict) and isinstance(base_dict[key], dict):
                check_params(value, base_dict[key], current_path)

    check_params(exp_cfg, base_cfg)

    if missing_params:
        print(f"❌ 发现 {len(missing_params)} 个缺失参数:\n")
        for param in missing_params:
            print(f"  - {param['path']}")
            print(f"    类型: {param['type']}, 值: {param['value']}")
        print(f"\n{'='*80}")
        print("需要在 lib/config/spt/config.py 中添加这些参数的默认值")
        print(f"{'='*80}\n")
        return False
    else:
        print(f"✅ 所有参数都已定义，配置文件兼容！\n")
        return True


def generate_fix_code(missing_params):
    """生成修复代码"""
    print("建议添加的代码:\n")
    print("```python")
    for param in missing_params:
        path_parts = param['path'].split('.')
        if len(path_parts) >= 2:
            section = '.'.join(path_parts[:-1])
            key = path_parts[-1]
            value = param['value']

            if isinstance(value, bool):
                value_str = str(value)
            elif isinstance(value, str):
                value_str = f"'{value}'"
            elif isinstance(value, list):
                value_str = str(value)
            else:
                value_str = str(value)

            print(f"cfg.{section}.{key} = {value_str}")
    print("```\n")


def main():
    parser = argparse.ArgumentParser(description='检查配置兼容性')
    parser.add_argument('--config', type=str, default='unimod1k_improved',
                       help='配置文件名（不含.yaml）')
    args = parser.parse_args()

    config_file = os.path.join(prj_path, f'experiments/spt/{args.config}.yaml')

    if not os.path.exists(config_file):
        print(f"错误: 配置文件不存在: {config_file}")
        sys.exit(1)

    compatible = check_config_compatibility(config_file)

    if compatible:
        print("✅ 可以开始训练！")
        print(f"\n推荐命令:")
        print(f"python train_improved.py --config {args.config} --save_dir ./checkpoints_improved")
    else:
        print("❌ 配置不兼容，请先修复 lib/config/spt/config.py")
        sys.exit(1)


if __name__ == '__main__':
    main()

