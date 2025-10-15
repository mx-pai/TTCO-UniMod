#!/usr/bin/env python3
"""
简化版SPT训练+自动清理脚本
确保训练命令和清理功能都正确工作
"""

import os
import sys
import subprocess
import threading
import time
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from auto_cleanup_checkpoints import CheckpointCleaner

def start_spt_training():
    """启动SPT训练，使用正确的参数"""
    print("🚀 启动SPT训练...")

    # 设置环境变量
    env = os.environ.copy()
    env['PYTHONPATH'] = f"{project_root}:{env.get('PYTHONPATH', '')}"

    # SPT训练命令 - 包含所有必要参数
    train_cmd = [
        'python', 'lib/train/run_training.py',
        '--script', 'spt',
        '--config', 'unimod1k',
        '--save_dir', './'  # 当前目录，SPT会自动创建checkpoints子目录
    ]

    print(f"📋 训练命令: {' '.join(train_cmd)}")
    print(f"📁 工作目录: {project_root}")
    print(f"🌍 PYTHONPATH: {env['PYTHONPATH']}")

    # 启动训练进程
    process = subprocess.Popen(
        train_cmd,
        env=env,
        cwd=project_root,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True,
        bufsize=1
    )

    # 实时输出训练日志
    def monitor_output():
        try:
            for line in process.stdout:
                print(line.rstrip())
        except:
            pass

    output_thread = threading.Thread(target=monitor_output, daemon=True)
    output_thread.start()

    return process

def start_auto_cleanup(keep_checkpoints=5, cleanup_interval=2.0):
    """启动自动清理"""
    print("🧹 启动checkpoint自动清理...")

    # SPT的实际checkpoint保存路径
    checkpoint_dir = project_root / "checkpoints" / "train" / "spt" / "unimod1k"

    print(f"📁 监控目录: {checkpoint_dir}")

    cleaner = CheckpointCleaner(
        checkpoint_dir=checkpoint_dir,
        keep_last=keep_checkpoints,
        interval_hours=cleanup_interval
    )

    cleaner.start_auto_cleanup()
    return cleaner

def main():
    print("🎯 SPT训练器 - 简化版集成自动清理")
    print("=" * 60)

    # 确认当前目录
    current_dir = Path.cwd()
    expected_dir = Path("/root/autodl-tmp/UniMod1K/SPT")

    if current_dir != expected_dir:
        print(f"⚠️ 请切换到正确的目录: {expected_dir}")
        print(f"当前目录: {current_dir}")
        return 1

    # 检查checkpoint目录
    checkpoint_dir = current_dir / "checkpoints" / "train" / "spt" / "unimod1k"
    if checkpoint_dir.exists():
        checkpoint_files = list(checkpoint_dir.glob("SPT_ep*.pth.tar"))
        print(f"📁 找到 {len(checkpoint_files)} 个现有checkpoint文件")
        if checkpoint_files:
            latest = max(checkpoint_files, key=lambda x: int(x.stem.split('ep')[1]))
            epoch_num = int(latest.stem.split('ep')[1])
            print(f"🔄 将从epoch {epoch_num}继续训练")
    else:
        print("📁 checkpoint目录不存在，将从头开始训练")

    print("=" * 60)

    try:
        # 启动自动清理
        cleaner = start_auto_cleanup(keep_checkpoints=5, cleanup_interval=1.5)

        # 启动训练
        training_process = start_spt_training()

        print("\n🎮 训练已启动！")
        print("📊 自动清理每1.5小时运行一次，保留最新5个checkpoint")
        print("⚠️ 按 Ctrl+C 停止训练")

        # 等待训练完成
        return_code = training_process.wait()

        if return_code == 0:
            print("\n✅ 训练完成！")
        else:
            print(f"\n❌ 训练异常退出，返回码: {return_code}")

        return return_code

    except KeyboardInterrupt:
        print("\n⚠️ 用户中断训练")
        if 'training_process' in locals():
            training_process.terminate()
        if 'cleaner' in locals():
            cleaner.stop_auto_cleanup()
        return 1
    except Exception as e:
        print(f"\n❌ 训练过程出错: {e}")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)