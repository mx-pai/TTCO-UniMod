#!/usr/bin/env python3
"""
SPT训练脚本 - 集成自动checkpoint清理功能
在训练过程中自动清理旧的checkpoint文件，节省磁盘空间
"""

import os
import sys
import subprocess
import threading
import time
import signal
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from auto_cleanup_checkpoints import CheckpointCleaner

class SPTTrainerWithCleanup:
    def __init__(self,
                 script_name='spt',
                 config_name='unimod1k',
                 checkpoint_dir='./checkpoints',
                 keep_checkpoints=5,
                 cleanup_interval=2.0):
        """
        Args:
            script_name: 训练脚本名称
            config_name: 配置文件名称
            checkpoint_dir: checkpoint保存目录
            keep_checkpoints: 保留的checkpoint数量
            cleanup_interval: 清理间隔(小时)
        """
        self.script_name = script_name
        self.config_name = config_name
        self.checkpoint_dir = Path(checkpoint_dir)
        self.keep_checkpoints = keep_checkpoints
        self.cleanup_interval = cleanup_interval

        # 创建清理器 - 指向实际的checkpoint保存目录
        actual_checkpoint_dir = self.checkpoint_dir / "checkpoints" / "train" / "spt" / "unimod1k"
        self.cleaner = CheckpointCleaner(
            checkpoint_dir=actual_checkpoint_dir,
            keep_last=keep_checkpoints,
            interval_hours=cleanup_interval
        )

        self.training_process = None
        self.cleanup_running = False

    def setup_signal_handlers(self):
        """设置信号处理器，确保优雅退出"""
        def signal_handler(signum, frame):
            print(f"\n🛑 收到信号 {signum}，正在停止训练和清理...")
            self.stop_training()
            sys.exit(0)

        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

    def start_training(self):
        """启动SPT训练"""
        print("🚀 启动SPT训练...")

        # 设置环境变量
        env = os.environ.copy()
        env['PYTHONPATH'] = f"{project_root}:{env.get('PYTHONPATH', '')}"

        # 构建训练命令 - 包含必要的参数
        train_cmd = [
            'python',
            'lib/train/run_training.py',
            '--script', self.script_name,
            '--config', self.config_name,
            '--save_dir', str(self.checkpoint_dir.parent)  # 使用父目录作为save_dir
        ]

        print(f"📋 训练命令: {' '.join(train_cmd)}")

        # 启动训练进程
        self.training_process = subprocess.Popen(
            train_cmd,
            env=env,
            cwd=project_root,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=1
        )

        # 启动输出监控线程
        def monitor_output():
            for line in self.training_process.stdout:
                print(line.rstrip())

        output_thread = threading.Thread(target=monitor_output, daemon=True)
        output_thread.start()

        return self.training_process

    def start_cleanup(self):
        """启动自动清理"""
        print("🧹 启动checkpoint自动清理...")
        self.cleaner.start_auto_cleanup()
        self.cleanup_running = True

    def stop_training(self):
        """停止训练和清理"""
        if self.training_process:
            print("🛑 停止训练进程...")
            self.training_process.terminate()
            try:
                self.training_process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                print("⚠️ 强制终止训练进程...")
                self.training_process.kill()

        if self.cleanup_running:
            print("🛑 停止清理服务...")
            self.cleaner.stop_auto_cleanup()
            self.cleanup_running = False

    def run(self):
        """运行训练和清理"""
        self.setup_signal_handlers()

        try:
            # 启动清理服务
            self.start_cleanup()

            # 启动训练
            process = self.start_training()

            # 等待训练完成
            return_code = process.wait()

            if return_code == 0:
                print("✅ 训练完成！")
            else:
                print(f"❌ 训练异常退出，返回码: {return_code}")

            return return_code

        except KeyboardInterrupt:
            print("\n⚠️ 用户中断训练")
            return 1
        except Exception as e:
            print(f"❌ 训练过程出错: {e}")
            return 1
        finally:
            self.stop_training()

def main():
    import argparse

    parser = argparse.ArgumentParser(description='SPT训练 - 集成自动清理')
    parser.add_argument('--script', type=str, default='spt',
                       help='训练脚本名称 (默认: spt)')
    parser.add_argument('--config', type=str, default='unimod1k',
                       help='配置文件名称 (默认: unimod1k)')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints',
                       help='Checkpoint保存目录 (默认: ./checkpoints)')
    parser.add_argument('--keep', type=int, default=5,
                       help='保留的checkpoint数量 (默认: 5)')
    parser.add_argument('--cleanup_interval', type=float, default=2.0,
                       help='清理间隔(小时) (默认: 2.0)')

    args = parser.parse_args()

    print("🎯 SPT训练器 - 集成自动清理")
    print("=" * 50)
    print(f"📝 脚本: {args.script}")
    print(f"⚙️ 配置: {args.config}")
    print(f"📁 Checkpoint目录: {args.checkpoint_dir}")
    print(f"🔄 保留checkpoint: {args.keep} 个")
    print(f"⏰ 清理间隔: {args.cleanup_interval} 小时")
    print("=" * 50)

    trainer = SPTTrainerWithCleanup(
        script_name=args.script,
        config_name=args.config,
        checkpoint_dir=args.checkpoint_dir,
        keep_checkpoints=args.keep,
        cleanup_interval=args.cleanup_interval
    )

    return trainer.run()

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)