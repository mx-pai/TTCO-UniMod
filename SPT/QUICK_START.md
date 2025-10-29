# 🚀 SPT 训练快速指南

> 当前仓库默认推荐直接使用 `experiments/spt/unimod1k.yaml`。若无需特殊调参，不需要额外配置。

---

## 1️⃣ 准备环境与路径

```bash
cd /root/autodl-tmp/UniMod1K/SPT
export PYTHONPATH=$(pwd):$PYTHONPATH
```

确认 `experiments/spt/unimod1k.yaml`（或 `unimod1k_improved.yaml`）里以下路径指向服务器实际位置：

```yaml
MODEL:
  PRETRAINED: '/root/autodl-tmp/STARKS_ep0500.pth.tar'
  LANGUAGE:
    PATH: '/root/autodl-tmp/bert/bert-base-uncased.tar.gz'
    VOCAB_PATH: '/root/autodl-tmp/bert/bert-base-uncased-vocab.txt'

PATHS:
  DATA_ROOT: '/root/autodl-tmp/data/1-训练验证集/TrainSet'
  NLP_ROOT:  '/root/autodl-tmp/data/1-训练验证集/TrainSet'
  OUTPUT_DIR: '/root/autodl-tmp/spt_runs'
```

---

## 2️⃣ 启动训练

### 标准脚本（保持与原论文一致）
```bash
python3 lib/train/run_training.py \
  --config unimod1k \
  --run_name baseline_$(date +%m%d_%H%M)
```

### 改进脚本（含长序列采样等增强）
```bash
python3 train_improved.py \
  --config unimod1k_improved \
  --run_name improved_$(date +%m%d_%H%M)
```

参数说明：
- `--run_name`：可选，默认为时间戳。用于区分不同实验目录。
- `--output_root`：可覆盖 `PATHS.OUTPUT_DIR`，按需将输出写到其他磁盘。

运行后会自动生成目录：  
`/root/autodl-tmp/spt_runs/<config>/<run_name>/`  
其中包含 `checkpoints/`, `logs/`, `tensorboard/`, `metadata/` 等子目录，并记录配置快照与 git 信息。

---

## 3️⃣ 监控训练

```bash
# 查看最新日志
tail -f /root/autodl-tmp/spt_runs/<config>/<run_name>/logs/*.log

# 查看 Loss / IoU
grep "Loss/total" /root/autodl-tmp/spt_runs/<config>/<run_name>/logs/*.log | tail
grep "IoU"        /root/autodl-tmp/spt_runs/<config>/<run_name>/logs/*.log | tail

# TensorBoard（如需）
tensorboard --logdir /root/autodl-tmp/spt_runs/<config>/<run_name>/tensorboard --port 6006

# GPU 监控
watch -n 1 nvidia-smi
```

---

## 4️⃣ 评测模型

1. 在配置文件中设置 `TEST.EPOCH` 为想要测试的 checkpoint 编号。  
2. 执行：
   ```bash
   python3 tracking/test.py \
     --tracker_name spt \
     --tracker_param unimod1k \
     --dataset_name unimod1k \
     --runid 1 \
     --threads 0 \
     --num_gpus 1
   ```
3. 结果位于 `lib/test/tracking_results/spt/<tracker_param>_001/`。

---

## 5️⃣ 清理旧实验

使用 `auto_clean.py` 可快速删除旧的 run，避免磁盘占满：

```bash
python3 auto_clean.py \
  --root /root/autodl-tmp/spt_runs \
  --keep 3 \
  --force
```

选项说明：
- `--config unimod1k_improved`：仅清理指定配置的 run。
- `--keep`：保留最新 N 个 run。
- 默认会先打印计划，只有加上 `--force` 才会真正删除。

---

## ✅ 常见问题排查

- **训练未写出日志或 checkpoint**：检查 `PATHS.OUTPUT_DIR` 与命令行参数，确认目标磁盘存在且可写。
- **找不到预训练模型或 BERT**：确保路径与文件名准确无误，并具有读取权限。
- **评测结果缺失**：确认 `TEST.EPOCH` 与实际存在的 checkpoint 编号一致。

如需调整训练策略（长序列比例、学习率等），可直接修改对应 YAML 中的参数，然后按上述流程重新启动即可。
