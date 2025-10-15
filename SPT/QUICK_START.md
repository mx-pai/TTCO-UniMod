# 🚀 SPT 改进版训练 - 快速开始指南

## 📋 一键启动（推荐）

### 最简单的方式
```bash
cd /root/autodl-tmp/UniMod1K/SPT

# 给脚本执行权限
chmod +x launch_training.sh

# 一键启动（自动检查环境+训练）
bash launch_training.sh
```

### 高级选项
```bash
# 指定配置文件
bash launch_training.sh --config unimod1k_improved

# 启用自动评测（在epoch 40/80/120/160/200/240自动评测）
bash launch_training.sh --auto-eval

# 指定保存目录
bash launch_training.sh --save-dir ./my_checkpoints

# 保留最近10个checkpoint（默认5个）
bash launch_training.sh --keep-ckpt 10

# 组合使用
bash launch_training.sh --config unimod1k_improved --auto-eval --keep-ckpt 10
```

---

## 🔧 手动启动（更灵活）

### 步骤1: 准备环境
```bash
cd /root/autodl-tmp/UniMod1K/SPT
export PYTHONPATH=$(pwd):$PYTHONPATH
```

### 步骤2: 更新配置文件路径
编辑 `experiments/spt/unimod1k_improved.yaml`，更新这3个路径：
```yaml
MODEL:
  PRETRAINED: '/root/autodl-tmp/STARKS_ep0500.pth.tar'  # 你的STARK-S预训练权重
  LANGUAGE:
    PATH: '/root/autodl-tmp/bert/bert-base-uncased.tar.gz'  # BERT模型
    VOCAB_PATH: '/root/autodl-tmp/bert/bert-base-uncased-vocab.txt'  # BERT词表
```

### 步骤3: 启动训练
```bash
# 基础版（使用改进配置）
python train_improved.py --config unimod1k_improved --save_dir ./checkpoints_improved

# 启用自动评测
python train_improved.py --config unimod1k_improved --save_dir ./checkpoints_improved --auto_eval

# 从checkpoint恢复
python train_improved.py --config unimod1k_improved --resume ./checkpoints_improved/SPT_ep0080.pth.tar
```

---

## 📊 训练监控

### 查看训练日志
```bash
# 实时查看最新日志
tail -f logs/training_unimod1k_improved_*.log

# 查看loss和IoU变化
grep "Loss/total" logs/training_*.log | tail -20
grep "IoU:" logs/training_*.log | tail -20
```

### TensorBoard可视化（如果配置了）
```bash
tensorboard --logdir tensorboard --port 6006
```

### GPU监控
```bash
# 实时监控GPU使用
watch -n 1 nvidia-smi

# 或者用更友好的工具
gpustat -i 1
```

---

## 🧪 测试训练好的模型

### 快速测试单个checkpoint
```bash
# 1. 更新配置中的TEST.EPOCH
# 在 experiments/spt/unimod1k_improved.yaml 中设置:
# TEST:
#   EPOCH: 120  # 你想测试的epoch

# 2. 运行测试
python tracking/test.py \
  --tracker_name spt \
  --tracker_param unimod1k_improved \
  --dataset_name unimod1k \
  --runid 1 \
  --threads 0 \
  --num_gpus 1

# 3. 查看结果
ls lib/test/tracking_results/spt/unimod1k_improved_001/rgbd-unsupervised/
```

### 自动评测多个checkpoint
```bash
# 评测epoch 80的checkpoint
python auto_evaluate.py --checkpoint_epoch 80 --config unimod1k_improved --save_results

# 评测epoch 120的checkpoint
python auto_evaluate.py --checkpoint_epoch 120 --config unimod1k_improved --save_results

# 查看评测历史
cat eval_history.json
```

---

## 📈 预期训练效果

### 训练指标目标
| Epoch | Loss/total | Loss/giou | IoU | 说明 |
|-------|-----------|-----------|-----|------|
| 0-20  | 0.8-1.0   | 0.35-0.45 | 0.65-0.70 | 初期快速下降 |
| 20-80 | 0.4-0.6   | 0.20-0.30 | 0.75-0.80 | 稳定学习 |
| **80** | **LR下降** | **继续下降** | **0.80-0.82** | **第一个里程碑** |
| 80-120 | 0.3-0.4  | 0.15-0.20 | 0.82-0.85 | 精细调整 |
| **120** | **LR下降** | **继续下降** | **0.85+** | **第二个里程碑** |
| 120-240 | 0.25-0.35 | 0.12-0.18 | 0.85-0.88 | 收敛 |

### 对比原版训练
- **原版**: Loss卡在0.3，IoU停在0.80，80轮后不再下降
- **改进版**: Loss持续下降到0.25，IoU达到0.85+，80轮后仍在优化

### 测试效果对比
- **原版**: 框容易"歪"，快速移动时丢失目标
- **改进版**: 框更稳定，抗漂移能力强，长序列跟踪准确

---

## ⚙️ 配置文件对比

### 原版 vs 改进版关键差异

| 参数 | 原版 (unimod1k.yaml) | 改进版 (unimod1k_improved.yaml) | 说明 |
|------|---------------------|--------------------------------|------|
| **TRAIN.LR** | 1e-5 | 2e-5 | 更快收敛 |
| **TRAIN.BACKBONE_MULTIPLIER** | 0.1 | 0.15 | Backbone学得更快 |
| **TRAIN.SCHEDULER.TYPE** | step | Mstep | 多阶段调整 |
| **TRAIN.SCHEDULER.MILESTONES** | - | [80,120,160,200] | 里程碑优化 |
| **TRAIN.GIOU_WEIGHT** | 2.0 | 2.5 | 更重视IoU |
| **TRAIN.L1_WEIGHT** | 5.0 | 4.0 | 平衡权重 |
| **TRAIN.WEIGHT_DECAY** | 1e-4 | 2e-4 | 更强正则化 |
| **DATA.SEARCH.CENTER_JITTER** | 4.5 | 5.5 | 更强增广 |
| **DATA.TEMPLATE.CENTER_JITTER** | 0 | 1.0 | Template也增广 |
| **DATA.TRAIN.LONG_SEQ_RATIO** | - | 0.3 | 30%长序列训练 |
| **DATA.TRAIN.LONG_SEQ_LENGTH** | - | 4 | 4连续帧 |

---

## 🐛 常见问题排查

### Q1: 训练时显示 "No module named 'lib'"
**解决**:
```bash
export PYTHONPATH=$(pwd):$PYTHONPATH
```
或使用一键启动脚本（自动设置）。

### Q2: CUDA Out of Memory
**解决**:
```yaml
# 在 unimod1k_improved.yaml 中降低batch size
TRAIN:
  BATCH_SIZE: 12  # 从16降到12
```

### Q3: 找不到预训练权重
**解决**:
```bash
# 检查文件是否存在
ls /root/autodl-tmp/STARKS_ep0500.pth.tar
ls /root/autodl-tmp/bert/bert-base-uncased.tar.gz

# 更新配置文件中的路径
vim experiments/spt/unimod1k_improved.yaml
```

### Q4: 训练Loss震荡
**解决**:
```yaml
# 降低学习率
TRAIN:
  LR: 0.00001  # 从2e-5降到1e-5

# 或增加梯度裁剪
TRAIN:
  GRAD_CLIP_NORM: 0.2  # 从0.1增加到0.2
```

### Q5: 长序列训练太慢
**解决**:
```yaml
# 降低长序列比例
DATA:
  TRAIN:
    LONG_SEQ_RATIO: 0.2  # 从0.3降到0.2
    LONG_SEQ_LENGTH: 3   # 从4降到3
```

### Q6: 测试时仍然"框很歪"
**检查清单**:
1. 确认使用了修复后的 `lib/test/tracker/spt.py`
2. 确认训练IoU真的达到了0.85+
3. 尝试降低 `TEST.SEARCH_FACTOR` 从5.0到4.5
4. 确认用的是长序列训练的checkpoint

---

## 📂 文件结构

```
UniMod1K/SPT/
├── train_improved.py                # 主训练脚本（新）
├── auto_evaluate.py                 # 自动评测脚本（新）
├── launch_training.sh               # 一键启动脚本（新）
├── QUICK_START.md                   # 本文档（新）
├── IMPROVEMENT_GUIDE.md             # 详细改进指南（新）
│
├── experiments/spt/
│   ├── unimod1k.yaml               # 原版配置
│   └── unimod1k_improved.yaml      # 改进版配置（新）
│
├── lib/train/
│   ├── base_functions_improved.py  # 混合采样数据加载（新）
│   ├── data/
│   │   └── sampler_longseq.py      # 长序列采样器（新）
│   └── actors/
│       └── spt_longseq.py          # 长序列Actor（新）
│
├── lib/test/tracker/
│   └── spt.py                      # 修复后的tracker（已修改）
│
└── lib/test/evaluation/
    └── running.py                  # 修复后的保存逻辑（已修改）
```

---

## 🎯 推荐训练流程

### 新手推荐（保守）
```bash
# 1. 先用原版配置训练40轮，确认环境OK
bash launch_training.sh --config unimod1k --keep-ckpt 3

# 2. 再用改进版配置从头训练
bash launch_training.sh --config unimod1k_improved --auto-eval --keep-ckpt 5
```

### 老手推荐（激进）
```bash
# 直接用改进版配置+自动评测，一步到位
bash launch_training.sh --config unimod1k_improved --auto-eval --keep-ckpt 10
```

### 资源受限（省显存/磁盘）
```bash
# 修改配置: BATCH_SIZE=12, LONG_SEQ_RATIO=0.2
# 然后启动
bash launch_training.sh --config unimod1k_improved --keep-ckpt 3
```

---

## 📞 获取帮助

如果遇到问题，请检查：
1. **训练日志**: `logs/training_*.log`
2. **改进指南**: `IMPROVEMENT_GUIDE.md`（详细技术说明）
3. **配置文件**: `experiments/spt/unimod1k_improved.yaml`（确认路径正确）
4. **环境检查**: `bash launch_training.sh`会自动检查并报告问题

---

## ✅ 成功标志

训练成功的标志：
- ✅ 日志中能看到 `Loss/total` 持续下降
- ✅ `IoU` 在80轮后仍在上升（不停滞）
- ✅ Checkpoint文件正常保存到 `checkpoints_improved/`
- ✅ 测试时框不再"歪"，能跟上快速移动

预计训练时间（单卡4090）：
- 每epoch约10-15分钟（含长序列）
- 总共240epoch ≈ 40-60小时
- 建议在epoch 80/120/160暂停评测，选最优继续

祝训练顺利！🎉

