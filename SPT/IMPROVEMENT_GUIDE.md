# SPT 大刀阔斧改进指南

> **提示**：目前仓库已将“改进版”配置合并回默认的 `experiments/spt/unimod1k.yaml`，`unimod1k_improved.yaml` 仅作为向后兼容的别名（内容与默认配置相同）。下面的记录保留了历史改动思路，若需启用可自行挑选落地。

## 📋 改进清单

### ✅ 优先级1：测试鲁棒性修复（已完成）
**改动文件**:
- `lib/test/tracker/spt.py`: 修复初始化返回值 + 异常框检测
- `lib/test/evaluation/running.py`: 修复保存格式为整数坐标

**改进内容**:
1. **初始化返回正确bbox**: 避免第一行出现 `[1]` 占位符
2. **保存整数坐标**: 将浮点坐标四舍五入为整数
3. **异常框检测**: 如果预测框宽高过小(<5px)或过大(>80%图像)，保留上一帧状态

**测试方法**:
```bash
cd /root/autodl-tmp/UniMod1K/SPT
export PYTHONPATH=$(pwd):$PYTHONPATH
python tracking/test.py --tracker_name spt --tracker_param unimod1k --dataset_name unimod1k --runid 10 --threads 0 --num_gpus 1
```
检查输出文件第一行是否为正确的4个整数，且后续帧无异常大框。

---

### 🚀 优先级2：训练策略改进（新增长序列训练）
**关键文件**:
- `lib/train/data/sampler_longseq.py`: 长序列采样器（3-5连续帧）
- `lib/train/base_functions.py`: 已内置长序列/短序列混合采样
- `experiments/spt/unimod1k_improved.yaml`: 改进版配置文件

**改进内容**:
1. **长序列采样**: 从原来的"随机2帧"改为"1模板+3-5连续搜索帧"
2. **累积损失**: 模拟真实跟踪中的drift，在连续帧上累积loss
3. **混合训练**: 70%短序列（快速收敛） + 30%长序列（抗drift）
4. **增强数据增广**:
   - Template jitter: 0→1.0
   - Search jitter: 4.5→5.5
   - Scale jitter: 0.5→0.6, template 0→0.1
5. **优化学习率策略**:
   - Base LR: 1e-5 → 2e-5
   - Backbone multiplier: 0.1 → 0.15
   - Scheduler: Step → Mstep (里程碑 [80, 120, 160, 200])
   - Weight decay: 1e-4 → 2e-4
6. **调整损失权重**:
   - GIoU: 2.0 → 2.5
   - L1: 5.0 → 4.0

**使用方法**:
1. 将 `unimod1k_improved.yaml` 中的路径更新为你的实际路径
2. 直接运行 `train_improved.py` 或 `lib/train/run_training.py`，长序列混合策略已默认启用

**预期效果**:
- 训练IoU从0.8提升到0.85+
- 测试时drift显著减少（从"框很歪"→"基本跟上"）
- Loss收敛速度略慢（因为任务更难），但最终性能更好

---

### 🎯 优先级3：损失函数优化（待实现）
**计划改进**:
1. **尺度感知损失**: 对小目标增加权重
2. **置信度分支**: 预测框的质量分数，用于过滤低质量预测
3. **Focal Loss**: 聚焦难例，减少简单样本的权重

**预期代码位置**:
- `lib/train/actors/spt.py` 的 `compute_losses` 函数

---

### 📊 优先级4：数据增广升级（部分已实现）
**已实现（在 `unimod1k_improved.yaml`）**:
- 更强的空间扰动（jitter增加）
- Template也加入jitter（原本为0）

**待实现**:
- 颜色扰动（亮度、对比度、饱和度）
- 遮挡模拟（随机mask部分区域）
- 运动模糊（模拟快速移动）
- Cutout/Mixup

**预期代码位置**:
- `lib/train/data/processing.py` 的 `STARKProcessing` 类

---

### 🔬 优先级5：模型结构改进（待研究）
**计划方向**:
1. **注意力可视化**: 分析哪些token对预测贡献最大
2. **多尺度融合**: 使用 FPN 融合多层backbone特征
3. **轻量化backbone**: 替换ResNet-50为 EfficientNet 或 MobileNetV3
4. **动态模板更新**: 每N帧更新一次模板特征（加权平均）

**风险**:
- 改动较大，需要重新训练+调参
- 可能引入新bug

---

## 📈 改进效果预估

| 改进项 | 难度 | 预期提升 | 副作用 |
|--------|------|----------|--------|
| 测试鲁棒性修复 | 低 | +++（从"完全错"→"基本对"） | 无 |
| 长序列训练 | 中 | ++++（抗drift能力显著增强） | 训练速度慢20-30% |
| 损失函数优化 | 中 | ++（小目标、难例性能提升） | 需要调参 |
| 数据增广升级 | 低 | ++（泛化能力提升） | 可能增加训练时间 |
| 模型结构改进 | 高 | +++（理论上限提升） | 需要大量实验 |

---

## 🛠️ 快速开始

### 步骤1: 测试修复后的基础版本
```bash
cd /root/autodl-tmp/UniMod1K/SPT
export PYTHONPATH=$(pwd):$PYTHONPATH

# 使用修复后的tracker测试（确保TEST.EPOCH=78）
python tracking/test.py --tracker_name spt --tracker_param unimod1k --dataset_name unimod1k --runid 11 --threads 0 --num_gpus 1

# 检查结果
head /root/autodl-tmp/UniMod1K/SPT/lib/test/tracking_results/spt/unimod1k_011/rgbd-unsupervised/Adapter/adapter1/adapter1_001.txt
```

### 步骤2: 使用改进配置重新训练（可选）
```bash
# 1. 更新 unimod1k_improved.yaml 中的路径
# 2. 备份原配置
cp experiments/spt/unimod1k.yaml experiments/spt/unimod1k_backup.yaml

# 3. 使用改进配置
cp experiments/spt/unimod1k_improved.yaml experiments/spt/unimod1k.yaml

# 4. 开始训练
python lib/train/run_training.py --script spt --config unimod1k --save_dir ./
```

### 步骤3: 监控训练指标
关注这些指标的变化：
- **Loss/total**: 应该在80轮后继续下降（原本停滞）
- **IoU**: 目标从0.8提升到0.85+
- **Loss/giou vs Loss/l1**: 比例应该平衡

---

## 🐛 常见问题

### Q1: 训练时OOM
**解决**: 降低 `BATCH_SIZE` 从16→12，或禁用长序列训练（`LONG_SEQ_RATIO: 0`）

### Q2: 训练速度太慢
**解决**: 长序列训练会慢20-30%，可以先用短序列训练100轮，再切换到长序列fine-tune

### Q3: Loss震荡
**解决**: 降低LR到1e-5，或增加 `GRAD_CLIP_NORM` 到0.2

### Q4: 测试时仍然"框很歪"
**解决**:
1. 检查是否使用了修复后的 `spt.py`
2. 尝试降低 `TEST.SEARCH_FACTOR` 从5.0→4.5（缩小搜索区域，减少漂移）
3. 检查训练时的IoU是否真的到了0.85+

---

## 📝 待办事项

- [ ] 集成长序列采样器到主训练流程（需修改 `base_functions.py`）
- [ ] 实现置信度分支（需修改模型head）
- [ ] 添加颜色增广（修改 `processing.py`）
- [ ] 实现动态模板更新（修改 `spt.py`）
- [ ] 训练一个完整的改进版模型并评测

---

## 📧 改进反馈

如果你测试后有任何问题或建议，请记录：
1. 使用的配置文件（原版/改进版）
2. 训练到多少epoch
3. 最终的Loss/IoU值
4. 测试集上的AUC/OP50/OP75指标
5. 是否出现"框很歪"或其他异常

这将帮助进一步优化改进方案。
