# 🚀 SPT 训练快速指南

> 当前仓库默认推荐直接使用 `experiments/spt/unimod1k.yaml`。若无需特殊调参，不需要额外配置。

---

## 1️⃣ 准备环境与路径

```bash
cd <PROJECT_ROOT>/code/SPT
export PYTHONPATH=$(pwd):$PYTHONPATH
```

配置路径建议按如下步骤执行，可根据实际部署位置调整：

1. 在 `lib/train/admin/local.py` 中确认训练阶段路径，例如：
   ```python
   self.workspace_dir = '<WORKSPACE_DIR>'
   self.pretrained_models = '<PROJECT_ROOT>/models/pretrained'
   self.unimod1k_dir = '/data/UniMod1K/TrainSet'
   self.unimod1k_dir_nlp = '/data/UniMod1K/TrainSet'
   ```
2. 在 `experiments/spt/unimod1k.yaml`（或 `unimod1k_improved.yaml`）中指定模型与数据位置：
   ```yaml
   MODEL:
     PRETRAINED: '<PROJECT_ROOT>/models/pretrained/STARKS_ep0500.pth.tar'
     LANGUAGE:
       PATH: '<PROJECT_ROOT>/models/pretrained/bert-base-uncased.tar.gz'
       VOCAB_PATH: '<PROJECT_ROOT>/models/pretrained/bert-base-uncased-vocab.txt'

   PATHS:
     DATA_ROOT: '/data/UniMod1K/TrainSet'
     NLP_ROOT:  '/data/UniMod1K/TrainSet'
     OUTPUT_DIR: '<WORKSPACE_DIR>'
   ```
   > 可通过 `TRAIN.AUG` 段开启/调整额外的数据增广（如颜色抖动、模糊、随机擦除等）。
3. 在测试阶段的 `lib/test/evaluation/local.py` 中设置：
   ```python
   settings.unimod1k_path = '/data/UniMod1K/TestSet'
   settings.network_path  = '<PROJECT_ROOT>/models/checkpoints'
   settings.results_path  = '<RESULTS_OUTPUT_DIR>'
   ```
   > `'<WORKSPACE_DIR>'` 与 `'<RESULTS_OUTPUT_DIR>'` 均为自定义输出目录，请提前创建并赋予写权限。

测试数据需包含 `list.txt` 与每个序列的 `color/`, `depth/`, `groundtruth.txt`, `nlp.txt`，文件名需使用 8 位数字。

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
`<WORKSPACE_DIR>/<config>/<run_name>/`  
其中包含 `checkpoints/`, `logs/`, `tensorboard/`, `metadata/` 等子目录，并记录配置快照与 git 信息。

---

## 3️⃣ 监控训练

```bash
# 查看最新日志
tail -f <WORKSPACE_DIR>/<config>/<run_name>/logs/*.log

# 查看 Loss / IoU
grep "Loss/total" <WORKSPACE_DIR>/<config>/<run_name>/logs/*.log | tail
grep "IoU"        <WORKSPACE_DIR>/<config>/<run_name>/logs/*.log | tail

# TensorBoard（如需）
tensorboard --logdir <WORKSPACE_DIR>/<config>/<run_name>/tensorboard --port 6006

# GPU 监控
watch -n 1 nvidia-smi
```

---

## 4️⃣ 评测模型

1. 在 `tracking/parameters/spt/unimod1k.yaml` 中指定要加载的 checkpoint，例如：
   ```yaml
   TEST:
     EPOCH: 240
   lang_threshold: 0.0
   ```
   （旧模型没有语言门控时建议将 `lang_threshold` 设为 0.0，避免输出被过滤。）
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
3. 结果会写入 `settings.results_path` 自动创建的子目录，具体结构可参考 `lib/test/evaluation/local.py`。

---

## 5️⃣ 清理旧实验

使用 `auto_clean.py` 可快速删除旧的 run，避免磁盘占满：

```bash
python3 auto_clean.py \
  --root <WORKSPACE_DIR> \
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
