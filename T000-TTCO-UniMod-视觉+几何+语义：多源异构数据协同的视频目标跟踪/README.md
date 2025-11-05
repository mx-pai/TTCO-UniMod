# TTCO-UniMod 项目说明

本目录（简称 `<PROJECT_ROOT>`）整合 TTCO-UniMod 方案的训练代码、测试脚本、模型权重与实验结果，用于团队内部协作与复现。

## 1. 目录结构

```
<PROJECT_ROOT>/
├─ README.md                     # 当前说明文档
├─ code/                         # 完整训练与测试源代码（SPT 工程位于 code/SPT）
└─ models/                       # 训练生成的 checkpoint 及预训练模型
```

- 将模型权重、BERT 词表、预训练骨干等放在 `models/`，并在配置文件中同步更新路径。

## 2. 环境依赖与安装

```bash
cd <PROJECT_ROOT>/code/SPT
conda env create -f environment.yml
conda activate spt
export PYTHONPATH=$(pwd):$PYTHONPATH
```

> `environment.yml` 适配 CUDA 10.2 + PyTorch 1.7.0，同时包含 `onnxruntime-gpu==1.6.0`、`jpeg4py` 等依赖。若安装 `jpeg4py` 失败，请先执行 `sudo apt-get install libturbojpeg`。

## 3. 数据集与路径配置

1. **数据准备**：将 UniMod1K 训练/验证集解压至自定义目录（示例 `/data/UniMod1K/TrainSet`），测试集目录需包含 `list.txt` 与编号子目录。
2. **训练路径**：修改 `code/SPT/lib/train/admin/local.py`：
   ```python
   self.workspace_dir = '<PROJECT_ROOT>/results/experiments'
   self.pretrained_models = '<PROJECT_ROOT>/models/pretrained'
   self.unimod1k_dir = '/data/UniMod1K/TrainSet'
   self.unimod1k_dir_nlp = '/data/UniMod1K/TrainSet'
   ```
3. **实验配置**：根据需求编辑 `code/SPT/experiments/spt/unimod1k*.yaml`：
   ```yaml
   MODEL:
     PRETRAINED: '<PROJECT_ROOT>/models/pretrained/STARKS_ep0500.pth.tar'
     LANGUAGE:
       PATH: '<PROJECT_ROOT>/models/pretrained/bert-base-uncased.tar.gz'
       VOCAB_PATH: '<PROJECT_ROOT>/models/pretrained/bert-base-uncased-vocab.txt'
   PATHS:
     DATA_ROOT: '/data/UniMod1K/TrainSet'
     NLP_ROOT:  '/data/UniMod1K/TrainSet'
     OUTPUT_DIR: '<PROJECT_ROOT>/results/experiments'
   ```
4. **测试路径**：在 `code/SPT/lib/test/evaluation/local.py` 中设置：
   ```python
   settings.unimod1k_path = '/data/UniMod1K/TestSet'
   settings.results_path = '<PROJECT_ROOT>/results/evaluations/tracking_results'
   settings.network_path = '<PROJECT_ROOT>/models/checkpoints'
   ```

## 4. 训练流程

```bash
cd <PROJECT_ROOT>/code/SPT

# 基线训练
python lib/train/run_training.py \
  --config unimod1k \
  --run_name ttco_baseline_$(date +%m%d_%H%M)

# 增强版训练（长序列采样等增强）
python train_improved.py \
  --config unimod1k_improved \
  --run_name ttco_improved_$(date +%m%d_%H%M) \
  --auto_eval --eval_epochs 10
```

训练输出（checkpoints、日志、TensorBoard 等）默认写入 `<PROJECT_ROOT>/results/experiments/<config>/<run_name>/`。训练完成后可将核心 checkpoint 拷贝到 `models/checkpoints/`，方便后续复用。

## 5. 测试与结果整理

1. 在 `tracking/parameters/spt/unimod1k.yaml` 中确认 `TEST.EPOCH` 与 `lang_threshold` 设置正确。
2. 运行评测：
   ```bash
   cd <PROJECT_ROOT>/code/SPT
   python tracking/test.py \
     --tracker_name spt \
     --tracker_param unimod1k \
     --dataset_name unimod1k \
     --runid 1 \
     --threads 0 \
     --num_gpus 1
   ```
3. 结果写入 `settings.results_path/spt/<tracker_param>_<runid>/rgbd-unsupervised/`

## 6. 日志监控与清理

- `tail -f <log_file>` 或 `tensorboard --logdir <tensorboard_dir>` 可实时查看训练状态。
- 使用 `python auto_clean.py --root <PROJECT_ROOT>/results/experiments --keep 3` 可定期清理旧实验，仅保留最新若干次运行。

