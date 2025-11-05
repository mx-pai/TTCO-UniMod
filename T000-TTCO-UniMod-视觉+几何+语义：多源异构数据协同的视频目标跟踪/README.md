# TTCO-UniMod 提交包说明

> 打包提交时请将此根目录重命名为实际的“参赛团队编号－代码模型－赛题名称”，例如 `T001-TTCO-UniMod-视觉+几何+语义：多源异构数据协同的视频目标跟踪`。

本目录（以下简称 `<SUBMISSION_ROOT>`）整合 TTCO-UniMod 方案的训练代码、测试脚本、模型权重放置位置与评测结果目录，对应“全球校园人工智能算法精英大赛·算法挑战赛 视觉+几何+语义：多源异构数据协同的视频目标跟踪”赛题。

## 1. 目录结构

```
<SUBMISSION_ROOT>/
├─ README.md                     # 当前说明文档
├─ code/                         # 完整训练与测试源代码
│   └─ SPT/                      # SPT 多模态跟踪器工程根目录
├─ models/                       # 训练生成的 checkpoint 与预训练模型
├─ results/                      # 测试输出、评测指标与可视化
└─ docs/                         # 额外说明、实验记录或报告
```

- 需要提交的模型文件放入 `models/`，并同步更新代码中的路径指向。
- 所有评测日志、指标表、可视化等放入 `results/`，保持结构清晰可读。

## 2. 环境依赖与安装

建议使用 Conda 创建独立环境，确保依赖一致。

```bash
cd <SUBMISSION_ROOT>/code/SPT
conda env create -f environment.yml
conda activate spt
export PYTHONPATH=$(pwd):$PYTHONPATH
```

> `environment.yml` 固定了 `pytorch==1.7.0`、`torchvision==0.8.1`、`cudatoolkit==10.2` 以及 `onnxruntime-gpu==1.6.0`、`jpeg4py` 等依赖。若 `jpeg4py` 安装失败，请先执行 `sudo apt-get install libturbojpeg`。

## 3. 数据集与路径配置

1. **数据准备**：将 UniMod1K 训练/验证集解压至自定义目录（例 `/data/UniMod1K/TrainSet`）。测试集保持官方要求的 `list.txt` 与编号子目录结构。
2. **训练路径设置**：修改 `code/SPT/lib/train/admin/local.py`：
   ```python
   self.workspace_dir = '<SUBMISSION_ROOT>/results/experiments'
   self.pretrained_models = '<SUBMISSION_ROOT>/models/pretrained'
   self.unimod1k_dir = '/data/UniMod1K/TrainSet'    # RGBD+文本训练集
   self.unimod1k_dir_nlp = '/data/UniMod1K/TrainSet'
   ```
3. **实验配置**：根据需求编辑 `code/SPT/experiments/spt/unimod1k*.yaml`：
   ```yaml
   MODEL:
     PRETRAINED: '<SUBMISSION_ROOT>/models/pretrained/STARKS_ep0500.pth.tar'
     LANGUAGE:
       PATH: '<SUBMISSION_ROOT>/models/pretrained/bert-base-uncased.tar.gz'
       VOCAB_PATH: '<SUBMISSION_ROOT>/models/pretrained/bert-base-uncased-vocab.txt'
   PATHS:
     DATA_ROOT: '/data/UniMod1K/TrainSet'
     NLP_ROOT:  '/data/UniMod1K/TrainSet'
     OUTPUT_DIR: '<SUBMISSION_ROOT>/results/experiments'
   ```
4. **测试路径**：在 `code/SPT/lib/test/evaluation/local.py` 中同步设置
   ```python
   settings.unimod1k_path = '<path to test set>'
   settings.results_path = '<SUBMISSION_ROOT>/results/evaluations/tracking_results'
   settings.network_path = '<SUBMISSION_ROOT>/models/checkpoints'  # 指向最终 checkpoint
   ```

## 4. 训练流程

```bash
cd <SUBMISSION_ROOT>/code/SPT

# 基线训练
python lib/train/run_training.py \
  --config unimod1k \
  --run_name ttco_baseline_$(date +%m%d_%H%M)

# 增强版训练（长序列采样、自动评测等）
python train_improved.py \
  --config unimod1k_improved \
  --run_name ttco_improved_$(date +%m%d_%H%M) \
  --auto_eval --eval_epochs 10
```

训练输出（checkpoints、日志、TensorBoard 等）默认写入 `<SUBMISSION_ROOT>/results/experiments/<config>/<run_name>/`。训练完成后请选择关键 checkpoint 复制/移动到 `<SUBMISSION_ROOT>/models/checkpoints/` 中，便于打包。

## 5. 测试与结果整理

1. 在 `tracking/parameters/spt/unimod1k.yaml` 中确认 `TEST.EPOCH` 与 `lang_threshold` 等参数与当前权重匹配。
2. 执行评测：
   ```bash
   cd <SUBMISSION_ROOT>/code/SPT
   python tracking/test.py \
     --tracker_name spt \
     --tracker_param unimod1k \
     --dataset_name unimod1k \
     --runid 1 \
     --threads 0 \
     --num_gpus 1
   ```
3. 评测输出默认写入 `settings.results_path/spt/<tracker_param>_<runid>/rgbd-unsupervised/`。请将关键指标（如 `performance.json`）、log 文件及可视化移动/汇总到 `<SUBMISSION_ROOT>/results/` 中，并补充说明文件便于评审查阅。

## 6. 日志与清理

- 使用 `tail -f <log_file>` 或 `tensorboard --logdir <tensorboard_dir>` 监控训练。
- `python auto_clean.py --root <SUBMISSION_ROOT>/results/experiments --keep 3` 可保留最近 N 次运行，释放磁盘空间。

## 7. 提交前检查清单

- [ ] 根目录已按“参赛团队编号－代码模型－赛题名称”命名（示例：`T001-TTCO-UniMod-视觉+几何+语义：多源异构数据协同的视频目标跟踪`）。
- [ ] `models/` 内包含最终提交所需的模型权重、预训练文件及必要说明。
- [ ] `results/` 内整理了最新评测指标、日志、可视化，并包含 README 描述。
- [ ] README、QUICK_START 等文档已根据实际路径与环境更新。
- [ ] 环境依赖与运行命令验证通过，具备完整复现信息。
