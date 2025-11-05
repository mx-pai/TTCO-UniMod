# SPT Tracking on UniMod1K

<p align="center">
  <img width="75%" src="./spt_vdl_framework.jpg" alt="SPT framework"/>
</p>

本目录包含 UniMod1K 论文中的多模态追踪器 SPT 的官方实现，以及我们当前比赛使用的改进版训练脚本。

> **Source**: SPT 模块源自原始 UniMod1K 发布仓库（示例参考 [UniMod1K 官方仓库](https://github.com/xuefeng-zhu5/UniMod1K)），本版本在其基础上进行了定制化修改。

---

## 1. 环境与依赖

```bash
cd <PROJECT_ROOT>/code/SPT
conda env create -f environment.yml
conda activate spt
export PYTHONPATH=$(pwd):$PYTHONPATH
```

> 文中出现的 `<PROJECT_ROOT>` 代表当前项目根目录，请替换为实际路径。

> `jpeg4py` 需要系统安装 `libturbojpeg`。Ubuntu 上执行 `sudo apt-get install libturbojpeg`。

---

## 2. 路径配置与数据准备

### 2.1 训练阶段配置

在 `lib/train/admin/local.py` 中填写训练所需路径（可按需修改）：

```python
class EnvironmentSettings(EnvSettings):
    def __init__(self):
        super().__init__()
        self.workspace_dir = '<WORKSPACE_DIR>'          # 训练输出根目录
        self.tensorboard_dir = f"{self.workspace_dir}/tensorboard"
        self.pretrained_models = '<PROJECT_ROOT>/models/pretrained'        # 存放 STARK/BERT 权重
        self.unimod1k_dir = '/data/UniMod1K/TrainSet'
        self.unimod1k_dir_nlp = '/data/UniMod1K/TrainSet'
```

同时在 `experiments/spt/unimod1k.yaml`（或 `unimod1k_improved.yaml`）中指定模型与数据位置：

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

`OUTPUT_DIR` 会自动生成 `/<config>/<run_name>/` 子目录，保存 checkpoints、日志、tensorboard、metadata、配置快照等。

若需更强的数据增广，可在 `TRAIN.AUG` 中分别开启 `COLOR_JITTER`、`GAUSSIAN_BLUR` 与 `RANDOM_ERASE` 并调整参数。

### 2.2 测试阶段配置

在 `lib/test/evaluation/local.py` 中填写测试路径（默认示例如下）：

```python
def local_env_settings():
    settings = EnvSettings()
    settings.prj_dir = '<PROJECT_ROOT>/code/SPT'
    settings.save_dir = '<RESULTS_OUTPUT_DIR>'
    settings.unimod1k_path = '/data/UniMod1K/TestSet'
    settings.results_path = '<RESULTS_OUTPUT_DIR>/tracking_results'
    settings.network_path = '<PROJECT_ROOT>/models/checkpoints'
    return settings
```

> `'<WORKSPACE_DIR>'`、`'<RESULTS_OUTPUT_DIR>'` 为用户自定义的训练与测试输出目录（例如 `/data/experiments/ttco` 和 `/data/experiments/ttco/eval`），请提前创建并保证可写。

测试数据目录结构需包含 `list.txt` 以及按序号命名的帧：

```
/data/UniMod1K/TestSet/
├── list.txt                 # 例如列出 001, 002, …
└── 001/
    ├── color/00000001.jpg …
    ├── depth/00000001.png …
    ├── groundtruth.txt      # 第一行 [x, y, w, h]
    └── nlp.txt
```

---

## 3. 启动训练

### 标准训练
```bash
python3 lib/train/run_training.py \
  --config unimod1k \
  --run_name baseline_$(date +%m%d_%H%M)
```

### 改进版训练（长序列采样等增强）
```bash
python3 train_improved.py \
  --config unimod1k_improved \
  --run_name improved_$(date +%m%d_%H%M)
```

可选参数：
- `--run_name`：自定义实验名称；默认使用时间戳。
- `--output_root`：覆盖 YAML 中的 `PATHS.OUTPUT_DIR`。
- `--auto_eval`/`--eval_epochs`（仅 `train_improved.py`）：训练中按指定 epoch 自动调用评测脚本。

运行日志保存在 `runs/<config>/<run_name>/logs/`。目录下还会生成 `metadata/`（包含配置快照与 git 信息）。

---

## 4. 监控与评测

```bash
# 日志
tail -f <WORKSPACE_DIR>/<config>/<run_name>/logs/*.log

# tensorboard（如配置）
tensorboard --logdir <WORKSPACE_DIR>/<config>/<run_name>/tensorboard --port 6006

# GPU 监控
watch -n 1 nvidia-smi
```

评测模型：
1. 在配置文件里设置 `TEST.EPOCH` 为要评测的 checkpoint 编号。
2. 准备测试参数：在 `tracking/parameters/spt/unimod1k.yaml` 中指定要加载的 epoch（旧模型默认 240），必要时将 `lang_threshold` 设为 0.0 避免语义门控直接清零预测。

   ```yaml
   TEST:
     EPOCH: 240
   lang_threshold: 0.0
   ```

3. 运行：
   ```bash
   python3 tracking/test.py \
     --tracker_name spt \
     --tracker_param unimod1k \
     --dataset_name unimod1k \
     --runid 1 \
     --threads 0 \
     --num_gpus 1
   ```
4. 结果会自动写入 `settings.results_path` 下的子目录（详见 `lib/test/evaluation/local.py`），无需手动移动文件。

> 旧版本 checkpoint 不包含 `lang_gate` 参数，加载时会提示 Missing keys…，属正常现象；语言门控会使用当前代码的默认初始化。若要充分利用语言约束，可重新训练模型。

---

## 5. 清理旧实验

使用 `auto_clean.py` 释放空间，例如仅保留每个配置最新 3 个 run：

```bash
python3 auto_clean.py \
  --root <WORKSPACE_DIR> \
  --keep 3 \
  --force
```

支持 `--config unimod1k_improved` 指定配置、`--quiet` 安静输出等。

---

## 6. 更多说明

- 详细的分步操作请参见 [`QUICK_START.md`](./QUICK_START.md)。
- 若需要调整数据读取或训练策略，可参考源码中 `lib/train/base_functions.py`、`lib/train/data/sampler_longseq.py`、`lib/train/actors/spt.py`。

---

## 致谢

本项目基于 [Unimod1K](https://github.com/xuefeng-zhu5/UniMod1K) 实现，感谢原作者开源贡献。
