# Model Weights

Store training checkpoints, pretrained backbones, and language-model assets required to reproduce TTCO-UniMod. Recommended layout：

```
models/
  ├─ checkpoints/                     # 训练得到的最终/中间权重
  │    ├─ ttco_unimod_ep0240.pth.tar
  │    └─ best_auc.pth.tar
  ├─ pretrained/                      # 预训练依赖
  │    ├─ STARKS_ep0500.pth.tar       # 空间骨干
  │    ├─ bert-base-uncased.tar.gz    # 语言编码器
  │    └─ bert-base-uncased-vocab.txt # 与语言模型配套的词表
  └─ README.md                        # 文件说明（本文件，可根据实际情况补充）
```

在运行训练/测试前，请同步更新下列路径以指向本目录中的实际文件：

- `code/SPT/lib/train/admin/local.py` → `self.pretrained_models`、`self.workspace_dir`
- `code/SPT/experiments/spt/*.yaml` → `MODEL.PRETRAINED`、`MODEL.LANGUAGE.PATH`、`MODEL.LANGUAGE.VOCAB_PATH`
- `code/SPT/lib/test/evaluation/local.py` → `settings.network_path`
