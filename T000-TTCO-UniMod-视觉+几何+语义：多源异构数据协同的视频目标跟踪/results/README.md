# Results Directory

Collect all evaluation outputs（tracking结果、指标表、可视化、日志）于此目录，推荐结构如下：

```
results/
  ├─ experiments/                     # 训练阶段原始输出（与 PATHS.OUTPUT_DIR 对齐）
  │    └─ unimod1k_improved/
  │         └─ ttco_improved_xx/      # run_name
  │              ├─ checkpoints/
  │              ├─ logs/
  │              └─ tensorboard/
  ├─ evaluations/
  │    └─ spt/
  │         └─ unimod1k_1/            # tracker_param + runid
  │              ├─ rgbd-unsupervised/
  │              │    ├─ 001/track_results.txt
  │              │    └─ ...
  │              ├─ performance.json
  │              └─ summary.log
  └─ reports/
       ├─ metrics.xlsx                # 可选：人工汇总的指标/表格
       └─ figures/                    # 可选：曲线、截图
```
