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

打包前请确保：

- `evaluations/` 下包含用于提交的最终追踪结果文件（含官方要求的命名格式）。
- 提供至少一份指标文件（如 `performance.json` 或人工汇总表）。
- 必要时在本 README 中补充额外说明（测试集版本、评测参数等）。
