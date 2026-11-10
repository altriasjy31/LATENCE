# LATENCE Atlas v0.6.2：四组完整指标与分块分类评估

本更新解决 `--classification all` 在 history 组触发 80,000,000 单元限制的问题。该组实际有 6,504 × 21,312 = 138,613,248 个分类单元。v062 默认评估四组全部蛋白、全部原始 GO 分类列，不再以矩阵大小拒绝历史组。它不重新运行模型。

## 安装与运行

这是现有 LATENCE 项目的增量更新，不是独立训练项目。需要现有 dark-weak ranking、native v085 指标和原 Stage1 diagnostics 源码。

在 LATENCE 项目根目录解压后运行：

```bash
python LATENCE_atlas_full_metrics_v0.6.2/install_atlas_v062.py --project-root .

python scripts/dark_weak/eval_atlas_v062.py \
  --stage1-export-dir bp=outputs/dark_weak_eval/stage1_atlas_bp_epoch100_v061 \
  --namespaces bp \
  --train-metadata data/unidata_with_exp_train_pseudo.pkl \
  --train-label-field prop_annotations \
  --classification all \
  --metric-chunk-rows 32 \
  --out outputs/dark_weak_eval/atlas_v062_stage1_bp_all
```

安装器新增 v062 文件；已有 v060/v061 和原始指标源码保持原样。若 v062 文件已存在且内容不同，安装器先备份；内容相同则跳过。必要的旧支持文件仅在缺失时补充。

默认 `--classification all`。`--metric-chunk-rows` 只控制每次读取行数，不改变样本、词表或评估指标。通常不需要调整。`--allow-large-metrics` 和 `--max-metric-cells` 为兼容保留，流式后端无需前者，也不会被后者拦住。

请使用新的输出目录。v061 评分失败前可能已创建 `atlas_v061_stage1_bp_inputs`，同名重跑会触发防覆盖保护。v062 命令采用新的 OUT 与 OUT_inputs，不需要删除旧结果或重导出预测。

如果要直接复用旧版已生成的输入配置，也支持：

```bash
python scripts/dark_weak/eval_atlas_v062.py \
  --prediction-config outputs/dark_weak_eval/atlas_v061_stage1_bp_inputs/prediction_inputs.json \
  --namespaces bp \
  --train-metadata data/unidata_with_exp_train_pseudo.pkl \
  --train-label-field prop_annotations \
  --classification all \
  --out outputs/dark_weak_eval/atlas_v062_stage1_bp_all
```

两种输入方式选一种，不需要手填新的 config。仍支持原 `--complete-manifest`。原生 Stage1 三通道导出会校验 manifest、蛋白/GO 顺序和 SHA256。不会补造缺失预测。

## 全部四组与指标输出

| evidence_layer | 测试集合 | 分类评估 |
|---|---|---|
| history | dark_weak_test_2022_ipr2go_direct | 全部蛋白、全部分类列 |
| future_union | dark_weak_test_direct | 全部蛋白、全部分类列 |
| future_exp | dark_weak_test_exp_direct | 全部蛋白、全部分类列 |
| future_ipr | dark_weak_test_ipr2go_direct | 全部蛋白、全部分类列 |

Stage1 三模型 Backbone、Expert、Modelout，四组均非空时，`classification_metrics.tsv` 应有 12 行。可用 pandas 验收：

```python
from pathlib import Path
import pandas as pd

out = Path('outputs/dark_weak_eval/atlas_v062_stage1_bp_all')
c = pd.read_csv(out / 'classification_metrics.tsv', sep='\t')
print(c.groupby(['namespace', 'evidence_layer']).model.agg(list))
assert set(c.evidence_layer) == {'history', 'future_union', 'future_exp', 'future_ipr'}
assert len(c) == 12  # 本命令仅 BP，且三个 Stage1 通道
```

`classification_metrics.tsv` 保留原完整指标包：

- native protein-centric Fmax、其最佳阈值与 Precision/Recall；0.001 与 0.01 网格对照。
- 精确 micro AP、精确梯形 micro PR-AUC、精确 micro Fmax。
- 原 Stage1 histogram micro Fmax、最佳阈值、Precision/Recall、histogram AP 和精确 AP。
- 原 Stage1 sample Fmax 及对应 Precision/Recall。
- 原 Stage1 official Fmax、最佳阈值、网格 PR-AUC。
- 样本、正标签数量、零分/同分诊断。

这些分类指标仍使用 `prop_annotations` 的祖先闭包与原始分类器全部 GO 列（含别名），保持与旧结果的轴定义一致。默认保留 official 分支；它仍需要项目环境中的 PyTorch，但仅使用 101 个 PR 点的小张量。

`direct_topk_metrics.tsv` 新增四组、全部已有频次分组、每个 K 的直接证据指标：

- `observed_micro_precision`、`observed_micro_recall`、`observed_micro_f1`。
- 蛋白平均 Precision/Recall/F1，以及 mean Precision/Recall 的调和均值。
- 实际 `n_predictions`、命中 `n_hits`、正例 `n_pairs`。
- 有候选/有命中的蛋白数、候选覆盖率与弃权率。

K 默认 10、50、100、512、1024。候选遵循 `score > 0` 且完整并列组的末位排名 ≤ K；排除 GO 根，别名取与原排名一致的均值。因此实际候选可能少于“蛋白数 × K”，Precision 使用真实分母。所有 `observed_*` 和 coverage 字段是 [0,1] 分数；分类 Fmax/AP/AUPR 字段为百分数。

频次掩码在完整词表排名之后应用。各频次分组保留该证据层所有蛋白，避免只保留有该组正例的蛋白而抬高 Precision；macro Recall 仍在有该组正例的蛋白上平均。没有已观测正例的分组 Recall/F1 为 unavailable，不伪造为零。

Precision 表示“已观测 direct 证据支持率”，未获该层证据支持的候选不等于经实验证伪的功能。F1@K 是固定 K 指标，与扫描阈值的闭包 Fmax 分开报告。

原 `atlas_metrics.tsv`、MRR、rank/tie bounds、bootstrap、证据关系与 Stage2 增量表继续输出。未提供 NBS 时 Stage2 增量表保留空表结构。四组存在包含关系，不能视为独立重复。

## 内存与精确性

不构造整个 cohort 的 dense gold/probability 副本，不执行全部 protein×GO 分数的全局 argsort。分类网格逐行累计；精确 AP、PR-AUC 和 exact micro Fmax 使用所有已观测正例的不同分数组，对全部背景分数分块统计严格大于/大于等于的数量。负例独占分数组对梯形 PR-AUC 的影响被保留；未对分数分箱近似精确指标。

算法工作数组随 `chunk_rows × GO数 + 正例数` 增长；操作系统的输入 memmap 页缓存也会占用进程 RSS。真实运行时间取决于数据规模和不同正例分数数量。

流式实现是对原定义的等价实现，manifest 明确记录执行后端、原指标函数的 AST 校验、原源文件哈希和新实现哈希；不声称执行了原 dense 函数。若参考数值函数已变化，会明确报错，避免悄悄换用不一致的指标。

## 验证

在安装后的项目根目录：

```bash
python -m unittest discover -s tests -p 'test_*v062.py' -v
```

覆盖：与原完整指标逐字段对照、同分与零分、阈值边界、empty-gold、精确 AP/AUC 的负例独占分数组、memmap 和重排行、四组全量分类、实际候选分母、频次分组、别名与根、两种输入方式和防覆盖。官方 PyTorch 数值对照在有 PyTorch 的环境运行；缺失时该项明确标记 skipped。

另见 `VALIDATION_V062.json` 中的本次验证记录。这里没有用户的完整预测矩阵，不能代替用户机器上的真实历史集合评分；不会伪造新增的历史 Fmax 结果。
