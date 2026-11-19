# 旧锚点迁移的逐对评估

本次增强 `scripts/nbs/eval_unseen_go_retrieval_v095.py`，沿用已有命令和 v095 预测文件格式。只有增加两项 `--extension-dir`、`--anchor-predictions-dir` 时才进行锚点比较；原来单独评估新 GO 的命令继续可用。

这里的锚点来自拟合新 GO 时已经确定的最近历史祖先。每个新 GO 使用这些历史 GO 分数的最大值作为迁移分数。程序不重拟合 box、不训练模型、不根据测试正例挑选锚点。

## Dark-Weak 全队列

从 LATENCE 项目根目录运行。以下路径采用当前实验命名；若实际目录不同，只替换目录变量。`DW_IN` 必须就是生成 `DW_PRED` 时使用的输入目录。

```bash
set -euo pipefail
LATENCE_DATA=/home/dataset-assist-0/datafile/latence-dataset
V095_RUN="$LATENCE_DATA/outputs/latence_nbs_experiments/v095/asl_hierarchy_single12800_scratch"
EXT=outputs/latence_nbs/ontology/go_2025_06_01/v095_bp_anchored
POP=outputs/latence_nbs_eval/v095_bp_unseen_go_population
DW_IN=outputs/latence_nbs_eval/v095_bp_dark_weak_full_inputs
DW_PRED=outputs/latence_nbs_eval/v095_bp_dark_weak_full_epoch160
DW_ANCHOR=outputs/latence_nbs_eval/v095_bp_dark_weak_full_epoch160_anchor
DW_COMPARE=outputs/latence_nbs_eval/v095_bp_dark_weak_full_epoch160_anchor_retrieval_exp
ANCHOR_GO="$POP/anchor_control/old_anchor_go_ids.txt"

# 只导出 extension 中原有的锚点计划，不使用测试正例选锚点。
python scripts/nbs/compare_unseen_go_controls_v095.py prepare-anchors \
  --test-manifest "$POP/test_manifest.json" \
  --extension-dir "$EXT" \
  --output-go-ids "$ANCHOR_GO"

# 使用原历史 ontology。这里不要添加 --ontology-file。
python scripts/nbs/predict_nbs_go_v095.py \
  --work-dir "$V095_RUN" --epoch 160 --input-dir "$DW_IN" \
  --go-ids-file "$ANCHOR_GO" \
  --output-dir "$DW_ANCHOR"

# 新 GO 预测沿用已有 DW_PRED；默认保留完整蛋白队列。
python scripts/nbs/eval_unseen_go_retrieval_v095.py \
  --test-manifest "$POP/test_manifest.json" --cohort dark_weak_all_exp \
  --predictions-dir "$DW_PRED" \
  --extension-dir "$EXT" --anchor-predictions-dir "$DW_ANCHOR" \
  --protein-pool full --top-k 10 50 100 500 1000 \
  --output-dir "$DW_COMPARE"
```

如果已生成完全相同 checkpoint、输入和锚点轴的 `DW_ANCHOR`，直接运行最后一步。已有 1,800 蛋白的锚点预测不能替代 40,384 蛋白的锚点预测。

`prepare-anchors` 使用 extension 绑定的原 registry。若原文件移动过，可加 `--full-ontology-manifest`、`--task-registry`、`--alignment-manifest` 指向迁移后的同一文件；内容 SHA256 必须一致。

## 独立测试集

使用独立测试对应的原特征输入与原预测目录，步骤完全相同。与 Dark-Weak 的预测及输出目录分开。

```bash
TEST=outputs/latence_nbs_eval/v095_bp_unseen_go_test
IND_IN=outputs/latence_nbs_eval/bp_shared_full_task_top512_v071_v2
IND_PRED=outputs/latence_nbs_eval/v095_bp_unseen_go_epoch160_pred
IND_ANCHOR=outputs/latence_nbs_eval/v095_bp_unseen_go_epoch160_anchor
IND_COMPARE=outputs/latence_nbs_eval/v095_bp_unseen_go_epoch160_anchor_retrieval_exp
IND_ANCHOR_GO="$TEST/anchor_control/old_anchor_go_ids.txt"

python scripts/nbs/compare_unseen_go_controls_v095.py prepare-anchors \
  --test-manifest "$TEST/test_manifest.json" --extension-dir "$EXT" \
  --output-go-ids "$IND_ANCHOR_GO"

python scripts/nbs/predict_nbs_go_v095.py \
  --work-dir "$V095_RUN" --epoch 160 --input-dir "$IND_IN" \
  --go-ids-file "$IND_ANCHOR_GO" --output-dir "$IND_ANCHOR"

python scripts/nbs/eval_unseen_go_retrieval_v095.py \
  --test-manifest "$TEST/test_manifest.json" --cohort ind_test_exp \
  --predictions-dir "$IND_PRED" \
  --extension-dir "$EXT" --anchor-predictions-dir "$IND_ANCHOR" \
  --protein-pool full --top-k 10 50 100 500 1000 \
  --output-dir "$IND_COMPARE"
```

如果上次 1,800 蛋白锚点导出的目录另有名称，将 `IND_ANCHOR` 改成那个目录并跳过重复预测。所有评估输出目录须尚不存在，避免覆盖已完成实验。

## 输出与解释

| 文件/字段 | 含义 |
|---|---|
| `positive_pairs_comparison.tsv` | 逐个正例对连接两个模型，保留全部已观测正例；若 manifest 仍含 9 个 Dark-Weak EXP 正例，表中应恰好 9 行数据 |
| `expanded_rank_min/max`、`anchor_rank_min/max` | 在完全相同蛋白队列中的最好/最坏并列名次 |
| `expanded_midrank`、`anchor_midrank` | 并列块内随机排序时的期望名次 |
| `midrank_improvement_over_anchor` | anchor midrank − expanded midrank，正值表示新 GO 推理改善 |
| `*_fraction_ranked_ahead` | (midrank − 1)/(蛋白数 − 1)，越低越好；单蛋白队列设为 0 |
| `top@K_credit_improvement_over_anchor` | 新 GO − 锚点的期望命中信用，正值表示改善；并列时可能为小数 |
| `anchor_ontology_rows`、`anchor_go_ids` | 当前正例 GO 实际使用的历史锚点 |
| `retrieval.json` 中的 `paired_positive_rank_summary` | 名次改善/持平/下降的对数、平均与中位名次变化，以及各 K 的信用增加/减少和净变化 |
| `expanded_per_go.tsv`、`anchor_transfer_per_go.tsv` | 全部候选 GO 的覆盖与逐 GO 汇总；无已观测正例的 GO 保留但指标未定义 |

例子：新 GO 的 midrank 是 900，锚点是 1,500，改善值为 +600。若两者在 Top-1,000 边界都有大并列块，比较的是进入前 1,000 的概率，不用行号打破并列。

程序逐项检查 checkpoint hash、epoch、step、配置 hash、输入 manifest hash、预测实现标识、历史 ontology 身份与所需历史行。蛋白可以采用不同排列顺序，但集合必须等于完整测试队列，缺行或多行直接报错。矩阵 SHA256、GO 轴和有限概率值也会验证。

这些统计回答新 GO 几何与图推理是否超过旧锚点已有的蛋白排序信号。9 个正例只支持逐对描述和探索性总结，不能据此声称普遍优越。未观测注释并非确定负例，因此 AP/precision 仍标为 observed proxy。

原来的蛋白 → GO 比较继续使用 `compare_unseen_go_controls_v095.py compare`，其 `positive_pairs_comparison.tsv` 的名次在候选 GO 轴上；本程序同名文件的名次在完整蛋白轴上。两个输出目录必须分开。

## 本地验证

```bash
python -m unittest discover -s tests -p 'test_anchor_retrieval.py'
```

合成测试覆盖乱序轴对齐、并列名次与命中信用、全队列/可评估子队列、无正例、重复或缺失正例、缺失蛋白、checkpoint 不一致、错误 ontology，以及被修改的概率文件。测试不代替服务器真实权重推理。
