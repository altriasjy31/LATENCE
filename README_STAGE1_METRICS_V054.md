# LATENCE comprehensive frozen-score metrics v0.5.4

本更新统一汇报 Dark–Weak 与 ind_test 的多套指标，完整保留 Stage-1 诊断指标，并加入原始 Stage-1 helper 的兼容计算。已有 Dark–Weak 概率可直接使用，无须重新推理。

## 指标及其用途

| 指标组 | 主要输出字段 | 定义与用途 |
|---|---|---|
| Stage-1 ind_test 诊断指标 | `stage1_hist__fmax_micro`, `threshold_micro`, `precision_micro`, `recall_micro`, `auprc_micro_hist`（均有 `stage1_hist__` 前缀） | 直接执行原诊断源码函数。默认 0.01 步长，float32 概率分箱；旧 `auprc_micro_hist` 是近似 AP |
| Stage-1 额外诊断 | `stage1_hist__fmax_sample`, `stage1_hist__auprc_micro_exact` | 原函数的 sample-Fmax，以及可用时 sklearn 精确 micro-AP。sample-Fmax 与 v085 的空标签行处理不同 |
| Stage-1 原始 helper | `stage1_official__fmax`, `stage1_official__threshold`, `stage1_official__auprc` | 原 `evalperf_torch` 在 CPU float32 上执行，严格 `score > threshold`，固定 101 个阈值；其 AuPRC 是阈值网格 PR 曲线的梯形面积 |
| 可选语义误差 | `stage1_official__smin` | 仅在提供历史训练 IC 向量时执行原 helper 的 Smin；不自动从测试标签估计 IC |
| v085 protein 指标 | `standard_protein_fmax`, `standard_protein_precision_at_fmax`, `standard_protein_recall_at_fmax` | 原 v085 源码，0.001 步长；该组没有真实标签的蛋白从 protein-Fmax 中排除 |
| v085 micro 指标 | `standard_micro_fmax_exact`, `standard_micro_ap`, `standard_micro_pr_auc` | 精确 micro-Fmax、非插值 AP、梯形 PR 面积，等分数整体分组 |
| 全词表排名 | `micro_recall_at_K`, `macro_recall_at_K`, `pairwise_mrr`, `mean_best_rr` | 完整 canonical GO 词表内先排名，再按目标 GO 频次分组；K 默认 10/50/100/512/1024 |
| 同分敏感性 | `*_lower`, `*_upper`, `positive_tie_fraction`, `mean_tie_size` | 同分块内最差/最好位置的排名界限。不是置信区间 |
| 分数诊断 | `score_zero_fraction`, `score_positive_zero_fraction`, `score_all_scores_equal` 等 | 识别 Expert 全零/同分、预测饱和与高梯形 PR 面积退化 |

除阈值、Smin、计数及明确标记的比例外，分类性能指标为百分数；排名 Recall、MRR 为 0–1。保留 `Fmax_percent/AuPRC_percent/AP_percent` 作为 v085 的旧别名，不覆盖为 Stage-1 指标。

Stage-1 两条路径的原函数均从现有源码中提取并原样执行，避免导入模型推理依赖；v085 也直接执行原有指标文件。报告记录完整源文件、函数 SHA256、依赖版本和计算设置。

原 helper 优先读取项目内 `helper_functions/helper.py` 或 `helper.py`；否则使用包内已审计的用户原始函数快照。也可用 `--stage1-helper-source PATH` 指定真实源文件。原 helper 在相同 recall 点上使用未指定稳定排序的 Torch argsort，因此 CPU 与历史 GPU/不同版本的阈值网格 AuPRC 不承诺逐位相同；其来源与设备会明确记录。histogram AP 和 v085 精确 AP 不受这条 helper 排序约定影响。

## 分组与比较口径

默认同时生成：

- `overall / all`：完整原始输出列，重新计算全部指标。
- `stage1_raw_classic`：历史训练标签在原始 GO 输出列上的频次与分位数，尽可能严格对接旧 ind_test rare 报告。
- `historical_classic`：先将历史训练别名对应的 GO 按唯一蛋白合并计数，再计算分位点；与 v0.5.3 保持相同分组思路。
- `historical_fixed`：同一 canonical 历史频次的固定区间。

`stage1_raw_classic` 仅提供分类指标；别名列可能跨组，故不将 canonical 排名强行归入原始列分组。其余分组提供 canonical 排名。所有分类指标都保留被所选分组覆盖的原始分类列，不额外平均或合并预测列。

要对接旧 ind_test 的 7,501 个 BP rare 列，应先查看 `stage1_raw_classic` 的列数与 `go_frequency_lookup.tsv`。精确复现还要求历史训练 metadata、标签字段、原始词表、Modelout 分支、筛选设置及概率文件与旧运行一致。旧诊断源码优先尝试 annotations/labels/prop_annotations；当前已确认的训练文件只有 prop_annotations，因此本程序显式默认使用 prop_annotations。若旧运行使用另一字段，应选择同一实际字段后再比较。当前输入要求 LATENCE 的 GO 索引列表格式。

主指标不删除无标签蛋白或零正例 GO 列；原始诊断 rare 路径也如此。零正例组标记 `no_positives`，空 GO 组标记 `empty_group`。整组没有正例时的零分是定义约定，不表示模型被充分评价。

## 安装与测试

在现有 LATENCE 项目根目录执行。依赖已安装的 Dark–Weak 原始评估代码、`native_v085_metrics.py`、原始 Stage-1 diagnostics 源码及 v085 metrics 源码。本包附带未修改的 v0.5.3 输入工具以方便独立叠加安装。

```bash
unzip -n LATENCE_dark_weak_metrics_v0.5.4.zip -d .

python -m unittest discover -s tests -p '*v054.py' -v
```

本地验证：14 项测试，13 项通过，1 项因没有 PyTorch 而跳过。跳过项是原 helper 的逐值对照（含 Smin）；请在现有训练环境运行该测试。其余测试覆盖原诊断函数和独立计数的一致性、全部 v085 输出字段、空标签行、全零概率、空组、无正例、原始列分组、排名同分上下界、ID 重排、输入哈希及输出保护。

## 对现有 Dark–Weak BP 结果运行

```bash
python scripts/dark_weak/eval_stage1_metrics_v054.py \
  --complete-manifest outputs/dark_weak_eval/stage1_full_bp_epoch100/bp_complete_manifest.json \
  --unified-root data/dark_weak_test/unified \
  --train-metadata data/unidata_with_exp_train_pseudo.pkl \
  --train-split train \
  --train-label-field prop_annotations \
  --families stage1_raw_classic historical_classic historical_fixed \
  --ks 10 50 100 512 1024 \
  --bootstrap 500 \
  --out outputs/dark_weak_eval/stage1_bp_comprehensive_v054
```

程序读取 complete manifest 中冻结的模型和概率文件，检验概率哈希及词表顺序，按蛋白 ID 对齐。旧 complete manifest 没有保存 ID 文件哈希，新报告会冻结当前 ID 文件哈希并注明这一追溯限制。输出目录必须是新目录。

默认启用全部指标。原 Torch helper 需要每个模型/分组运行 101 次阈值计算，较大 ind_test 上可能比 histogram 路径慢。在不具备 Torch 的机器上，可加 `--skip-stage1-official`；此时明确不输出 `stage1_official__*`，不能称该次运行覆盖了全部 Stage-1 路径。

如已确认当前概率的确切模型分支，可增加 `--model-branch 'modelout=实际分支名称'`。这只是显式来源标记，不会改变概率，也不自动证明分支身份。

## 用同一入口评价 ind_test

必须使用 ind_test 自己导出的概率及对应 ID 文件；Dark–Weak 的概率文件不能代替。以下环境变量应填写已有导出的真实路径，三个概率矩阵都需要与相应 GO ID 顺序匹配。

```bash
# 先设置以下实际文件路径：
# IND_BACKBONE, IND_EXPERT, IND_MODELOUT
# IND_BACKBONE_IDS, IND_EXPERT_IDS, IND_MODELOUT_IDS
# IND_BACKBONE_GOS, IND_EXPERT_GOS, IND_MODELOUT_GOS

python scripts/dark_weak/eval_stage1_metrics_v054.py \
  --metadata data/unidata_with_exp_train_pseudo.pkl \
  --namespace bp \
  --splits ind_test \
  --unified-root data/dark_weak_test/unified \
  --train-metadata data/unidata_with_exp_train_pseudo.pkl \
  --train-label-field prop_annotations \
  --prediction "backbone=$IND_BACKBONE" \
  --prediction "expert=$IND_EXPERT" \
  --prediction "modelout=$IND_MODELOUT" \
  --prediction-ids "backbone=$IND_BACKBONE_IDS" \
  --prediction-ids "expert=$IND_EXPERT_IDS" \
  --prediction-ids "modelout=$IND_MODELOUT_IDS" \
  --prediction-go-ids "backbone=$IND_BACKBONE_GOS" \
  --prediction-go-ids "expert=$IND_EXPERT_GOS" \
  --prediction-go-ids "modelout=$IND_MODELOUT_GOS" \
  --families stage1_raw_classic historical_classic historical_fixed \
  --out outputs/dark_weak_eval/ind_test_bp_comprehensive_v054
```

`--metric-label-field` 默认 prop_annotations。排名优先使用 annotations，若该 split 只有 prop_annotations，则明确回退并记录。ind_test 上的排名不自动称为“未来直接新增功能恢复”；报告使用中性 `n_rank_positive_*` 字段。

显式入口也可评价冻结的 NBS 概率，条件是同一原始输出词表、正确 ID 顺序及所需蛋白完整覆盖。这不替代独立的 NBS 导出来源或训练数据审计。

## 可选 IC / v094 频次

有严格按原始 GO 列对齐的历史训练 IC 时，增加：

```bash
--ic-vector /实际路径/historical_train_ic.npy \
--ic-provenance '历史训练数据版本、标签字段、IC公式及GO顺序来源'
```

这会新增 `stage1_official__smin`；IC 单位与原始向量一致。程序验证维度/有限性/非负性并保存哈希，无法凭数值判断向量是否真的只使用历史数据，需由使用者确保来源。

要使用 v094 实际 core_counts，添加 `--v094-unified-manifest 实际路径/unified_dataset_manifest.json` 并将 `v094_native_core_fixed` 加入 `--families`。这个家族的别名频次采用原 v0.5.3 的 max 约定，独立记录，不与 historical 频次混同。

## 输出与优先检查

| 输出 | 内容 |
|---|---|
| `comprehensive_metrics.tsv` | 每个 split × family × bin × model 的全部指标、分数诊断、计数和排名汇总 |
| `overall_metrics.tsv` | 完整 GO 输出空间的整体指标 |
| `frequency_groups.tsv` | 原始/合并频次分组的规模和边界 |
| `go_frequency_lookup.tsv` | 每个原始 GO 列的频次与组归属 |
| `positive_global_ranks.tsv` | 正例稳定排名、同分块上下界和各 K 命中 |
| `metric_deltas_vs_reference.tsv` | 相对 Backbone（可用 `--reference` 修改）的同口径性能差；附单位 |
| `paired_rank_bootstrap_deltas.tsv` | 同序列 SHA256 聚类配对 bootstrap 的稳定排名差异区间；没有哈希时回退蛋白 ID |
| `metrics_manifest.json` | 源码/输入哈希、标签字段、频次来源、模型分支标记、指标约定与序列哈希覆盖 |

最先检查 `stage1_raw_classic / rare` 的 `stage1_hist__fmax_micro` 和 `stage1_hist__auprc_micro_hist`，与旧 ind_test 同名数值对接。再看 `standard_protein_fmax` 与 `standard_micro_ap`，以及 canonical 分组内的 Recall@K/MRR。若 `positive_tie_fraction` 高，应同时读排名上下界。

建议回传前四项汇总文件中的 `comprehensive_metrics.tsv`、`overall_metrics.tsv`、`frequency_groups.tsv`，以及 `metrics_manifest.json` 和 `paired_rank_bootstrap_deltas.tsv`。

## 解释边界

统一代码消除的是指标和分组口径上的歧义，不能消除 Dark–Weak 与 ind_test 蛋白/标签分布的差异。Dark–Weak 的已观测未来标签并不完整；Fmax、AP 使用未知位置作为代理负例。分组最优阈值是根据当前评估标签选择的描述性上界，不能直接作为部署阈值。子词表 Fmax 与完整词表 Recall@K 应并列解释。

Stage-1 histogram 的零阈值包含零概率预测，而原 helper/protein-Fmax 的严格大于零规则排除它们。因此全零分数组即使有非零 histogram micro-Fmax，也不表示存在判别能力。全同分时 exact 梯形 PR 面积可接近 50%，应结合 AP、正例占比及同分标记解释。

本包不执行训练、模型推理、概率混合、候选选择或本体传播，不改写原报告；既有训练进程可继续运行。
