# LATENCE v095 BP 新 GO 评估

将归档解压到已安装 v095 的 LATENCE 项目根目录。归档包含测试集构建、冻结 checkpoint 推理、评估和对照脚本。预测脚本使用项目中的 v095 模型定义和检查函数，并读取已有的 v095 checkpoint、独立测试特征和本体。

## 构建测试集

```bash
LATENCE_DATA=/home/dataset-assist-0/datafile/latence-dataset
V095_RUN="$LATENCE_DATA/outputs/latence_nbs_experiments/v095/asl_hierarchy_single12800_scratch"
EXT=outputs/latence_nbs/ontology/go_2025_06_01/v095_bp_anchored
IND=outputs/latence_nbs_eval/bp_shared_full_task_top512_v071_v2
TEST=outputs/latence_nbs_eval/v095_bp_unseen_go_test

python scripts/nbs/build_unseen_go_test_v095.py \
  --metadata "$LATENCE_DATA/unidata_with_dark_weak_eval.pkl" \
  --swiss25 "$LATENCE_DATA/swissprot_2025_03.pkl" \
  --new-obo "$LATENCE_DATA/go-2025_06_01.obo" \
  --extension-dir "$EXT" \
  --ind-test-input-dir "$IND" \
  --future-pairs "$LATENCE_DATA/dark_weak_test/tables/future_go_pairs.tsv" \
  --match-index "$LATENCE_DATA/dark_weak_test/tables/cohort_match_index.tsv" \
  --output-dir "$TEST"
```

`IND` 指向原 BP 独立测试集准备好的 v095 输入目录，包含 `ind_test_input_manifest.json`、`protein_ids.txt` 和预测所需数组。脚本读取准备目录的 `normalized_sequences.fasta`（若存在），也可显式指定来自原独立集的 `--ind-test-fasta PATH`。有序列时必须一致。没有序列时，唯一名称或 accession 匹配保留为 `sequence_unverified`。正式的严格序列一致分析可增加 `--require-sequence-match`，并给出可溯源的原独立集 FASTA。输出身份审计表会记录实际可评范围。

`unidata_with_dark_weak_unseen_go.pkl` 在原 unidata 的键上追加 `unseen_go_test`，该字段与 `test_manifest.json` 内容相同。评估程序读取 JSON 清单。预测继续读取原有特征目录。候选轴按扩展清单固定，标签使用 2025 Swiss-Prot 的直接实验注释和 2025-06-01 OBO 的 active canonical ID 或唯一 alt ID。

最后两个 dark-weak 数据参数可一同省略。指定时构建 `dark_weak_future_exp` 补充队列，其蛋白轴来自原严格未来证据表中具有原始直接注释记录的全部匹配蛋白，按历史行编号排列。脚本用原始 2025 pickle 的位置、蛋白 ID 和序列哈希复核这些记录。对该组运行预测时，需用该组输出的蛋白轴单独准备与 v095 兼容的特征。该组历史蛋白可能曾进入弱监督资源，结果用于补充分析。

## 导出 v095 预测并评估

将 `EPOCH` 设为已选定且存在的 `nbs_epochN.pt` 的数字 N。从同一 shell 运行。

```bash
PRED="outputs/latence_nbs_eval/v095_bp_unseen_go_epoch${EPOCH}_pred"
METRICS="outputs/latence_nbs_eval/v095_bp_unseen_go_epoch${EPOCH}_metrics"

python scripts/nbs/predict_nbs_go_v095.py \
  --work-dir "$V095_RUN" --epoch "$EPOCH" --input-dir "$IND" \
  --go-ids-file "$EXT/new_go_ids.txt" \
  --ontology-file "$EXT/ontology_bundle.pt" \
  --output-dir "$PRED"

python scripts/nbs/eval_unseen_go_v095.py \
  --test-manifest "$TEST/test_manifest.json" \
  --predictions-dir "$PRED" --cohort ind_test_exp \
  --output-dir "$METRICS"
```

评估要求预测覆盖所选队列的全部蛋白和扩展的全部 GO，按蛋白 ID 对齐。评估还核对模型配置、ontology bundle、候选行，以及配置了准备目录时的输入清单哈希。它验证目标在 Stage2 任务中没有监督列。

`metrics.json` 提供候选覆盖、并列分数统计、按蛋白和按配对的 Recall@K、Precision@K、MRR、随机排序参考，以及基于已观测正例的 Fmax、AP 和 AUPRC。逐配对表保留最佳与最差名次，便于核查并列。主指标使用有已观测正例的可评蛋白，完整队列和无已观测正例数单独报告。阈值和精确率指标按现有注释计算。未标注关系在生物学上仍然未知。前期盘点的 30 个 BP 直接实验配对来自名称匹配，最终规模以构建清单为准。

## 对照实验

复用前两节的 `TEST`、`EXT`、`V095_RUN` 和 `IND`，以及已经导出的 epoch 160 新 GO 预测目录。确认 `PRED` 指向含 `prediction_manifest.json` 和 `probabilities.f32.npy` 的目录。第二条命令省略 `--ontology-file`，以便在原 v095 本体图上给历史锚点打分。若首次导出显式设置了 `--device` 或 `--batch-size`，在此复用相同设置。

```bash
PRED=outputs/latence_nbs_eval/v095_bp_unseen_go_epoch160_pred
CONTROL=outputs/latence_nbs_eval/v095_bp_unseen_go_epoch160_controls

python scripts/nbs/compare_unseen_go_controls_v095.py prepare-anchors \
  --test-manifest "$TEST/test_manifest.json" \
  --extension-dir "$EXT" \
  --output-go-ids "$CONTROL/old_anchor_go_ids.txt"

python scripts/nbs/predict_nbs_go_v095.py \
  --work-dir "$V095_RUN" --epoch 160 --input-dir "$IND" \
  --go-ids-file "$CONTROL/old_anchor_go_ids.txt" \
  --output-dir "$CONTROL/old_anchor_pred"

python scripts/nbs/compare_unseen_go_controls_v095.py compare \
  --test-manifest "$TEST/test_manifest.json" \
  --extension-dir "$EXT" \
  --predictions-dir "$PRED" \
  --anchor-predictions-dir "$CONTROL/old_anchor_pred" \
  --cohort ind_test_exp --n-permutations 1000 --seed 42 \
  --output-dir "$CONTROL/comparison"
```

旧锚点来自扩展清单中每个新 GO 的 `fit.per_term[*].anchor_rows`。脚本使用原 v095 registry 将这些历史行解析为可查询的 GO ID。旧锚点打分后，对每个蛋白和每个新 GO 取其最近锚点集合的最大分数。当前清单的 265 个新 GO 共用 256 个历史锚点，锚点只需各打分一次。两套预测按蛋白 ID 对齐，共用候选轴、直接实验正例及并列处理。比较前会核对 checkpoint、测试特征、配置和候选行。扩展模型与锚点迁移的差异包含新增 GO 几何和图编码共同产生的影响，也取决于选定的锚点聚合规则。

蛋白行置换每次在完整队列内随机交换整条扩展模型分数向量，固定 GO 列、正例和评分规则。同一列的分数分布、同行 GO 分数间的关系及并列全部保留。程序计算置换均值、2.5% 和 97.5% 分位数，以及单侧经验 p 值。对于 Recall、Precision、Hit rate 和 MRR，统计置换值至少达到观测值的次数，对平均正例名次使用较小值的一侧。分子和分母都加 1。各项 p 值独立报告，未做多重比较校正。置换依赖全队列蛋白行可交换的假设。只运行置换时，可从最后一条命令同时省略 `--extension-dir` 和 `--anchor-predictions-dir`，并使用新的输出目录。

| 输出 | 内容 |
| --- | --- |
| `summary.tsv` | 扩展模型和锚点迁移的指标、差值，以及排名指标的置换统计 |
| `comparison.json` | 完整指标、队列规模、输入哈希、置换参数和成对命中变化 |
| `positive_pairs_comparison.tsv` | 每个已观测正例在两套分数中的名次、并列范围和 top-K 命中权重 |
| `anchor_mapping.tsv` | 新 GO 对应的历史 GO ID 和历史本体行 |
| `permutation_draws.tsv` | 每次置换的排名指标 |

## 全队列 unseen GO

先用现有的 1,800 蛋白预测做 GO → 蛋白检索。每个有已观测正例的新 GO 单独在完整蛋白轴上排序。这个视角与原来的逐蛋白排序 265 个 GO 互补。

```bash
python scripts/nbs/eval_unseen_go_retrieval_v095.py \
  --test-manifest "$TEST/test_manifest.json" --cohort ind_test_exp \
  --predictions-dir "$PRED" --extension-dir "$EXT" \
  --anchor-predictions-dir "$CONTROL/old_anchor_pred" \
  --output-dir "$CONTROL/protein_retrieval"
```

`retrieval.json` 汇总按 GO 宏平均和按正例配对汇总的 Recall@K、观测正例 AP 和名次分位。逐 GO 表覆盖完整候选轴，逐配对表给出正例名次和并列范围。缺少正例的 GO 只报告覆盖。默认排序使用完整蛋白轴，`--protein-pool evaluable` 可将排序池限制为身份核验通过的蛋白。Precision@K 和 AP 都按已观测标签给出代理值。

### 先盘点标签和 InterPro 覆盖

```bash
POP=outputs/latence_nbs_eval/v095_bp_unseen_go_population
python scripts/nbs/extend_unseen_go_population_v095.py \
  --test-manifest "$TEST/test_manifest.json" \
  --new-obo "$LATENCE_DATA/go-2025_06_01.obo" \
  --swiss22-dark "$LATENCE_DATA/dark_weak_2022_04.pkl" \
  --swiss25 "$LATENCE_DATA/swissprot_2025_03.pkl" \
  --match-index "$LATENCE_DATA/dark_weak_test/tables/cohort_match_index.tsv" \
  --interpro2go-2025 "$LATENCE_DATA/interpro2go/2025-06-01/interpro2go" \
  --interpro2go-2022 "$LATENCE_DATA/interpro2go/2022-10-07/interpro2go" \
  --output-dir "$POP"
```

这里的 `--swiss25` 继续用原匹配索引所用的 pickle，按位置和序列校验。全库扫描使用独立的 `--swiss25-full`。若手头只有经过筛选的 pickle，从原 2025_03 DAT 生成全量版本，再在上面的命令中加入 `--swiss25-full "$SWISS25_FULL" --include-swiss25`，并指定一个新输出目录。填写原始归档的实际路径后运行下面的命令。

```bash
DAT25=/path/to/archived/uniprot_sprot_2025_03.dat.gz
SWISS25_FULL="$LATENCE_DATA/swissprot_2025_03_full_go20250601.pkl"
python uni2pandas_go2025.py --swissprot-file "$DAT25" --keep-all \
  --go-file "$LATENCE_DATA/go-2025_06_01.obo" --out-file "$SWISS25_FULL"
```

| 队列 | 正例来源 |
| --- | --- |
| `dark_weak_all_exp` | 完整历史队列中，与 2025 条目严格同序列匹配的原始直接实验注释 |
| `dark_weak_all_ipr25` | 同一批已核验匹配蛋白的 2025 InterPro 命中 × 2025 InterPro2GO |
| `dark_weak_all_union` | 前两层的直接证据并集 |
| `dark_weak_history_ipr22_map25` | 完整历史队列的 2022 InterPro 命中 × 2025 InterPro2GO |
| `dark_weak_history_ipr22_map22` | 完整历史队列的 2022 InterPro 命中 × 2022 InterPro2GO |
| `swissprot_2025_all_exp` / `swissprot_2025_all_ipr25` | 全 Swiss-Prot 2025_03 的直接实验注释或 InterPro 映射 |

先看 `cohort_inventory.tsv` 和 `interpro2go_overlap.tsv`。前者给出每个证据层的队列规模和正例覆盖，后者按映射 IPR 和候选 GO 列出各来源的命中。`inventory.json` 记录文件头日期、全库及匹配来源的覆盖计数。IPR 映射用于计算证据层，独立实验队列用于评估直接实验正例。全库与历史队列可能含 checkpoint 曾见蛋白，报告时注明其回顾性范围。加 `--metadata PATH` 可在输出目录同时写入扩展后的 unidata pickle，输入使用原版 unidata 或与所给基础测试清单一致的已合并 pickle。

### 用同一套预测评估多个证据层

所有 dark-weak 视图共用 `dark_weak_population_protein_ids.txt`。先核对现有 MSA shards 能覆盖该轴，按蛋白 ID 重建有序索引。使用项目已有的 `scripts/dark_weak/prepare_msa_index_view.py` 和 `scripts/nbs/prepare_nbs_ind_test_inputs.py`，并沿用原独立集准备时的冻结 Stage-1 checkpoint、registry 和全词表 selector。用 `dark_weak_population_inputs.pkl` 作为只含蛋白 ID 的 `--metadata-file`，配合 `--msa-index` 读取多序列比对。若用导出的 `dark_weak_population.fasta` 直接传给 `--fasta`，准备器会走单序列路径，应单独记录这项输入设置。

```bash
STAGE1="$LATENCE_DATA/outputs/weak_exp_train_detr/bp_weak_detr_v3_expert_prob_warmstart340_to400/weak_detr_decoder_epoch100.pt"
GRAPH="$LATENCE_DATA/outputs/latence_nbs/bp_weak_detr_v3_expert_prob_warmstart340_to400/epoch100/bp"
DW_IN=outputs/latence_nbs_eval/v095_bp_dark_weak_full_inputs
DW_PRED=outputs/latence_nbs_eval/v095_bp_dark_weak_full_epoch160
MSA_VIEW=/path/to/verified/dark_weak_msa_view/index.pkl
MSA_SOURCE=/path/to/frozen/msa/index.pkl

python scripts/dark_weak/prepare_msa_index_view.py \
  --target-ids "$POP/dark_weak_population_protein_ids.txt" \
  --source-index "$MSA_SOURCE" --out-index "$MSA_VIEW"

python scripts/nbs/prepare_nbs_ind_test_inputs.py \
  --task bp --metadata-file "$POP/dark_weak_population_inputs.pkl" \
  --msa-index "$MSA_VIEW" --stage1-checkpoint "$STAGE1" --data-root "$GRAPH" \
  --weak-graph-manifest "$GRAPH/weak_graph_predictions_full_task_top512/weak_graph_predictions_manifest.json" \
  --candidate-selector-scope full_task --candidate-topk 512 --pp-topk 8 \
  --output-dir "$DW_IN"

python scripts/nbs/predict_nbs_go_v095.py \
  --work-dir "$V095_RUN" --epoch 160 --input-dir "$DW_IN" \
  --go-ids-file "$EXT/new_go_ids.txt" --ontology-file "$EXT/ontology_bundle.pt" \
  --output-dir "$DW_PRED"

python scripts/nbs/eval_unseen_go_retrieval_v095.py \
  --test-manifest "$POP/test_manifest.json" --cohort dark_weak_all_exp \
  --predictions-dir "$DW_PRED" --output-dir "${DW_PRED}_retrieval_exp"
```

已有完整有序 MSA view 时可跳过第一条命令。多个冻结 index 分担覆盖时可重复 `--source-index`。根据 `resolved_config.json` 中的 `full_task.core_neighbors` 调整 `--pp-topk`。可对同一 `DW_PRED` 改用其他 dark-weak 证据层并指定新输出目录，或调用原 `eval_unseen_go_v095.py` 做蛋白 → GO 评估。若需旧锚点 GO → 蛋白对照，用同一 `DW_IN` 按上一节生成原本体预测，再把两个预测目录都交给检索评估器。若盘点时已有准备目录，可用 `--dark-input-dir` 或 `--swiss25-input-dir` 绑定其清单哈希。全 Swiss-Prot 用相同流程替换蛋白轴及输入缓存，先确认 MSA 与磁盘可用。评分矩阵虽只有 265 列，原 BP 输入缓存仍需完整的 backbone 概率轴。
