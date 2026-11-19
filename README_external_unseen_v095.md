# v095 unseen GO：外部基线与旧锚点评估

本包接入现有 `test_manifest.json`，保留独立测试的完整 1,800 蛋白轴，以及 Dark-Weak 的完整 40,384 蛋白轴。候选沿用既定的 265 个新 GO，不根据测试正例选词。这里的数量来自当前实验；程序以实际 manifest 为准，人数不符不会靠截取凑齐。

LATENCE 的训练、模型结构和 `predict_nbs_go_v095.py` 没有修改。新增训练入口仅训练外部模型。补充的 `deepgozero.tgz` 是原版 DeepGOZero 源码，含 59 个本地 JAR；当前主流程改用该原版，先前 deepgo2 适配保留为补充。源码附件没有作者权重或配套训练词表，已有权重评估仍需这些资源。

## 采用的路线

| 方法 | 源码判断 | 本轮交付 |
|---|---|---|
| MZSGO | 重训准备成本较高。缺训练数据、权重、预计算特征；训练入口存在 tokenizer/model 返回值接反、ESM 未定义变量；依赖文件包含作者本机路径 | 使用真实 checkpoint 的完整分数导出器；完整多模态及明确标记的 no-domain 消融；统一双向评估 |
| 原版 DeepGOZero | 原始 `DGELModel` 与 `predict_zero` 已接入；JAR 提供 OWLAPI/JCEL。需显式修复源码的 `nn.BatchNorm` 构造，并正确处理规范化辅助类 | 本地本体规范化、原版 checkpoint 覆盖/分数导出、label-zero-shot 重训、统一双向评估 |
| 先前附件 deepgo2 | 实际是 DeepGO-SE，包含 DeepGOZero-IPR/ESM 变体 | 原适配程序继续保留，运行入口见 `DEEPGO.md`，与原版分开命名 |
| LATENCE 旧锚点 | 可以直接复用 epoch 160 及原图输入，需为完整 Dark-Weak 队列另外打分 | 新增逐正例名次变化表和汇总，见 [ANCHOR_EVAL.md](ANCHOR_EVAL.md) |

DeepGO 重新训练时允许 2025 ontology 公理进入 EL loss，但目标 GO 不进入标签损失。这是 **label-zero-shot**。LATENCE 当前是冻结模型后追加 GO 表示，属于另一种时间扩展设定。MZSGO 作者 checkpoint 的训练范围未核实前，结果只能称为 checkpoint 迁移评估。三者需分别注明训练蛋白、标签、ontology 与特征的可见范围。

更详细的源码核查与方法参数见 [MZSGO](docs/MZSGO.md)、[原版 DeepGOZero](DEEPGOZERO_ORIGINAL.md)、[本体规范化](NORMALIZE_DEEPGOZERO.md)；旧 deepgo2 说明保留在 [DEEPGO.md](DEEPGO.md)。指标口径见 [EXTERNAL_EVALUATION.md](EXTERNAL_EVALUATION.md)。

## 1. 放置文件与设置现有实验路径

将 ZIP 中 `scripts/` 合并到 LATENCE 项目根目录，保留项目原有 `nbs_models/` 等依赖。本包包含原 v095 预测脚本，旧锚点推理不再缺少该文件。两个外部源码解压到各自目录，不需要复制进 LATENCE 的 Python 包。

以下命令在 LATENCE 项目根目录运行。先设已有数据路径；其中外部资源路径必须换成真实文件，不能填一个空目录。

```bash
export LATENCE_DATA=/home/dataset-assist-0/datafile/latence-dataset
export EXT=outputs/latence_nbs/ontology/go_2025_06_01/v095_bp_anchored
export TEST=outputs/latence_nbs_eval/v095_bp_unseen_go_test
export POP=outputs/latence_nbs_eval/v095_bp_unseen_go_population
export IND_IN=outputs/latence_nbs_eval/bp_shared_full_task_top512_v071_v2
export IND_PRED=outputs/latence_nbs_eval/v095_bp_unseen_go_epoch160_pred
export DW_PRED=outputs/latence_nbs_eval/v095_bp_dark_weak_full_epoch160
export EXT_BASE=outputs/latence_nbs_eval/external_unseen_bp
export IND_EXT="$EXT_BASE/ind_test_inputs"
export DW_EXT="$EXT_BASE/dark_weak_inputs"

# 对应原 LATENCE 独立测试输入的 FASTA；若目录中没有此文件，填原独立集 FASTA。
export IND_FASTA="$IND_IN/normalized_sequences.fasta"

# MZSGO 与新补充的原版 DeepGOZero 源码目录。
export MZ_REPO=/absolute/path/to/MZSGO-src
export DGZ_REPO=/absolute/path/to/deepgozero
```

如果之前用 MSA 二进制准备独立测试，`normalized_sequences.fasta` 可能没有生成。此时必须提供原测试蛋白的查询序列，不能拿“只有正例蛋白”的 FASTA 替代。程序按 manifest 的完整蛋白集合核对，缺少序列直接报错。Dark-Weak 的完整 FASTA 已由 population 脚本导出，不需要 MSA。

## 2. 一次准备两种外部模型共用的输入

下面使用冻结历史 2022 表作为 Dark-Weak 的 InterPro 特征来源，独立测试使用已重新解析的 2025 表。特征来源会写入 manifest。**InterPro accession 命中与 InterPro2GO 标签是不同输入；程序只取前者。**

```bash
python scripts/nbs/prepare_external_unseen_inputs.py \
  --test-manifest "$TEST/test_manifest.json" --cohort ind_test_exp \
  --fasta "$IND_FASTA" \
  --interpro-metadata "$LATENCE_DATA/swissprot_2025_03_go20250601_reparsed.pkl" \
  --feature-provenance "Swiss-Prot 2025_03 original InterPro hits; GO labels excluded" \
  --output-dir "$IND_EXT"

python scripts/nbs/prepare_external_unseen_inputs.py \
  --test-manifest "$POP/test_manifest.json" --cohort dark_weak_all_exp \
  --fasta "$POP/dark_weak_population.fasta" \
  --interpro-metadata "$LATENCE_DATA/swissprot_2022_04.pkl" \
  --feature-provenance "Swiss-Prot 2022_04 historical InterPro hits; GO labels excluded" \
  --output-dir "$DW_EXT"
```

默认列名 `proteins`、`sequences`、`interpros` 可通过对应参数调整。按 ID 加标准化序列一致性连接，重复 ID、序列变化、缺少原始序列均报错。不要用 2025 标签回填输入。若确实有缺失的 InterPro 记录，先确认原因；只有明确接受零特征的实验才加 `--allow-missing-interpro-rows`，程序保留蛋白并输出缺行清单。

若仅跑 MZSGO no-domain 或 DeepGO-ESM，可以省略两个 `--interpro-*` 相关参数（即 `--interpro-metadata` 和 `--feature-provenance`），使用新输出目录。程序输出的 `metadata.pkl` 仅含蛋白、序列及可选 IPR，不含 GO 标签。用于 DeepGO 的 ESM 特征由其原生抽取器计算，不能直接拿 MZSGO 的 1,280 维 ESM2-650M 向量代替 2,560 维 ESM2-3B。

## 3. MZSGO：两种完整队列导出

先取得真实 checkpoint 和 domain 文本特征缓存。附件没有这两个文件；文件名来自原配置，不表示它们已经在服务器上存在。安装说明、离线模型参数与显存选项见 [MZSGO 文档](docs/MZSGO.md)。

```bash
MZ_CKPT="$MZ_REPO/ckpt/cafa5/MZSGO/biological_process.pt"
MZ_DOMAIN="$MZ_REPO/data/embeddings_cache/domain/domain_embeddings_qwen_4b.pkl"

for AXIS in ind dark; do
  if [ "$AXIS" = ind ]; then INPUT="$IND_EXT"; else INPUT="$DW_EXT"; fi
  python scripts/nbs/export_mzsgo_unseen_scores.py \
    --repo-dir "$MZ_REPO" --checkpoint "$MZ_CKPT" \
    --input-dir "$INPUT" --go-obo "$LATENCE_DATA/go-2025_06_01.obo" \
    --namespace bp --domain-mode metadata --domain-embeddings "$MZ_DOMAIN" \
    --device cuda --batch-size 16 --go-batch-size 128 \
    --output-dir "$EXT_BASE/mzsgo_${AXIS}_pred" || break
done
```

若暂时无法取得 domain 缓存，可把 `--domain-mode metadata --domain-embeddings ...` 改为 `--domain-mode none`，并将输出改成 `mzsgo_no_domain_${AXIS}_pred`。这会被标记为消融，不能冒充完整 MZSGO。

有准确 checkpoint 训练 GO/蛋白列表时，增加 `--training-go-ids FILE`、`--training-protein-ids FILE`。程序报告与当前候选、蛋白的交集；未提供时标记训练覆盖未知。

## 4. 原版 DeepGOZero：先覆盖审计，再选择旧权重或重训

已有原权重及原配套数据时，先运行 [DEEPGOZERO_ORIGINAL.md](DEEPGOZERO_ORIGINAL.md) 中 `export_deepgozero_original_scores.py --audit-only` 命令。必须用该 checkpoint 原本的监督词表、InterPro 词表和按原顺序保存的 normalized axioms。原训练脚本默认使用 `terms_zero_10.pkl`，公开预测脚本却默认使用 `terms.pkl`；应按实际 checkpoint 配对，不能凭默认名字猜测。

缺失新 GO 不是低分，也不是负例。覆盖为零时，程序仍生成覆盖诊断和零列矩阵，通用 evaluator 返回“无可评分 GO”；它不会输出虚假的零召回率。要为更多新 GO 产生模型分数，需有训练过这些 ontology 类的 checkpoint，或运行下面的 label-zero-shot 重训。

准备旧训练划分的命令如下。`--label-format go-ids` 适用于标签已是 `GO:xxxxxxx` 的列表；若原标签是旧任务词表中的**零起始整数索引列表**，改成 `--label-format indices`。程序不会猜测数字的含义，密集 multi-hot 矩阵不按索引列表处理。验证分区可以用 `--valid-split` 指定实际键名；未指定时，仅在 `valid/val/validation` 中唯一匹配时自动选择。

```bash
DG_PREP="$EXT_BASE/deepgo_train_data"
python scripts/nbs/prepare_deepgo_training_from_unidata.py \
  --metadata "$LATENCE_DATA/unidata_with_exp_train_pseudo.pkl" \
  --protein-table "$LATENCE_DATA/swissprot_2022_04.pkl" \
  --namespace bp --terms-file "$LATENCE_DATA/go_terms_bp.txt" \
  --train-split train --label-field prop_annotations --label-format go-ids \
  --output-dir "$DG_PREP"
```

准备器只取明确指定的 train/valid 分区和标签字段，不自动合并 LATENCE 的其他 weak/pseudo 分区或教师软分数。报告中应列出两边实际训练蛋白和监督来源；不能仅凭使用同一 unidata 文件便宣称监督完全等量。整数索引的含义还依赖原词表顺序，`--terms-file` 必须来自当初编码该字段的词表。

现在可以使用原版附带的 JAR，从冻结的 2025-06-01 OBO 或 OWL 生成规范式。**修正之前的表述：OBO 不等于 `go.norm`，但包含逻辑关系及定义的 OBO 可以交给 OWLAPI/JCEL 规范化。**它与 GO-Plus OWL 的逻辑内容不保证相同。下面默认使用同一份新 OBO；程序检查 NF1–NF4 和候选覆盖。详细依赖、原脚本辅助类编号问题及修正方式见 [规范化说明](NORMALIZE_DEEPGOZERO.md)。

```bash
DGZ_NORM="$EXT_BASE/deepgozero_go20250601_norm"
python scripts/nbs/normalize_deepgozero_ontology.py \
  --repo-dir "$DGZ_REPO" \
  --ontology-file "$LATENCE_DATA/go-2025_06_01.obo" \
  --candidate-go-ids "$EXT/new_go_ids.txt" \
  --output-dir "$DGZ_NORM"
```

原版 MLP 中的 `nn.BatchNorm` 不是有效 PyTorch 类，因此重训必须明确选择修复方式。下面指定 `BatchNorm1d`；已有权重导出按参数和运行统计量判断归一化类型并严格加载。NF4 半径索引的原生与修正版也分开记录，下面保留原生损失。

```bash
DG_NEW_NORM="$DGZ_NORM/go.norm"
DG_TRAIN="$EXT_BASE/deepgozero_original_bp_label_zero"
python scripts/nbs/train_deepgozero_original_baseline.py \
  --repo-dir "$DGZ_REPO" \
  --train-pickle "$DG_PREP/train.pkl" --valid-pickle "$DG_PREP/valid.pkl" \
  --terms-file "$DG_PREP/terms.pkl" --candidate-go-ids "$EXT/new_go_ids.txt" \
  --normal-forms "$DG_NEW_NORM" --exclude-input-dir "$IND_EXT" \
  --feature-kind interpros --mlp-norm batchnorm1d --epochs 256 --batch-size 37 \
  --nf4-radius-policy native --seed 1 --device cuda:0 --output-dir "$DG_TRAIN"

for AXIS in ind dark; do
  if [ "$AXIS" = ind ]; then INPUT="$IND_EXT"; else INPUT="$DW_EXT"; fi
  python scripts/nbs/export_deepgozero_original_scores.py \
    --repo-dir "$DGZ_REPO" --input-dir "$INPUT" \
    --terms-file "$DG_TRAIN/terms.pkl" --normal-forms "$DG_TRAIN/go.norm" \
    --checkpoint "$DG_TRAIN/deepgozero_original.th" \
    --interpros-file "$DG_TRAIN/interpros.pkl" --candidate-policy registry \
    --device cuda:0 --output-dir "$EXT_BASE/deepgozero_original_${AXIS}_pred" || break
done
```

训练器按 ID 和精确序列排除独立测试重叠，并记录排除量；不会自动把 Dark-Weak 历史蛋白全部剔出训练。后者保留 retrospective 角色。真正同源性独立的外部比较还需另做一致的同源聚类划分。

## 5. 统一双向评估，直接对照现有 LATENCE 输出

下面对两种外部方法分别运行。没有完成某方法的分数导出时，只运行已完成的方法；不要创建空预测文件。DeepGO 若只做已有权重评估，改成实际预测目录即可。

```bash
for METHOD in mzsgo deepgozero_original; do
  python scripts/nbs/eval_external_unseen_go_v095.py \
    --test-manifest "$TEST/test_manifest.json" --cohort ind_test_exp \
    --predictions-dir "$EXT_BASE/${METHOD}_ind_pred" \
    --reference-predictions-dir "$IND_PRED" \
    --output-dir "$EXT_BASE/${METHOD}_ind_vs_latence" || break

  python scripts/nbs/eval_external_unseen_go_v095.py \
    --test-manifest "$POP/test_manifest.json" --cohort dark_weak_all_exp \
    --predictions-dir "$EXT_BASE/${METHOD}_dark_pred" \
    --reference-predictions-dir "$DW_PRED" \
    --output-dir "$EXT_BASE/${METHOD}_dark_vs_latence" || break
done
```

两种外部方法相互比较时，将 `--reference-predictions-dir` 指向另一方法导出目录。相同 population manifest、相同完整蛋白轴的其他证据 cohort 可以复用 Dark-Weak 分数，更换 `--cohort` 与输出目录即可，无需再次推理。无正例集合会输出覆盖、完整分母和未定义指标。

| 输出 | 用途 |
|---|---|
| `metrics.json` | 自身支持轴与双方共同 GO 轴的全部指标，及模型 provenance |
| `coverage.tsv` | 每个候选 GO 的支持情况与已观测正例数 |
| `paired_protein_to_GO.tsv` | 每个正例在 GO 候选中的名次比较 |
| `paired_GO_to_protein.tsv` | 每个正例在完整蛋白队列中的名次比较 |
| 各 scope 子目录 | 正例、蛋白、GO 的详细指标表 |

蛋白→GO 报告 P@K、R@K、F1@K、MRR、Fmax、AP；GO→蛋白报告 P@K、R@K、F1@K、AP 和名次。Fmax/AP 的原口径只对有已观测正例的可评蛋白计算，另保留以全部可评蛋白为分母的 precision proxy。它们不能解释为所有未标注配对都是可靠负例的分类性能。共同 GO 轴若缩小，必须同时报告剩余 GO、正例和正例蛋白数，不能把较小候选池的分数直接与全 265 GO 分数比较。

## 6. 旧锚点与验证

完整命令在 [ANCHOR_EVAL.md](ANCHOR_EVAL.md)。新输出 `positive_pairs_comparison.tsv` 会保留当前 9 个 Dark-Weak 正例，显示原锚点和新 GO 推理的完整名次区间与期望 Top-K 变化。已有 1,800 蛋白的锚点矩阵不能直接用于 40,384 蛋白的评估。

本包附带轻量测试，均用合成数据检查接口和数学边界。运行方式如下。

```bash
python -m unittest discover -s tests -p 'test_external_input_preparation.py'
python -m unittest discover -s tests -p 'test_anchor_retrieval.py'
python -m unittest discover -s tests -p 'test_prepare_deepgo_training.py'
python tests/check_external_unseen_eval.py
python tests/test_mzsgo_export.py --repo-dir "$MZ_REPO"
DEEPGOZERO_SOURCE="$DGZ_REPO" python tests/test_original_deepgozero.py
```

已验证完整轴保留、重排对齐、缺列、精确并列、标签隔离、输入哈希、零正例/零支持和旧锚点身份检查。当前开发环境没有 PyTorch、作者权重及大型编码器，真实 GPU 推理、训练时长、数值精度复现未执行；模型加载/forward 测试在服务器具备 PyTorch 后会启用。源码审查和合成测试通过不等于已经跑出了外部模型在本测试集上的结果。
