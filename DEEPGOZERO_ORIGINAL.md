# 原版 DeepGOZero：训练与固定测试轴评估

本轮新增入口针对补充的 `deepgozero.tgz`，使用其中 `deepgozero.py::DGELModel`。上一轮的 `export_deepgo_unseen_scores.py` / `train_deepgo_unseen_baseline.py` 仍对应 deepgo2/DeepGO-SE 变体，保留为独立实验，不能交叉加载两种模型权重。

本轮原版入口如下。

- `export_deepgozero_original_scores.py`：已有原版 checkpoint 的候选覆盖审计与完整分数导出。
- `train_deepgozero_original_baseline.py`：原版 DGELModel 的受控 label-zero-shot 训练，保留项目旧训练/验证划分，新 GO 只进入 ontology loss。
- 输入仍由 `prepare_external_unseen_inputs.py` 生成；输出仍交给 `eval_external_unseen_go_v095.py`，支持独立集与完整 Dark-Weak。

## 源码差异与必要修正

原版默认 hidden/embed 维数为 1024，使用 InterPro multi-hot 和 `predict_zero()`。其 InterPro 分支本来就有正确的 `= 1`，上一轮 deepgo2 附件的遗漏不适用于这份源码。原版 `predict_zero` 的计算是 `sigmoid(MLP(x) · (GO + hasFunction) + |radius|)`，没有把 `go_norm` 再应用于推理类向量。导出器保持该读出，不做阈值截断、祖先传播或分数舍入。

但上传源码仍有实际运行问题，需要明确处理。

1. `MLPBlock` 写成 `nn.BatchNorm(out_features)`，PyTorch 没有这个类。适配器只在内存中的模型定义里替换该构造器，不修改源目录。**训练必须显式选择** `--mlp-norm batchnorm1d` 或 `--mlp-norm layernorm`。示例选择 `batchnorm1d`，这是明确的兼容修正，不声称推知作者真实意图。旧 checkpoint 导出默认 `auto`，根据 MLP running-mean buffers 判断 BatchNorm1d/LayerNorm，再 strict 加载；选择会写入 provenance。
2. 原版 NF4 半径也用 relation 索引读取 class radius。默认 `--nf4-radius-policy native` 保留该行为；`class-radius` 是另行标记的修正版，结果需分开。
3. `deepgozero_zero.py` 导入缺失的 `deepgoel`，并把 train/valid/test 拼接后评估。新入口按项目 manifest 评估，不能把原脚本的 all-protein 结果直接作为独立集结果。
4. 原训练默认文件是 `terms_zero_10.pkl` / `deepgozero_zero_10.th`，预测默认却是 `terms.pkl` / `deepgozero.th`。本入口要求显式传入配套文件，不能只凭目录名猜是哪组权重。

模型定义与 normal-form 解析器通过 AST 从附件提取，避免启动原 CLI 的无关 DGL/import 依赖。这里改变的是运行适配与评估口径；真实作者数据、权重、原环境仍是严格复现论文数值的前提。

## 已有 checkpoint 先审计

以下变量使用主 README 已准备的 `$IND_EXT`、`$DW_EXT`、`$TEST`、`$POP`、`$IND_PRED`、`$DW_PRED`。`DGZ_DATA` 必须是该权重训练时的词表与规范化公理目录。新生成的 2025 `go.norm` **不能拿来重新解释旧 checkpoint 的类行号**。

```bash
DGZ_REPO=/absolute/path/to/deepgozero
DGZ_DATA=/absolute/path/to/original-deepgozero-data
DGZ_CKPT="$DGZ_DATA/bp/deepgozero.th"

python scripts/nbs/export_deepgozero_original_scores.py \
  --repo-dir "$DGZ_REPO" --input-dir "$IND_EXT" \
  --terms-file "$DGZ_DATA/bp/terms.pkl" --normal-forms "$DGZ_DATA/go.norm" \
  --audit-only --output-dir "$EXT_BASE/deepgozero_original_registry_audit"
```

查看 `go_coverage.tsv`。监督类默认不作为 zero-shot 候选；不在原 registry 中的类没有有效模型行。零覆盖会生成 `(N, 0)` 矩阵和明确诊断，统一评估器给出未定义指标，不将其伪装成零召回率。

原 `deepgozero_predict.py` 还筛选 `definitions_go.txt` 中的类，并按 namespace 过滤。这里默认 `--candidate-policy registry`，在已经由项目 manifest 限定 namespace 的 265 个候选中，对所有有训练表示的 zero-class 打分；这是为了固定候选池作比较的明确适配。若要复核原脚本的逻辑定义筛选，添加：

```bash
--candidate-policy definitions --definitions-file "$DGZ_DATA/definitions_go.txt"
```

该文件是 ontology 的逻辑定义集合，并非 OBO 中的自然语言 `def:` 描述。两种策略的覆盖可能不同，需分别报告；不根据测试正例选择策略。

## 已有权重的完整分数导出

```bash
for AXIS in ind dark; do
  if [ "$AXIS" = ind ]; then INPUT="$IND_EXT"; else INPUT="$DW_EXT"; fi
  python scripts/nbs/export_deepgozero_original_scores.py \
    --repo-dir "$DGZ_REPO" --input-dir "$INPUT" \
    --terms-file "$DGZ_DATA/bp/terms.pkl" --normal-forms "$DGZ_DATA/go.norm" \
    --interpros-file "$DGZ_DATA/bp/interpros.pkl" --checkpoint "$DGZ_CKPT" \
    --mlp-norm auto --candidate-policy registry --device cuda:0 --batch-size 64 \
    --output-dir "$EXT_BASE/deepgozero_original_${AXIS}_pred" || break
done
```

全部蛋白都保留，包括没有词表内 IPR 的蛋白。这些蛋白可能产生大并列块，统一 evaluator 按随机并列顺序下的期望命中计算，不用行号破坏并列。完整 Dark-Weak 和独立集使用相同 API。

## 用项目旧划分重新训练

同上一轮，先使用 `prepare_deepgo_training_from_unidata.py` 生成 `$DG_PREP/train.pkl`、`valid.pkl` 和 `terms.pkl`。只取指定旧 train/valid 标签，原始蛋白表仅提供序列与 IPR；无需重建测试集。

新的 `go.norm` 由本轮 normalizer 程序生成，查看其 normalization manifest 确认 GO 版本、输入格式、候选覆盖及丢弃公理。JAR/Normalizer 补全了该准备环节，但不会自动为历史 checkpoint 加入新 GO 模型参数。用新 `go.norm` 训练后，应永久保留该文件字节和行序。

本轮检查发现原 Groovy normalizer 的额外转换问题，新的 wrapper 使用明确记录的 corrected 路径。训练器若发现旁边有 `normalization_manifest.json`，只接受 completed/corrected/normalized 且 `go.norm` 哈希相符的结果，并复制该证明到训练目录；`native` 诊断输出不能用于新训练。旧外部 normal forms 没有 sidecar 时仅标记为来源未验证，不据此宣称与新 corrected 路径等价。

```bash
DGZ_NORM=/absolute/path/to/go20250601_normalized/go.norm
DGZ_TRAIN="$EXT_BASE/deepgozero_original_bp_label_zero"

python scripts/nbs/train_deepgozero_original_baseline.py \
  --repo-dir "$DGZ_REPO" \
  --train-pickle "$DG_PREP/train.pkl" --valid-pickle "$DG_PREP/valid.pkl" \
  --terms-file "$DG_PREP/terms.pkl" --candidate-go-ids "$EXT/new_go_ids.txt" \
  --normal-forms "$DGZ_NORM" --exclude-input-dir "$IND_EXT" \
  --mlp-norm batchnorm1d --nf4-radius-policy native \
  --audit-only --output-dir "${DGZ_TRAIN}_audit"

python scripts/nbs/train_deepgozero_original_baseline.py \
  --repo-dir "$DGZ_REPO" \
  --train-pickle "$DG_PREP/train.pkl" --valid-pickle "$DG_PREP/valid.pkl" \
  --terms-file "$DG_PREP/terms.pkl" --candidate-go-ids "$EXT/new_go_ids.txt" \
  --normal-forms "$DGZ_NORM" --exclude-input-dir "$IND_EXT" \
  --mlp-norm batchnorm1d --nf4-radius-policy native \
  --embed-dim 1024 --epochs 256 --batch-size 37 --seed 1 --device cuda:0 \
  --output-dir "$DGZ_TRAIN"
```

先看审计中的 `training_ready` 和 `training_blockers`。原 EL loss 无条件计算四种 normal form，且使用类向量 BatchNorm；每种至少两行才进入训练。训练时末尾只有一个蛋白会并入前一批，不丢样本。独立测试按 ID 和精确序列排除；新 GO 不进入监督输出，也不会被当作全零负类。验证选择只用旧标签 BCE。与论文原环境的差异、MLP 修正、NF4 策略和实际数据 hashes 都写入 sidecar。

训练产生 `deepgozero_original.th`、同名 `.json` sidecar、`terms.pkl`、`interpros.pkl`、`go.norm` 和训练历史。然后使用上方导出命令，但把路径改成：

```bash
--terms-file "$DGZ_TRAIN/terms.pkl" \
--normal-forms "$DGZ_TRAIN/go.norm" \
--interpros-file "$DGZ_TRAIN/interpros.pkl" \
--checkpoint "$DGZ_TRAIN/deepgozero_original.th"
```

新 checkpoint 的 sidecar 会验证这几个文件及 MLP normalization。原作者 checkpoint 没有该 sidecar 时，输出明确记录无法从权重自身证明 registry 配套，用户必须提供对应的原始文件。相同矩阵形状无法证明 GO 行号顺序相同。

该重训称为“基于原版 DeepGOZero 的项目 label-zero-shot 对照”，不称为复现原论文的固定拆分和数值。它在训练时看见新 GO 公理，与 LATENCE 冻结模型后扩充新 GO 几何的协议不同，论文需明确区分。

## 两种测试的统一评估

```bash
python scripts/nbs/eval_external_unseen_go_v095.py \
  --test-manifest "$TEST/test_manifest.json" --cohort ind_test_exp \
  --predictions-dir "$EXT_BASE/deepgozero_original_ind_pred" \
  --reference-predictions-dir "$IND_PRED" \
  --output-dir "$EXT_BASE/deepgozero_original_ind_vs_latence"

python scripts/nbs/eval_external_unseen_go_v095.py \
  --test-manifest "$POP/test_manifest.json" --cohort dark_weak_all_exp \
  --predictions-dir "$EXT_BASE/deepgozero_original_dark_pred" \
  --reference-predictions-dir "$DW_PRED" \
  --output-dir "$EXT_BASE/deepgozero_original_dark_vs_latence"
```

共享完整蛋白轴的其他 Dark-Weak 证据视图可更换 cohort 复用同一次导出。两种方向、覆盖率、共同候选比较及并列规则与 LATENCE 的统一评估保持一致。

## 验证范围

```bash
DEEPGOZERO_SOURCE="$DGZ_REPO" python tests/test_original_deepgozero.py
```

无需 PyTorch 的检查覆盖原 registry 顺序、normalization 修正规则、定义筛选、零覆盖完整输出、标签隔离、训练审计和 singleton 合批。PyTorch 可用时自动运行实际原模型的 `forward`/`predict_zero` 与导出一致性测试。当前环境没有 PyTorch 和真实作者权重，真实训练/GPU 推理没有执行，不以这些接口检查代替模型效果复现。
