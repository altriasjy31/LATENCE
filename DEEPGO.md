# 附件 deepgo2 的实现核查与运行

补充原版 `deepgozero.tgz` 后，主实验入口已改为 [DEEPGOZERO_ORIGINAL.md](DEEPGOZERO_ORIGINAL.md)。本页继续记录之前的 deepgo2 / DeepGO-SE 变体，不能将两组 checkpoint 与源码混用。

附件实际上是 **DeepGO-SE / deepgo2** 项目。它提供 `DeepGOModel`，`train.py -m deepgozero` 使用 InterPro，`-m deepgozero_esm` 使用 ESM2-3B。本文及输出将这一区分保留为“deepgo2 仓库中的 DeepGOZero 变体”，不将它称作原 2022 DeepGOZero 仓库的严格复现。

## 核查结果

- 附件没有模型权重、`terms.pkl`、`interpros.pkl`、原始训练集或 `go.norm`。重新训练 IPR 单模型的计算负担低于先运行 ESM2-3B 再训练六模型 ensemble，但仍需要数据准备和 GPU；源码不能证明具体训练时长。
- 自带 `predict.py` 默认运行 ESM2-3B 六模型集成，只打分 `terms.pkl` 中的监督类，并将 ≥0.1 的分数保留三位小数。这样导出的结果不适合当前完整排名实验。
- `load_normal_forms()` 按规范化公理**首次出现顺序**给 `zero_classes` 分配 embedding 行号。因此评估旧权重必须提供训练时**原文件、原顺序**的 `terms.pkl` 和 `go.norm` / `go-plus.norm`。形状相同不保证行号含义相同，不能只换成 2025 OBO/规范化文件。
- 已有 checkpoint 的零样本支持范围是它已经训练过的 `zero_classes`。真正不在该 registry 的新 GO 没有可用行，不能随机补 embedding 然后宣称零样本性能。导出器会保存缺列原因；零覆盖时仍给出 `(N, 0)` 文件和 `no_supported_go` 状态。
- 附件 `deepgo/data.py:get_data()` 的 InterPro 分支遗漏 `= 1`，原样运行会产生全零输入。新导出器及训练程序显式构造 multi-hot，并报告没有词表内 IPR 特征的蛋白，保留这些蛋白以同分规则评估。曾用这份缺陷代码训练的旧 IPR 权重并不会被修复输入自动修好，应重新训练。
- 附件 `deepgo/base.py:nf4_loss()` 使用 relation 列给类半径索引。训练程序默认 `--nf4-radius-policy native` 保留附件行为，并记录该事实；`--nf4-radius-policy class-radius` 是显式修正版，仅改 NF4 的 C 类半径索引。这两者应作为不同训练配置，不混称为相同实现。已有权重评估不涉及训练损失，不改变其 readout。

## 准备好的两种输入均可直接接入

主 README 先用 `prepare_external_unseen_inputs.py` 生成 `$IND_EXT` 和 `$DW_EXT`。每个目录含 `external_input_manifest.json`、完整 `protein_ids.txt`、`go_ids.txt`、`sequences.fasta`、`metadata.pkl`。IPR 模型需要在准备时接入可靠的、序列核对过的 InterPro 命中。**InterPro 命中是模型特征，InterPro2GO 映射是评估证据，不能把后者作为模型输入。**

导出器核查输入文件哈希，保留所有队列蛋白，包括零覆盖 IPR 的蛋白。不会读取标签列。使用新的输出目录；已有目录会报错以避免旧结果与新结果混合。

## 先审计已有权重的 GO 支持范围

下面的 `DG_DATA` 必须是对应权重的原训练数据资源，而不是项目自己的新 GO 目录。

```bash
DG_REPO=/path/to/deepgo2
DG_DATA=/path/to/deepgo2-original-data

python scripts/nbs/export_deepgo_unseen_scores.py \
  --repo-dir "$DG_REPO" --input-dir "$IND_EXT" \
  --terms-file "$DG_DATA/bp/terms.pkl" \
  --normal-forms "$DG_DATA/go.norm" \
  --audit-only --output-dir outputs/external_unseen/deepgo_ind_registry_audit
```

查看 `go_coverage.tsv`。`zero_class` 可以做 label-zero-shot 打分；`supervised_class` 默认排除，因为该类曾出现在外部模型的监督输出轴；`absent_from_checkpoint_registry` 不能用该权重打分。`--allow-supervised-targets` 只用于有明确标记的非零样本诊断。

## 已有 IPR checkpoint 的完整分数导出

```bash
python scripts/nbs/export_deepgo_unseen_scores.py \
  --repo-dir "$DG_REPO" --input-dir "$IND_EXT" \
  --terms-file "$DG_DATA/bp/terms.pkl" \
  --normal-forms "$DG_DATA/go.norm" \
  --checkpoint "$DG_DATA/bp/deepgozero.th" \
  --feature-kind interpros --interpros-file "$DG_DATA/bp/interpros.pkl" \
  --device cuda:0 --output-dir outputs/external_unseen/deepgo_ind_pred

python scripts/nbs/export_deepgo_unseen_scores.py \
  --repo-dir "$DG_REPO" --input-dir "$DW_EXT" \
  --terms-file "$DG_DATA/bp/terms.pkl" \
  --normal-forms "$DG_DATA/go.norm" \
  --checkpoint "$DG_DATA/bp/deepgozero.th" \
  --feature-kind interpros --interpros-file "$DG_DATA/bp/interpros.pkl" \
  --device cuda:0 --output-dir outputs/external_unseen/deepgo_dark_pred
```

若权重实际由 `deepgozero_plus` 训练，必须改用原 `go-plus.norm` 与对应 checkpoint。不能用这组 IPR 参数加载 ESM 模型。

ESM 模型将 `--feature-kind` 改成 `esm2`、去掉 `--interpros-file`，改用对应 checkpoint。没有 `esm2` 特征列或显式 `--esm-cache` 时，直接调用附件原 ESM2-3B 抽取器，从 prepared FASTA 按 layer 36 均值生成 2560 维向量，保留其前 1022 aa 截断规则。模型资源须预先下载或网络可达。`--checkpoint` 可以重复指定以平均多个模型。不要直接平均不同训练 registry 的权重，原数据轴必须一致。外部 `--esm-cache` 虽核对蛋白 ID 并记录文件哈希，但没有独立序列证明；需用户确认来自这份原 FASTA，优先直接抽取。

分数来自原 `DeepGOModel.forward()`，将 `all_gos` 换成 registry 中明确的候选行；计算原生 sigmoid 点积、hasFunction 关系和类半径。没有阈值、三位小数舍入或祖先传播。导出 `external_prediction_manifest.json` 后使用主 README 中统一 evaluator 比较两个方向，缺列会独立统计。

## 需要所有 265 个候选时重新训练

提供 `train_deepgo_unseen_baseline.py`。它基于附件实际模型构建**label-zero-shot 对照**，旧标签训练、新 GO 只进入 EL ontology loss。这是从头训练的外部模型，并非“冻结历史 checkpoint 后扩充新类”的同一协议。LATENCE 的训练和权重不受影响。

1. 使用旧训练/验证划分、2022 冻结的监督标签及 InterPro 特征。主包的 `prepare_deepgo_training_from_unidata.py` 可准备 `train.pkl`、`valid.pkl`、`terms.pkl`。不能把全量 Swiss-Prot 随意作为训练集，不能用独立测试标签训练。
2. 需要 2025-06-01 对应的规范化公理。**OBO 文件不等于规范式，但原版补充 JAR 的 OWLAPI 支持解析 OBO。**可以从冻结的 OBO 或 OWL 做正式规范化；不能仅提取普通 is_a 边后声称得到完整逻辑定义。本次在原 Normalizer 的辅助类编号处理中发现问题，建议使用 [修正规范化入口](NORMALIZE_DEEPGOZERO.md)。以下保留旧仓库的原生命令供核对，不应在未检查规范化结果时直接训练。

```bash
groovy "$DG_REPO/groovy/Normalizer.groovy" \
  -i /path/to/go-2025-06-01.owl -o /path/to/go-2025-06-01.norm
```

使用 GO-Plus 变体时还要使用匹配版本 GO-Plus OWL 和 `makeEl.groovy` 过滤步骤，并单独标记变体。不要让未绑定版本的 OWL imports 在运行时更新。

3. 先运行无需 PyTorch 的审计，再在另一个新目录训练。

```bash
DG_PREP=outputs/external_unseen/deepgo_train_data
DG_NEW_NORM=/path/to/go-2025-06-01.norm
DG_TRAIN=outputs/external_unseen/deepgo_bp_label_zero

python scripts/nbs/train_deepgo_unseen_baseline.py \
  --repo-dir "$DG_REPO" \
  --train-pickle "$DG_PREP/train.pkl" --valid-pickle "$DG_PREP/valid.pkl" \
  --terms-file "$DG_PREP/terms.pkl" --candidate-go-ids "$EXT/new_go_ids.txt" \
  --normal-forms "$DG_NEW_NORM" --exclude-input-dir "$IND_EXT" \
  --audit-only --output-dir "${DG_TRAIN}_audit"

python scripts/nbs/train_deepgo_unseen_baseline.py \
  --repo-dir "$DG_REPO" \
  --train-pickle "$DG_PREP/train.pkl" --valid-pickle "$DG_PREP/valid.pkl" \
  --terms-file "$DG_PREP/terms.pkl" --candidate-go-ids "$EXT/new_go_ids.txt" \
  --normal-forms "$DG_NEW_NORM" --exclude-input-dir "$IND_EXT" \
  --feature-kind interpros --epochs 128 --batch-size 37 \
  --nf4-radius-policy native --seed 1 --device cuda:0 \
  --output-dir "$DG_TRAIN"
```

程序明确拒绝将任一新 GO 放进监督 terms，包括“将它列为全零负类”这种做法。训练和验证的 annotation/label 列里的目标 GO 被移除并统计，验证选择只使用旧任务 BCE。所选标签列必须是标准 `GO:0000000` 形式的列表；索引、证据编码字符串及缺失浮点值会报错。训练或验证整体没有旧任务正例也会报错，单个蛋白的空标签列表允许保留。只从训练集建立 InterPro 词表，采用原 Adam 学习率与 scheduler 默认值。`--exclude-input-dir "$IND_EXT"` 根据 ID 和标准化后精确序列排除独立测试重叠；不自动排除 dark-weak 中历史训练蛋白，这部分作为 retrospective 评估报告。与训练集高度同源但非完全相同的序列尚未自动排除，因此不能宣称完成了 30% identity 的独立划分。

候选 GO 在新公理中缺失默认报错。`--allow-partial-go` 只允许明确的部分覆盖实验，仍输出覆盖清单。保存 `deepgozero.th`、`deepgozero.th.json`、`terms.pkl`、`interpros.pkl`、`go.norm`、训练审计和历史。训练文件当前要求 DataFrame；多层 unidata 字典先经过准备程序处理，不能直接传给训练器。

4. 将上方导出命令中的 `--terms-file`、`--normal-forms`、`--checkpoint`、`--interpros-file` 分别替换为 `$DG_TRAIN/terms.pkl`、`$DG_TRAIN/go.norm`、`$DG_TRAIN/deepgozero.th`、`$DG_TRAIN/interpros.pkl`。对 `$IND_EXT` 和 `$DW_EXT` 都运行，然后用统一 evaluator。新训练权重的 sidecar 会强制验证 ontology、terms、InterPro 词表哈希。

## 依赖和验证范围

IPR 导出/训练需要 Python、NumPy、pandas 和适配服务器 CUDA 的 PyTorch。导出器从附件读取原 `BaseDeepGOModel`、`DeepGOModel` 和 `load_normal_forms` 定义，跳过不相关的 DGL 导入，无需为了这个非图模型安装 DGL。ESM 推理额外需要 `fair-esm` 和其模型资源。

本次环境缺少 PyTorch、fair-esm 和真实 checkpoint，已验证 registry 原顺序、IPR 输入修复、零覆盖导出、标签移除、精确序列排除和训练审计。真实训练与 GPU 推理须在服务器执行；附带的原生 forward 一致性及合成 checkpoint 端到端导出测试在有 PyTorch 的环境自动启用。

```bash
DEEPGO_SOURCE="$DG_REPO" python tests/test_deepgo_adapter.py
```
