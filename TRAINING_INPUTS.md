# 将已有 unidata 训练划分接入外部 DeepGO 基线

`scripts/nbs/prepare_deepgo_training_from_unidata.py` 沿用 unidata 的原训练、验证蛋白轴与所选标签字段，只从 2022 Swiss-Prot 表补充序列与 InterPro 命中。不读取 Swiss-Prot 表中的 GO 注释，不建立新划分，不使用测试标签。

```bash
DG_DATA=outputs/latence_nbs_eval/deepgo_old_training_inputs
python scripts/nbs/prepare_deepgo_training_from_unidata.py \
  --metadata "$LATENCE_DATA/unidata_with_exp_train_pseudo.pkl" \
  --protein-table "$LATENCE_DATA/swissprot_2022_04.pkl" \
  --namespace bp --terms-file "$LATENCE_DATA/go_terms_bp.txt" \
  --label-field prop_annotations --label-format go-ids \
  --train-split train --output-dir "$DG_DATA"
```

`--label-format` 必须明确填写。GO 字符串列表使用 `go-ids`；若原字段是整数索引列表，使用 `indices`，程序按给定旧 `go_terms_bp.txt` 的准确行顺序映射。不会根据数值猜测含义。若原字段是二维 0/1 dense/sparse 矩阵，需要同时指定 `--label-format indices --label-representation multihot`，其列数必须等于旧词表长度。

元数据结构应为 `metadata[train][bp 或 biological_process]` 及对应验证 block，block 含 `proteins` 和指定标签字段。验证划分默认在顶层 `valid`、`val`、`validation` 中唯一识别；如果存在多个或命名不同，添加 `--valid-split 实际键名`。不自动拿 `test` 或 `ind_test` 作为验证集。

输出 `train.pkl`、`valid.pkl` 和 `terms.pkl` 可直接传入本包的外部训练器：

```bash
python scripts/nbs/train_deepgo_unseen_baseline.py \
  --repo-dir "$DEEPGO_REPO" \
  --train-pickle "$DG_DATA/train.pkl" --valid-pickle "$DG_DATA/valid.pkl" \
  --terms-file "$DG_DATA/terms.pkl" --candidate-go-ids "$EXT/new_go_ids.txt" \
  --normal-forms "$GO2025_NORMAL_FORMS" \
  --exclude-input-dir "$IND_EXT" \
  --audit-only --output-dir outputs/latence_nbs_eval/deepgo_train_audit
```

审计输出与实际训练输出使用不同新目录。通过后去掉 `--audit-only`，按 `DEEPGO.md` 中的训练参数运行。`--exclude-input-dir "$IND_EXT"` 在外部训练阶段按 ID 和完全相同序列剔除独立测试蛋白，并记录剔除统计；准备步骤自身保持原划分。

准备后的标签统一写到 `prop_annotations`，即使 `--label-field` 选择了其他原字段；外部训练器可保留其默认值。旧词表外的 GO 标签会计数后排除，所有蛋白仍保留，包括没有剩余正例的蛋白。新的 GO 候选能否与旧词表完全分离，由后续训练器再次检查。

`prepare_training_manifest.json` 记录输入/输出 hash、原划分名称、排除的词表外配对数量和每个 split 的标签数。如果 block 自带序列，必须与旧蛋白表完全一致（规范化后）；若不带序列，记录为未经过 metadata 序列交叉验证。蛋白缺失、重复 ID、索引越界、格式歧义、train/valid 的 ID 或完全相同序列重叠均报错，不隐式删行或重新切分。

```bash
python -m unittest discover -s tests -p 'test_prepare_deepgo_training.py'
```
