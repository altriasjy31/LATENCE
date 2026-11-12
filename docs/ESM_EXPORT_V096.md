# v096 的 ESM 分类前表征

这里提取的是附件 `esm_models/model.py` 中 **ESM2ResNet 最终分类 Linear 的输入**：

```
ESM2-3B 第 34/35/36 层的 mean 向量（各 2560 维）
→ 各分支 BN 和 DNN → 拼接 → concat_layer
→ 可选 res → [在此取 h_esm] → output_layer[0]: Linear → GO logits
```

因此，`h_esm` 是训练过 GO 分类任务的 ESM2ResNet 隐表示，不是 ESM Transformer 第 36 层的原始 2560 维向量，也不是 GO 概率。维度来自 checkpoint 的 `layer_list[-1]`，不能仅按默认值猜测。导出器使用实际前向的 pre-hook，不另写近似前向；每个 checkpoint 的第一个小批次执行完整前向，并验证 `Linear(h_esm)` 与 logits 一致。其余批次在进入最终 Linear 前停止，减少 GO logits 的内存和计算。

导出期间执行 `eval()`、冻结权重并关闭梯度，计算用 float32，默认缓存为 float16。使用内存映射按批写出，不把全量特征放入内存。`features.npy`、`protein_ids.npy` 与 `manifest.json` 构成一个完整缓存目录；失败时不发布半成品。

## 已有原始 ESM 缓存时

首先由 `prepare_esm_ids_v096.py`准备 core、weak、ind_test 所需的唯一蛋白质 ID 列表。然后选择**事先固定的 BP checkpoint**，不能根据独立测试性能挑选：

```bash
python scripts/nbs/export_esm_penultimate_v096.py \
  --esm-source-dir /path/to/esm_models \
  --checkpoint /path/to/esm_models/checkpoints/BPO/model.pt \
  --aspect BPO \
  --features-dir /path/to/esm_cache/core \
  --features-dir /path/to/esm_cache/weak \
  --features-dir /path/to/esm_cache/ind_test \
  --ids /path/to/protein_ids.txt \
  --output-dir outputs/latence_esm_v096/bp_penultimate \
  --device cuda:0 --batch-size 512 --dtype float16
```

`--features-dir` 可重复。缺少任一 ID、重复 ID、原始层缺失、维度错误、非有限数值，或不同缓存目录中相同 ID 的内容冲突，都会报错。ID 必须与 LATENCE 使用的 ID 完全一致；不会自动删除 `sp|` 等前后缀或猜测别名。

若要使用多个独立训练的分类器，必须明确 `--ensemble concat`，并按固定顺序列出 `--checkpoint model1.pt model2.pt ...`；也可对 **BPO 子目录**使用 `--model-dir /path/to/BPO --ensemble concat`，按文件名排序。独立模型的隐空间没有坐标对齐依据，因此不平均这些隐藏向量。拼接后维度等于各模型隐藏维度之和，后续投影层适配该维度。

多个集合可以分别导出，只要 `representation_identity` 完全相同；这要求源代码、checkpoint 内容及顺序、层选择和拼接策略一致。训练与推理不能换一套分类器或调换拼接顺序。

## 缺少原始 ESM 缓存时

先使用附件已有 `plm.py`，仅计算尚缺的蛋白质。示例：

```bash
python /path/to/esm_models/plm.py \
  /path/to/required_missing_proteins.fasta outputs/esm3b_extract_v096 \
  --model_name esm2_t36_3B_UR50D \
  --model_path /path/to/esm2_t36_3B_UR50D.pt \
  --cache_dir /path/to/esm_cache/v096_missing \
  --use_gpu --repr_layers 34 35 36 --include mean
```

显式指定 `34 35 36`；原脚本的默认参数含负索引，不能只凭默认参数认定缓存键与分类器输入一致。现有 ESM 缓存的来源、序列版本和截断方法也应与既有预测流程一致。仅有 `esm_models_src.zip` 源码不足以生成实际表示，还需要分类器 checkpoint、原始 ESM 缓存；缓存缺失时还需要 ESM2-3B 权重与对应蛋白质序列。

## 监督来源与结果解释

这是**冻结的、经 GO 监督训练的分类器表示**。模型的训练蛋白质、注释日期、GO 范围与独立测试集是否重叠，需要审计，尤其是 dark/weak 和新 GO 实验。未经审计时，不能把此分支称为严格零样本或严格时间外推。导出器允许 `--provenance-json provenance.json` 保存这些事实，但不会把用户填写的声明自动视为已经验证。

默认安全载入 checkpoint；若自己生成的旧 checkpoint 因 Python/Numpy 对象被 PyTorch `weights_only` 拒绝，可在确认来源后显式添加 `--trust-checkpoint`。源 ESM 缓存本身为旧版 pickled dictionary 格式，只应读取自己生成或信任的缓存。

验证命令：

```bash
ESM_SOURCE_DIR=/path/to/esm_models \
  python -m unittest discover -s tests -p 'test_v096_esm_export.py' -v
```
