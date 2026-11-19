# MZSGO 源码审查与固定测试轴评估

本适配器读取附件中的 `model.py::CustomModel`，加载真实作者 checkpoint，输出完整的 sigmoid 分数。它不重新训练 LATENCE，不查询实时 InterPro，不向输入加入 GO 正例，也不按有无已观测标签筛掉蛋白。独立测试和 Dark-Weak 全队列使用相同入口，区别只有 `--input-dir`。

## 复现难度判断

先做 checkpoint 迁移评估。此次源码包未包含 `ckpt/`、训练数据或 `embeddings_cache`，因此单凭该 ZIP 不能完成作者结果的数值复现，也不能在当前环境验证真实权重预测。还存在可直接定位的可运行性问题。

- `main.py` 用 `nlp_tokenizer, nlp_model = load_nlp_model(config)` 接收结果，但 `go_embed.py` 返回顺序是 `model, tokenizer`。
- `esm_embed.py::precompute_esm_embeddings` 引用未定义的 `tokenizer`、`model`、`num_layers`。预计算缓存存在时可能遮蔽该路径。
- `requirements.txt` 包含作者机器上的 `file:///...` 包路径和大量无关环境包，不适合直接在当前项目中整体安装。
- 默认配置使用作者本机 Qwen 路径、CAFA5/Swiss-Prot 数据、IA 和 domain 缓存。重训还需核对训练/验证划分、旧 GO 监督词表、InterPro 版本、特征缓存与论文超参数。
- 原生 `predict.py` 会查询实时 InterPro，且默认按阈值截断输出。这样的 JSON 不足以直接作固定测试轴的排名比较。

本轮给出的程序走模型评估路线。若后续重训，应先冻结与 LATENCE 相同的历史训练蛋白、可见标签和特征时间点，再修复以上入口。用作者 checkpoint 评估应写作“外部 checkpoint 迁移评估”，不能仅因 265 个 GO 相对 LATENCE 为 unseen，就称其相对 MZSGO 也是 unseen。

## 需要准备的资源

1. 解压后的 MZSGO 源码目录。
2. 真实 BP checkpoint，例如作者预期路径 `ckpt/cafa5/MZSGO/biological_process.pt`。源码包中没有该文件，请从作者发布资源取得；程序不会创建随机替代权重。
3. ESM2-650M 权重与 Qwen3-Embedding-4B 权重。可使用本地模型，或让相应框架下载。
4. 若评估完整多模态模型，需要作者格式的 domain embedding 字典 `domain_embeddings_qwen_4b.pkl`，以及与冻结输入序列一致的本地 IPR 命中记录。
5. `prepare_external_unseen_inputs.py` 生成的输入目录，包含固定轴、FASTA、剥离标签后的 metadata 及校验 manifest。

在独立环境安装本评估入口需要的包即可。现有服务器已安装 PyTorch 时，不要为此执行原始环境的整个 `requirements.txt`。

```bash
python -m pip install numpy pandas tqdm fair-esm==2.0.0 transformers==4.57.1 sentencepiece safetensors
```

另需适合服务器 CUDA 的 PyTorch。代码使用 `fair-esm` 的 `esm.pretrained` API；不要把同名的新版 `esm` 包替换成该接口。Qwen 默认使用源码配置所列的模型 commit `5cf2132abc99cad020ac570b19d031efec650f2b`；若换模型，必须提供相应 checkpoint 与相应维数，不能只换为较小的文字编码器来节省显存。

## 完整多模态模型分数导出

下列 `INPUT` 指向已经准备好的独立测试或 Dark-Weak 输入目录，命令完全相同。

```bash
MZSGO=/absolute/path/to/MZSGO-src
MZ_CKPT="$MZSGO/ckpt/cafa5/MZSGO/biological_process.pt"
MZ_DOMAIN="$MZSGO/data/embeddings_cache/domain/domain_embeddings_qwen_4b.pkl"
INPUT=outputs/latence_nbs_eval/external_unseen_bp/ind_test_inputs
MZ_OUT=outputs/latence_nbs_eval/external_unseen_bp/mzsgo_ind

python scripts/nbs/export_mzsgo_unseen_scores.py \
  --repo-dir "$MZSGO" --checkpoint "$MZ_CKPT" \
  --input-dir "$INPUT" --go-obo "$LATENCE_DATA/go-2025_06_01.obo" \
  --namespace bp --domain-mode metadata --domain-embeddings "$MZ_DOMAIN" \
  --output-dir "$MZ_OUT" --device cuda \
  --batch-size 16 --go-batch-size 128
```

将 `INPUT` 改成 Dark-Weak 完整输入目录，并更换 `MZ_OUT`，即可导出 40,384 行的完整矩阵。程序要求 FASTA 与蛋白轴精确匹配，始终保留没有已观测正例的蛋白。首次生成 ESM 特征是主要开销，程序逐行缓存，重跑相同输出目录可恢复已经完成的特征行。完整输出存在时拒绝覆盖。若模型、序列、文字或处理规则改变，使用新的缓存目录。

本地权重可附加以下参数，省去网络下载。

```bash
--nlp-model /absolute/path/to/Qwen3-Embedding-4B \
--esm-checkpoint /absolute/path/to/esm2_t33_650M_UR50D.pt
```

本地 NLP 权重、配置和 tokenizer 文件按内容计算校验值，远程 NLP 必须使用不可变 commit。IPR domain 缓存也记录校验值。ESM 的官方加载器可能需要相邻的 contact-regression 文件。

使用 TSV 命中表时，增加 `--domain-mode tsv --domain-tsv frozen_protein_ipr.tsv --domain-embeddings "$MZ_DOMAIN"`。TSV 为两列 `protein<TAB>IPR000001;IPR000002`，可有表头。没有命中的蛋白需要明确的空记录，缺行不会悄悄视作没有命中。使用 metadata 时默认字段为 `proteins` 和 `interpros`。

若暂时没有 domain 文本 embedding 字典，可以先运行明确标注的消融。

```bash
python scripts/nbs/export_mzsgo_unseen_scores.py \
  --repo-dir "$MZSGO" --checkpoint "$MZ_CKPT" \
  --input-dir "$INPUT" --go-obo "$LATENCE_DATA/go-2025_06_01.obo" \
  --namespace bp --domain-mode none \
  --output-dir "${MZ_OUT}_no_domain" --device cuda
```

它把 domain 特征置零，输出方法名为 `MZSGO_no_domain_ablation`。它可以检查序列/GO 文字信号，但不能替代完整 MZSGO 的主结果。完整模型中，单个蛋白的 IPR 命中不在 domain embedding 字典内时仍按原方法用零向量，保留该行并在 `feature_coverage.tsv` 统计。

## 与原代码保持一致的处理选择

默认采用论文测试入口所用的 GO 文字形式 `name: definition`、最多 1,024 字符并按 512 字符分块，再做 token 数加权平均。蛋白序列最多保留 2,048 个氨基酸，与 `dataset.py` 的训练/测试预处理保持一致。`feature_coverage.tsv` 逐个报告原序列长度和实际编码长度。

原生任意 FASTA 预测入口与测试入口存在差异，它对 GO 文字采用最多 512 token 的编码，也没有 2,048 aa 的截断。若需要单独复核原生预测路径，可运行另一目录并使用 `--text-encoding source-predict --max-seq-length 0`。两种设置需分别报告，不能混用缓存。默认字符分块设置是对提供的源码测试路径的适配选择，不代表已经验证了作者预计算缓存的真实生成历史。

Qwen 与 ESM 顺序加载、释放，避免同时驻留两套大编码器；默认保持源码 float32 计算。若设备显存不足可以明确使用 `--device cpu`，但 40,384 条序列会明显变慢。程序不会截掉低分项、按测试标签筛词、做 GO 层级传播或用 2025 InterPro2GO 标签替换原始 IPR 特征。

## 公平性与输出

`--training-go-ids` 可给出 checkpoint 的准确监督 GO 集合，`--training-protein-ids` 可给出准确训练蛋白 ID 列表；提供后记录交集。缺少这些文件时，manifest 明确标记训练覆盖尚未核实。蛋白 ID 交集检查不能替代 accession/entry-name 规范化或同源性去重。MZSGO 使用的旧 GO 时点不同于 LATENCE，必须额外检查 2022 年后新增 GO 是否已经进入 MZSGO 的训练标签。

主要输出为 `probabilities.npy`、两个轴文件、`external_prediction_manifest.json`、`go_coverage.tsv`、`feature_coverage.tsv`。不在当前 OBO 中的主 ID、obsolete 或不匹配 namespace 的候选会标记为 unsupported，不用零分伪造支持。统一评估程序会同时报告覆盖率，并在各方法共同可评分的 GO 轴上作成对比较。

## 已执行与待执行的验证

```bash
python tests/test_mzsgo_export.py --repo-dir "$MZSGO"
```

当前开发环境执行了轴匹配、缺行拒绝、GO 覆盖、domain 平均与零向量、缓存恢复和输入哈希检查。它没有 PyTorch、Qwen、ESM 或作者 checkpoint，真实模型推理未执行。测试脚本在有 PyTorch 的环境自动进一步加载附件的实际 `CustomModel`，用合成权重验证 strict 加载、不同分块大小和逐个蛋白调用的数值一致性；这一检查验证适配接口，不等同于作者真实模型精度复现。
