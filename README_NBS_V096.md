# LATENCE v096：从头训练配置（scratch r2）

本包明确用于 **Stage2 从头训练**。入口为 `scripts/nbs/run_nbs_v096_scratch.py`，配置为 `configs/nbs/v096_scratch_single12800.json`。不加载 v095 checkpoint，模型、AdamW 和学习率调度均重新初始化。

“从头训练”指新增 ESM 融合路径、异构 GNN、GO编码器和共享评分器等 Stage2 可训练参数随机初始化；Stage1 和 ESM 分类器作为预训练特征来源仍冻结，复用已导出的缓存。无需重新预训练 ESM2-3B。

## 已核实 v095 的实际学习率

观测台的1220条 v095 训练记录逐点符合 `peak=.003、warmup=5、hold=10、cosine=200、min=.00003`，最大绝对误差 `8.67e-19`，详见 `V095_LR_AUDIT.json`。

| 训练位置 | 实际学习率 |
|---|---:|
| step5，首轮warmup | 0.0000625 |
| E5 / step240 | 0.003 |
| E10 / step480 | 0.003 |
| E50 | 0.00268687 |
| E100 | 0.00163763 |
| E125 | 0.00103282 |

因此，v095 确实使用过0.003峰值，后期约0.001来自cosine衰减。`resume`恢复同一日程与step，不会主动降低峰值。此前 v096 r1 推荐的0.001则是针对加载v095权重另设的新阶段峰值；本配置纠正为从头训练。

这里固定0.003，是为了明确使用 v095 已实际运行的峰值，并让本轮主要比较新增ESM表示。当前数据不能证明仅因从头训练就应改成0.006或0.01；本包不把这些更高学习率写成已验证结果。

## 本轮配置

| 项目 | 配置 |
|---|---|
| 初始化 | `fresh`；禁止`--init-from` |
| 峰值LR | **0.003**，显式固定，不继承源目录里可能不同的值 |
| 完整训练预算 | **200 epochs** |
| 日程 | 5轮线性warmup，保持至10轮，随后cosine到0.00003 |
| 有效batch | **12800**：weak10240 + core2560 |
| 梯度累积 | 2；每次微批约6400，不能再把有效batch乘2 |
| 优化器 | AdamW，矩阵weight decay=0.0001，grad clip=5 |
| 精度 | BF16 |
| 随机种子 | 1096 |
| 蛋白质编码器 | hetero tunedGNN，3层，hidden320，JK sum |
| dropout | 原输入0.1、GNN0.25、ESM分支0.1 |
| ESM融合 | LN → dropout → Linear → tanh门控，加入原节点输入 |
| ESM门控初值 | 0；首步门控学习，之后投影学习，保持r1结构不变 |
| 监督 | 完整沿用v095 ASL + selected teacher KL + GO hierarchy |
| 保存 | 每5轮固定checkpoint；同实验中断后恢复latest |

保留原图、原检索空间、GO原始几何路径和共享GO评分器。提取位置仍为 `ESM2ResNet.output_layer[0]` 的输入；特征维度从manifest读取。多分类器的隐藏向量采用concat，不直接平均。

配置文件是由scratch入口读取的训练profile，不是独立包含全部数据路径的worker配置。入口结合v095目录中的数据路径与既有协议，产生该v096实验自己的 `training_config.json` 和 `resolved_config.json`。`--source-work-dir` 只读取配置，不读取任何源模型权重。若源协议与固定profile不一致，会报错，不静默替换。

## 安装

在现有 LATENCE 工程根目录执行：

```bash
unzip -o LATENCE_nbs_v096_esm_penultimate_scratch_r2.zip -d .
```

这是包含r1全部v096文件的完整更新包，同时新增scratch入口和配置。r1的训练/模型/数据Python文件保持逐字不变；不会覆盖任何v094/v095文件。已安装r1也可覆盖安装，使用新的 `_scratch` 工作目录。

## ESM特征准备

如果已经按r1完成 `outputs/latence_esm_v096/bp_penultimate/manifest.json`，可直接复用，无需重新提取。若尚未提取，先设置实际路径：

```bash
V095=outputs/latence_nbs_experiments/v095/asl_hierarchy_single12800_scratch
INPUT=outputs/latence_nbs_eval/bp_shared_full_task_top512_v071_v2
ESM_SRC=/path/to/esm_models
ESM_BP=/path/to/classifier_checkpoints/BPO
RAW_ESM=/path/to/existing/esm_embedding_cache
ESM_OUT=outputs/latence_esm_v096/bp_penultimate

python scripts/nbs/prepare_esm_ids_v096.py \
  --source-work-dir "$V095" --input-dir "$INPUT" \
  --features-dir "$RAW_ESM" \
  --output-dir outputs/latence_esm_v096/required_ids

CUDA_VISIBLE_DEVICES=0 python scripts/nbs/export_esm_penultimate_v096.py \
  --esm-source-dir "$ESM_SRC" \
  --model-dir "$ESM_BP" --ensemble concat --aspect BPO \
  --features-dir "$RAW_ESM" \
  --ids outputs/latence_esm_v096/required_ids/protein_ids.txt \
  --output-dir "$ESM_OUT" --device cuda:0 --batch-size 512 --dtype float16
```

上述`/path/to`需要替换。多个原始缓存目录可重复`--features-dir`；单分类器用`--checkpoint /path/to/model.pt`替换`--model-dir ... --ensemble concat`。已经存在的导出目录不会被覆盖。缺原始ESM缓存时，按 `docs/ESM_EXPORT_V096.md` 补齐后导出。模型权重与全量数据不在本包中。

## 检查和冒烟测试

`ESM_SOURCE_DIR` 指向原模型源目录，使完整提取测试也参与验收：

```bash
ESM_SOURCE_DIR=/path/to/esm_models \
  python scripts/nbs/run_nbs_v096_scratch.py check

python scripts/nbs/run_nbs_v096_scratch.py smoke --gpu 0 \
  --esm-manifest outputs/latence_esm_v096/bp_penultimate/manifest.json
```

smoke默认3次优化更新，写入独立`_scratch_smoke`目录。早期warmup学习率很小是预期行为；不能据此认为峰值没有设置成0.003。

## 从头正式训练200轮

```bash
python scripts/nbs/run_nbs_v096_scratch.py train --gpu 0 \
  --esm-manifest outputs/latence_esm_v096/bp_penultimate/manifest.json
```

默认复用v095 scratch目录中的数据配置。正式输出固定为：

`outputs/latence_nbs_experiments/v096/esm_penultimate_single12800_scratch`

无需传`--init-from`，入口也会拒绝它。无需传`--learning-rate`，profile明确固定0.003。可用`--source-work-dir`指定不同位置的v095配置，或`--work-dir`指定另一空目录。

若想先运行到20轮，添加`--stop-epoch 20`；完整日程仍为200轮。不要通过`--epochs 20`改变预算。默认训练命令运行到200轮，不设置额外早停。

## 同一次实验中断后恢复

```bash
python scripts/nbs/run_nbs_v096_scratch.py resume --gpu 0
```

这是恢复已经从头启动的v096实验，包括本实验模型、优化器、RNG和调度位置，不是从v095加载权重。不会重启warmup。不要在resume时另外指定LR、ESM缓存或初始化checkpoint。

## 评估

在空闲GPU上使用固定checkpoint：

```bash
python scripts/nbs/run_nbs_v096_scratch.py evaluate --gpu 0 --epoch 5
```

默认测试输入与v095一致，summary写入 `eval_step*/full/metrics_v096/summary.json`。使用同一评估锁。新GO预测入口仍为同一scratch脚本的`predict-go`动作；模型与r1相同，仍需要原框架要求的GO几何与注册表。

## 验证与解释

新增测试验证：即使源配置LR为0.001也显式生成0.003、拒绝权重初始化、固定日程、独立目录，以及resume沿用本实验配置。完整验证结果见 `VALIDATION_V096_SCRATCH.json`。

这是配置与启动方式调整，ESM提取和融合算法未变。真实A100的显存、速度和精度仍需服务器验证。监督过GO的ESM分类器属于上游表示来源，Stage2从头训练不会消除其上游监督暴露；不会因此变成严格全流程zero-shot。
