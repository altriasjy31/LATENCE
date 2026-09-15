# NBS v0.7.1：修订说明与顺序实验指南

日期：2026-09-10。基线：此前交付的 v0.7.0 alignment 完整源码；合入用户附件中的 v0.6 epoch-series 评估工具。可从当前 v0.6 直接更新到本包，无需先安装 v0.7.0。

本包包含源码、配置、回归契约和历史聚合指标，不包含生产数据、第一阶段模型或 NBS checkpoint。本地完成的是 CPU 契约验证；真实数据的审计、CUDA/DDP smoke、重训和独立测试需要在训练服务器执行。

## 1. 本轮修复

1. **candidate evidence 与标签解耦。** `candidate_evidence_scope=decoded_all` 为全部 `[Q,C]` 位置从第一阶段稀疏候选提取特征，再建立监督 mask；训练与推理共用 `align_candidate_evidence`。正例、hard PU、unknown 均按同一成员关系取值。分块提取避免一次展开所有候选边。旧配置保留 `sampled_hard` 采样口径，便于原样审计旧数据流。
2. **解除双零梯度锁。** 残差 MLP 最后一层保持零初始化，乘法 scale 的默认值及新配置改为 `0.1`。初始证据仍等于原来的 reciprocal rank；`graph_delta_scale=0` 仍使初始最终 logits 精确等于 base。不能要求第一步所有分支都有非零梯度，应观察前20步是否逐步启动。
3. **正负 query 共同分配。** 新主配置 Q=144：96个正例 query 名额、16个 hard query 名额、16个层级 GO、16个覆盖 GO。hard query 根据 primary weak 的 candidate-minus-pseudo 集合，为缺少对照的蛋白列分配；不是只等待负 GO 偶然落入正例 query bank。未用满名额仍由现有 filler 补齐。
4. **补齐已解码正标签。** 每个 primary weak 先争取4个正 GO；其余已知 pseudo GO 若也进入当前 query 集合，同样加入 pseudo 与 column loss，因此实际正 GO 数可以超过4。未知位置不补成负例。
5. **修复轮换优先级。** epoch 轮换后的 GO 排名参与共享 query 的选择，而非仅影响选定 GO 后的蛋白排序。共享覆盖仍会影响选择，不承诺每个蛋白每轮都获得新标签；用固定 cohort 审计实际去重覆盖。
6. **区分新增负 pair 与实际列监督。** 已有 row-hard 计入每蛋白 hard 配额；column mask 包含该 primary 蛋白所有已监督位置。新增 `weak_primary_total_hard_pairs`、每蛋白 hard 数及配额满足率。
7. **可复核的梯度与残差日志。** 前20个优化步骤按 rank 写 `gradient_probe_rank*.json`，记录解除 AMP scale 后、裁剪前的关键参数梯度。增加 primary positive/PU/unknown 的概率修正，以及 PU 高于 ASL clip 的比例。该比例是活跃负分数诊断，不等于实际梯度大小。
8. **推理消融与旧模型兼容。** 增加 candidate-only correction：candidate 内保留 NBS，外部精确保留输入 base 概率，同时将外部 applied delta/gate 置零。v0.6 checkpoint 仅允许补齐明确关闭且整组缺失的 GO residual query 参数；所有已训练参数严格加载，其他缺失、额外参数或形状错误仍报错。不迁移旧优化器状态。
9. **评估元信息与实验入口。** 修复 candidate selector scope 的读取位置；合入 epoch-series 工具；阶段启动器清理旧 NBS epoch/Q/resume/截断变量，生成具体配置并调用原训练和评估程序。

### 三组配置

| variant | Q | 强制正GO预算 | column loss | base anchor | GO residual query | target编码 |
|---|---:|---:|---:|---|---|---|
| `correctness_control` | 64 | 1 | 0 | supervised | 关闭 | joint_local |
| `sampling_loss` | 144 | 4，并补齐query内已知pseudo | 0.25 | unknown_decoded | 开启 | joint_local |
| `inductive_aligned` | 144 | 同上 | 0.25 | unknown_decoded | 开启 | isolated_similar_to_core |

三者均使用全解码证据和非零 evidence scale。正确性对照与后两组的比较仍是一个采样/loss/query增强组合实验，不能独立归因于 Q。新主配置 `max_candidates=7056` 是完整监督的保守上界，不是强制填充蛋白数；显存以真实 smoke 为准。

`inductive_aligned` 对齐的是 target 隔离方式和 similar_to 运算。训练邻居来自已有 similar_to 索引中的 core 子集，而测试可能来自 core-only 检索；两者的邻居分布尚不能视作完全相同。新增 core/weak 分组邻居覆盖日志用于识别这种差异。

## 2. 安装和共同条件

在训练服务器项目根目录执行。请将压缩包路径替换为实际下载位置。

```bash
cd /home/dataset-local/data_local/shaojiangyi/latence-project

tar -czf ../latence_nbs_code_before_v071_20260910.tar.gz \
  nbs_models/nbs_protein_go scripts/nbs experiments/nbs

tar -xzf /path/to/latence_nbs_v071_alignment_20260910.tar.gz \
  --strip-components=1 -C "$PWD"

export LATENCE_PROJECT_ROOT="$PWD"
python scripts/nbs/run_nbs_v071_contract_tests.py --require-pyg
```

目标服务器沿用 PyTorch 2.4、PyG 2.6.1 和此前成功运行的环境，无需为本包升级框架。契约脚本不依赖 pytest。`--require-pyg` 不允许跳过图相关契约。

评估继续使用 epoch16–20 那次成功运行的第一阶段模型、metadata、MSA/ESM 相关环境变量，以及**同一份 full-task top-512 输入缓存**。以下检查不会替你猜路径：

```bash
: "${STAGE1_CHECKPOINT:?请设置为epoch16-20评估使用的同一个第一阶段checkpoint}"
: "${METADATA_FILE:?请沿用此前成功评估的metadata路径}"
: "${NBS_IND_TEST_WORK_DIR:?请设置为epoch16-20使用的full-task输入缓存目录}"
```

缓存目录中应已有 `ind_test_input_manifest.json`，其中 `candidate_evidence.selector_scope=full_task`、`fixed_k=512`。若缓存不存在，原评估程序可重建；但本轮应优先复用已有的正确缓存。不要复用早期 rare-only 缓存。

默认旧训练目录：

```text
outputs/latence_nbs_train/bp_nbs_v060_perfopt_tailfix_fullrestore_formal
```

如果实际位置不同，给 `baseline-ablation` 和 `audit` 同时传入 `--checkpoint-dir /实际目录`。默认从该目录读取 `resolved_config.json`；若配置另存，追加 `--baseline-config /实际配置.json`。

所有新结果默认保存至 `outputs/latence_nbs_experiments/v071`。可以给各命令一致追加 `--work-root outputs/latence_nbs_experiments/v071_run2` 建立新实验。训练目录非空时不会覆盖已有实验。`--dry-run` 只展示将使用的命令和关键配置，不执行训练或评估。

## 3. 第一步：固定 epoch20 的推理消融

```bash
python scripts/nbs/run_nbs_v071_experiments.py --stage baseline-ablation
```

依次评估三种设置，每种只评 epoch20，共用输入缓存：

| 输出目录 | candidate evidence | candidate外输出 | 目的 |
|---|---|---|---|
| `baseline/full` | 开启 | NBS | 核对旧模型复现 |
| `baseline/evidence_off` | 关闭 | NBS | 检查证据通道的影响 |
| `baseline/candidate_only` | 开启 | 精确保留base | 检查候选外修正的影响 |

先看 `full` 是否复现 Fmax约55.7068、AUPRC约28.0646；同次 backbone 应约55.6317、32.0087。明显不一致时，应核查 checkpoint、缓存、GO chunk和环境，而不是解释为模型优化效果。

再看相对 `full` 的 AUPRC、P/R@10/50/100：

- evidence-off恢复较多：与证据输入错位或证据门控不合适相容，但不能证明它是唯一原因。
- candidate-only恢复较多：说明候选外修正造成了损失；这项消融不把候选外GO置零。
- 两者仍明显低于backbone：候选内部和图残差自身的排序也需改进。

固定报告三种设置，不根据独立测试挑选“最佳epoch”。包内 `experiments/nbs/references/v060_epoch16_20_reference.json` 保存了用户最新聚合结果及原文件SHA256。旧1–4轮候选口径不同，不能直接拼接为连续曲线。

## 4. 第二步：采样审计

```bash
python scripts/nbs/run_nbs_v071_experiments.py --stage audit --num-gpus 2
```

此阶段不运行GNN或GPU；`num-gpus` 在这里指定与正式DDP相同的分片数。默认分别审计 legacy、correctness_control、sampling_loss，每个rank32个episode；另以相同、固定顺序的primary cohort重放3个epoch、每轮最多8个episode。

输出位于 `audit/*/anchor_alignment.csv` 和 `alignment_summary.json`。重点检查：

| 指标 | 解读 |
|---|---|
| `evidence_mismatched_pairs` | 新配置必须为0；旧配置用于显示错位规模 |
| `all_query_positives_supervised` | 新sampling配置应为1 |
| `loss_pseudo_hits` | 每个蛋白真正进入loss的正GO分布，通常应明显高于旧版 |
| `positive_quota_met` | 配额按该蛋白实际可用pseudo数封顶，避免惩罚只有1个标签的蛋白 |
| `hard_pu_go` / `hard_quota_met` | 实际hard监督数，不是配置值4 |
| `background_pu_go` | 与hard分开看，低分背景不一定有有效梯度 |
| `cross_epoch_repeated_proteins` | 没有重复蛋白，就不能讨论跨epoch覆盖增长 |
| `cross_epoch_new_positive_go_after_first` | 相同蛋白新增的去重正GO数 |
| `cross_epoch_adjacent_positive_jaccard` | 配对重复程度；若已覆盖全部pseudo，重复本身不是缺陷 |

固定cohort是用于隔离GO轮换的受控审计，不可用它声称生产epoch覆盖率达到100%。`positive_effective_weight` 等只包含confidence×supervision_weight，未含ASL因子或分支系数，不能冒充梯度比例。

如果hard配额普遍填不满，先看共享hard query数量和candidate-minus-pseudo可用集合，再调整预算；不要直接增加背景负样本。邻居gold GO在后续路径扩展实验中只能作为query候选和图证据，不能直接成为weak蛋白的gold标签。

## 5. 第三步：20-step CUDA/DDP smoke

先运行正确性对照，再运行采样与loss版本：

```bash
python scripts/nbs/run_nbs_v071_experiments.py --stage smoke --variant correctness_control
python scripts/nbs/run_nbs_v071_experiments.py --stage smoke --variant sampling_loss
```

每次先执行单进程配置预检，再用默认2卡跑1个截断epoch、20步并保存epoch1。该smoke固定学习率，不使用OneCycle；这是语义和梯度验证，不用于比较预测性能。

结果目录为 `train/<variant>_smoke20`。自动分析检查：20步完成、loss有限、无cap丢失、完整保留监督、正例和unknown位置均有candidate evidence、关键梯度在前20步启动。sampling版本还检查多正列、column loss和hard列监督。检查未通过时，入口返回错误；后续probe也会再次读取smoke结果。

重点文件：

- `training_history.json`：实际pair数、loss贡献、残差方向、显存和耗时。
- `gradient_probe_rank0.json`、`gradient_probe_rank1.json`：每卡裁剪前梯度；第一步部分分支为0正常，但20步内应有启动。
- `analysis/experiment_analysis.json`：逐项检查及结果。

`weak_primary_positive_pairs_requested`是强制匹配数；`retained`包括补齐的已解码pseudo，因此后者可以更大。`weak_primary_hard_pairs_retained`是额外加入的column-hard数；判断实际列负监督请看 `weak_primary_total_hard_pairs`、`weak_primary_hard_go_per_anchor_mean`、`weak_primary_hard_quota_met_rate`。

若OOM，先根据CPU/GPU日志识别query/candidate union或局部图的增长；不要单独调小max_candidates并允许静默丢监督。降低配额、Q或primary蛋白数后，应同步修改上界并重做smoke。新入口会清理旧的NBS环境覆盖，所以参数调整应写入相应JSON，或使用原训练入口显式启动一个另命名实验。

## 6. 第四步：从头训练1–3轮并固定评估

默认3轮，保留epoch1、2、3。从同一第一阶段产物新建NBS，不继承v0.6 epoch20或smoke的权重、优化器、scheduler。

```bash
python scripts/nbs/run_nbs_v071_experiments.py --stage probe --variant correctness_control
python scripts/nbs/run_nbs_v071_experiments.py --stage evaluate --variant correctness_control

python scripts/nbs/run_nbs_v071_experiments.py --stage probe --variant sampling_loss
python scripts/nbs/run_nbs_v071_experiments.py --stage evaluate --variant sampling_loss
```

若先只跑1轮，在对应的 `probe`、`evaluate` 命令都加 `--probe-epochs 1`。不同长度探针使用新work-root；3轮OneCycle与20轮OneCycle的轨迹不同，不能把短探针直接视作20轮训练的前三轮。

先判断仅修正确性是否减少排序损失，再比较sampling_loss是否进一步改善。两组必须共用第一阶段产物、外部邻居、candidate缓存、评估GO chunk和报告epoch。相同epoch代表相同weak-primary遍历，但Q不同会改变监督量与计算量，还应比较记录的耗时、pair数和GPU峰值。

如果采样与loss版本稳定，再开展编码对照：

```bash
python scripts/nbs/run_nbs_v071_experiments.py --stage smoke --variant inductive_aligned
python scripts/nbs/run_nbs_v071_experiments.py --stage probe --variant inductive_aligned
python scripts/nbs/run_nbs_v071_experiments.py --stage evaluate --variant inductive_aligned
```

检查 `isolated_weak_with_core_neighbor_fraction`、`isolated_core_with_core_neighbor_fraction`和邻居数。如果大量weak只能走feature-only，不应把该结果解释成图传播无效，应先统一训练/测试的core邻居检索口径。

## 7. 汇总与后续判据

```bash
python scripts/nbs/run_nbs_v071_experiments.py --stage analyze
```

生成：

```text
outputs/latence_nbs_experiments/v071/analysis/experiment_analysis.json
outputs/latence_nbs_experiments/v071/analysis/experiment_comparison.tsv
```

主指标和分层micro histogram指标分列保存，不混用绝对值。判断顺序：

1. evidence一致、全部query内已知正标签入loss、关键梯度启动。
2. 每蛋白正GO与hard PU覆盖增加，跨epoch统计有真实重复cohort作为分母。
3. 相对backbone的AUPRC和Top-k退化明显减轻，不能只看最佳阈值下Fmax增加。
4. candidate内部排序改善，候选外不再普遍产生无益抬升；rare分层有方向一致的证据。
5. 再比较邻居丰富/稀疏蛋白、邻居打乱或source消融，验证图证据是否提供额外信息。

本轮没有把 weak→core→GO 的新query来源混入主配置。下一步若直接weak→GO多标签覆盖和输入修复已奏效，再单独引入“可靠core邻居GO补充query”，重点测试候选外真实标签恢复和长尾召回；邻居标签不自动转成weak监督，PU不当作可靠生物学负例。

满足上述方向后再规划20轮正式实验及其独立配置；本入口不会自动启动长训练或选择最佳checkpoint。若持续根据当前ind_test修改模型，应将它视为诊断集，最终研究结论另用未参与迭代的评估数据确认。

## 8. 本地验证与边界

具体结果见包内 `VALIDATION_V071.json` 和 `RELEASE_MANIFEST_V071.json`。

已验证：全解码证据训练/推理一致、轴重排不改变对应关系、query内正标签补齐、hard预算与pseudo排除、轮换实际影响GO选择、双零反例、完整matcher梯度逐步启动、初始base精确保持、column等权与unknown anchor梯度方向、candidate-only导出及诊断数组一致、旧checkpoint仅补齐关闭分支、旧v0.6与新关闭分支query的CPU数值一致、实验配置/环境隔离和梯度日志。

本地没有生产mmap数据、旧NBS/第一阶段checkpoint、CUDA或PyG；PyG图构造/isolated卷积契约列为跳过，必须在服务器通过 `--require-pyg` 与真实20-step DDP smoke。CPU契约通过不代表真实数据性能提升。
