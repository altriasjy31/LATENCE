# LATENCE NBS v0.8.6：受控的 PP encoder 归一化实验

本版依据已上传的完整 v085 项目、12 份 schedule/lowlr/aligned 结果、tunedGNN 论文与实际源码制定。它是依赖现有 v084/v085 的**增量实验包**，不是原完整模型迁回版，也不声称已经提升真实任务性能。正在运行的 v085 dynamic 应按原先预算完成；本包不修改旧运行文件。

## 1. 当前结论与交接状态

代码、指标定义、三组实验的配置和预测来源已能对上，足以继续受控开发。服务器上的真实矩阵尚未搬入本地；下面的数值来自六份已经由服务器计算的标准指标 JSON，不是从旧指标换算，也不是本地重新计算矩阵。

- lowlr、aligned 从 600 到 2000 步均提升；schedule 仍回退。
- aligned 在 2000 步未超过 lowlr，首跳全部改为检索的独立收益尚未成立。
- G 对 B 的小幅提升存在，但距离 E、M 仍大；这些 full 输出不能单独证明新的图特异性收益。
- 三组的 `development_contract` 均为 null。core holdout 已被 Stage1 见过，只是 monitor，不能代替独立开发集。
- 尚缺 dynamic 结果、新实验的图干预、标准指标的配对不确定性估计，以及 Stage1 未见过的开发数据来源。无需为继续代码开发再上传一次相同完整版；服务器依赖若与本包检查值不同，应提供差异文件。

最终目标仍是 G 同时超过 E 的 **protein-centric Fmax 与 exact micro PR-AUC**；M、B 单独比较，exact micro AP 另列。保留 weak–GO、weak–core–GO、全部 21,312 个 BP GO 和全部已知正例。不重新加入 E/M 概率直连、dense teacher loss 或 v082 双塔。

## 2. 12 份结果的实际含义

每组文件为 step600 comparison、step2000 comparison、step2000 resolved_config、step2000 training_history。六份 comparison 的 E/M/B、蛋白与 GO 身份、metadata、标准指标实现、Stage1 reference generation 一致；预测源码哈希与上传项目匹配。均为 fixed sampler 下的 full 输出，checkpoint step 与文件名相符。

数值单位为百分数；差值为百分点。Fmax 是 v085 的逐蛋白 P/R 聚合后、0.001 阈值网格上取最大值，并非“每个蛋白各选最佳阈值再平均”，也不是精确遍历所有阈值。PR-AUC 是所有 protein–GO pair 的精确经验曲线梯形面积，AP 是另一种精确面积定义，不能互换。

| 方法 | 步数 | Protein Fmax | Exact micro PR-AUC | Exact micro AP |
|---|---:|---:|---:|---:|
| E | — | 71.690220 | 67.808017 | 67.554690 |
| M | — | 71.644822 | 68.811595 | 68.790736 |
| B | — | 57.723149 | 50.621882 | 49.450788 |
| schedule G | 600 | 58.129510 | 51.953354 | 51.946546 |
| schedule G | 2000 | 57.724437 | 51.605834 | 51.599665 |
| lowlr G | 600 | 57.795986 | 51.688238 | 51.685364 |
| lowlr G | 2000 | 57.975766 | 52.080227 | 52.076289 |
| aligned G | 600 | 57.817400 | 51.653894 | 51.650489 |
| aligned G | 2000 | 57.881132 | 51.888288 | 51.884631 |

| 对照 | ΔFmax | ΔPR-AUC | ΔAP |
|---|---:|---:|---:|
| schedule：2000 − 600 | −0.405073 | −0.347520 | −0.346881 |
| lowlr：2000 − 600 | +0.179780 | +0.391989 | +0.390925 |
| aligned：2000 − 600 | +0.063732 | +0.234394 | +0.234142 |
| 2000：lowlr − schedule | +0.251329 | +0.474393 | +0.476624 |
| 2000：aligned − lowlr | −0.094634 | −0.191939 | −0.191658 |

schedule→lowlr 只把峰值/最低 LR 从 3e-4/3e-5 同比降到 1e-4/1e-5；lowlr→aligned 只把 seed native dropout 从 0.25 改为 1.0。模型、loss、world size=2、seed=8080、sampler seed=8084、global weak/core batch=128/32 相同。cosine horizon 均为 4000，实际只完成前 2000 updates；不能说完整 cosine 日程已经验证。

三组 Top-10/50/100 的召回变化与上述方向相同，因此不只是 Fmax 阈值移动。lowlr2000 对 B 的 Fmax/PR-AUC 增量是 +0.252617/+1.458345，对 E 仍差 −13.714453/−15.727790。没有这些新标准指标的置信区间或多种子结果，暂不称差异显著。

### 2.1 是否“没有学完整图”导致回退？

现有证据不支持这个单独解释。到 2000 步，三组均约经历 0.523280 次 weak、1.076064 次 core 的等效监督遍历；schedule/lowlr 每个日志窗口的图采样工作量也一致，但只有高 LR 组继续回退。更符合现有证据的解释是：优化强度与训练目标的拟合影响了泛化轨迹。它尚不足以区分过拟合、目标冲突与校准等具体原因。

监督遍历、节点/边累计访问率、每个种子的邻居覆盖，是三个不同量。`sampled_new_neighbor_fraction=0` 符合 fixed 模式，不能读作“没有见到新图”；dynamic 的这个量也不是全图累计覆盖率。mini-batch GNN 不要求每步访问整张图，但传播深度必须有对应采样支持。

### 2.2 比增加 encoder 容量更需要关注的信号

对 80 个训练日志窗口累计，三组均有 64,000 次 core 监督曝光、3,372,561 个 core gold pair 曝光，其中冻结 B 概率 <0.5 的已知 gold 仅 5,272 个，即 **0.156320%**，平均每次 core 曝光 0.082375 个。这里是曝光次数，不是去重蛋白或去重标签数，也不能把其余 gold 全称为“无用”。B≥0.5 的 gold 仍可能存在排序困难；原 loss 先按行归一再平衡 weak/core，两角色各占一半，pair 数量不等于梯度贡献。

| step2000 日志窗口 | schedule | lowlr | aligned |
|---|---:|---:|---:|
| loss | 0.079107 | 0.089020 | 0.090890 |
| absolute logit delta | 0.801906 | 0.612114 | 0.526571 |
| PP message norm | 7.572328 | 5.087233 | 1.871285 |
| hard-PU 加权贡献 | 0.032709 | 0.036403 | 0.036145 |
| positive 加权贡献 | 0.035864 | 0.040814 | 0.042803 |
| hard-PU 平均剩余权重 | 0.972673 | 0.972533 | 0.972511 |

schedule 的训练目标拟合更强，但外部指标更差。hard-PU 的 0.972 表示仍保留约 97.2% 的惩罚，不能解释为降低了 97.2%。直接 decoder core 支持与 loss 固定 top-8 支持使用不同来源/规则，值得检查冲突；这些摘要不能证明其中哪些 unknown 是假阴性。aligned 改变了关系组成，不能只凭 PP norm 更小就判定更稳定或更优。

## 3. 论文与专家意见的核验

阅读版本：[论文 v2](https://arxiv.org/html/2406.08993v2)，以及 tunedGNN commit `23f9604e8b13a9a6d3faa2f691cd844006979153`。论文支持认真控制归一化、dropout、残差和优化配置；它没有验证本项目这种 21,312 列、PU、二值 weak 证据与 B 残差的组合。不能把 benchmark 的增益数值移植到 LATENCE。

| 专家讨论中的判断 | 实际核验 | 本轮处理 |
|---|---|---|
| scheduler 缺 `raise` | 上传项目 `full_task_schedule_v085.py` 已有 `raise ValueError(...)` | 不修改旧调度器，不把不存在的 bug 当作改善原因 |
| checkpoint 浮点往返会误报 | 当前保存的是同一配置算出的 Python 浮点值；正常 pickle 往返不会必然改变它 | 保留严格契约；若服务器真报错，再根据实际差值定位 |
| v084 完全没有 norm | `protein_encoder`、GO 编码已有 LayerNorm，matcher key/value 也做向量归一；PP 两层残差内部确实没有 norm | 实验只补 PP 消息源的逐层 LN |
| large_graph 必须 post-BN | `models.py` 是 BN 路径；`product.py` 的卷积支路使用预 LN，并支持残差；`lg_model.py` 也有可选 pre-LN | 不把单个文件推广成统一配方 |
| BN 普遍优于 LN | 论文与这些源码不支持针对本项目的该结论 | 本轮先用逐节点 LN；BN 留作独立、有明确统计策略的后续对照 |
| pre-norm 会破坏恒等路径 | 若只归一化消息支路输入，残差仍直接加原 state；反而整体 post-norm 会变换残差输出 | 保留原 state 的恒等支路 |
| “密残差＋JK”是必要成分 | `models.py` 的层间残差不是所有历史层的 dense 连接；JK 是可选项。论文附录 B.3 说明最终移除了 JK 搜索 | 本版不加 JK |
| per-edge gate 零初始化等于零消息 | 原 gate 的最后一层为零，但乘子是 `2*sigmoid(0)=1`；source projection 非零 | 保留原置信度与门控语义 |
| mean-of-means 必须换 sum | 均值可能稀释单一强关系，也可能有效控制多关系度数尺度；当前没有因果证据认定为 bug | 本轮保留，单独诊断后再试 sum/关系权重 |
| PP DropEdge 值得尝试 | 合理的正则化假设；上游的实现位于 GAT 路径，不是所有 SAGE/GCN 自动使用 | 提供单独可选实验，不与首轮 LN 同时开启 |
| Optuna 30–50 次是论文必须步骤 | 论文描述验证集选参与搜索空间；已核验材料未证实专家所述的具体 Optuna 流程。当前更缺独立 dev | 不发起大规模 sweep |

对应源码：[models.py](https://github.com/LUOyk1999/tunedGNN/blob/23f9604e8b13a9a6d3faa2f691cd844006979153/large_graph/models.py)、[product.py](https://github.com/LUOyk1999/tunedGNN/blob/23f9604e8b13a9a6d3faa2f691cd844006979153/large_graph/product.py)、[lg_model.py](https://github.com/LUOyk1999/tunedGNN/blob/23f9604e8b13a9a6d3faa2f691cd844006979153/large_graph/lg_model.py)、[protein.py](https://github.com/LUOyk1999/tunedGNN/blob/23f9604e8b13a9a6d3faa2f691cd844006979153/large_graph/protein.py)。本版是在现有 LATENCE 实现上独立改写，只借鉴受控经典 GNN 的原则，不宣称复现 tunedGNN。

正确启用 running statistics 并进入 eval 的 BN 通常可保持推理批次不变；这里不把 BN 一概判错。问题在于训练子图节点组成、共享邻居和 DDP rank 会影响其统计，而且跨节点统计会引入额外训练依赖。本轮逐节点 LN 更便于保持已经验证的局部感受野和批次契约。

## 4. 本版代码设计

### 4.1 最小的 encoder 改动

新增 `nbs_pg/full_task_model_v086.py`，继承完整 v084，配置包含独立的 `encoder_variant`；`full_task.variant=fixed/dynamic` 仍只表示采样方式，两者不混用。

原消息算子仍是按正边数归一的 weighted relation SAGE。令 `Agg_r` 表示这个带置信度、学习门控及 source projection 的算子：

```text
legacy: source = state
preln:  source = LayerNorm_layer(state)
message = sum_r Agg_r(source) / max(number_of_active_relations, 1)
state_next = state + pp_context_scale * dropout(SiLU(message))
```

归一化只作用于消息源，不直接归一化已聚合的 message，避免把很小的边置信度重新放大。两层共有 512 个新增 LN 参数（hidden_dim=128）。这与上游 `product.py` 的预归一思路相关，但并非照搬其 ReLU/预测头，也没有改变本项目现有 SiLU 与 decoder。

公共模块先由父类构造，再创建 LN，因此同一 seed 下公共参数初值可对齐；legacy 使用原 `_encode_sampled`。预设中新增加的随机正则默认关闭。最终 correction head 继续零初始化，初始 G=B；后续验证梯度时必须打开该 head，避免被初始零输出掩盖问题。

### 4.2 保留大图训练契约

- 固定两层，首跳最多 32 个不同蛋白、后续通常每蛋白最多 4 个；原生池和检索池仍各按既有 64 预算。同一局部 batch 中兼为监督 seed 的节点仍按 seed 预算处理，不能把合并子图机械算成每 seed 恰好 161 个节点。不先扩大邻居池、扇出或深度。
- seed 读取第二层；core anchor 仍读取第一层。anchor 距 seed 已有一跳，若读取它的第二层会引入第三跳，并可能让结果依赖同批其他外部查询。
- 保留 sampled candidate、binary pseudo、core gold 的独立注入；不改变监督 seed 标签遮罩与 holdout 遮罩。
- 保留 GO 编码、normalized key/value、exact annotation incidence、vote、query-conditioned decoder、完整 GO 列与分块 activation checkpoint。
- 不新增全图 attention，也不复制巨型 `[batch, GO, neighbor, hidden]` 张量；仍使用现有采样和 GO chunk。这是可在当前训练框架验证的增量，不是另造一条大图 pipeline。
- 推理始终确定性；E/M 仅作为外部比较引用，二值 pseudo 来源沿用原契约。

### 4.3 可选 DropEdge 的准确含义

`preln_dropedge` 在 preln 上只增加 `pp_edge_dropout=0.1`。训练时对所有已采样 PP 关系的正边作伯努利保留，每层独立掩码；保留边仍按保留正边数归一。它不会删掉 sampled GO 证据、decoder 的 anchor incidence 或直接 core vote，所以是 **PP 消息支路的 DropEdge**，不能冒充“全部邻居证据缺失”。它也不等价于 seed 原生关系缺失增强。

掩码使用由 encoder seed、optimizer step、rank、layer 确定的独立随机生成器；不消耗全局 torch 随机流，以免顺带改变 candidate dropout 和 PU 随机采样。runner 每步设置上下文，resume 按恢复后的 step 继续；推理不丢边。此时 `pp_edge_count` 是两层保留正边数的均值；dropout=0 时与旧定义相同，原始 sampled edge 计数仍由采样器记录。

### 4.4 版本与恢复

新增 v086 runner/evaluator/shell 和配置；复用原 v085 标准指标、固定 cosine 调度器、独立 dev 支持，以及 v084 数据与 v081 loss。新增 `local_loader_v086.py` 显式允许 v086 阶段读取相同的、经过来源审核的二值监督；保留原来源和遮罩契约，不伪装阶段名称。旧文件逐字节保留，旧 dynamic 的源码身份不受影响。

v086 是 fresh run。不能拿 v085 checkpoint resume 到 v086，也不能把 legacy checkpoint resume 为 preln 或 DropEdge。需要同版本、同模型/数据/loss/日程/实现身份恢复；改变停止预算不会改变 horizon。`runner_version` 为 0.8.6，标准指标 schema 继续是 0.8.5 标准指标定义，二者不是同一个版本概念。

## 5. 下一轮实验：先两组，第三组有条件追加

第一轮暂定公共设置沿用既有 lowlr 参数：1e-4→1e-5、25 warmup、4000 cosine horizon、fixed、native_dropout=0.25、hidden128、两层、32/4、candidate_dropout=0.15、原 loss。用途是固定优化条件、不新增 LR 搜索，不宣称该 LR 最优。这些测试结果已参与探索性研究判断，不能因 lowlr 曾预先定义就称后续设计对原 ind_test 仍是盲选验证。新的模型选择需先冻结可信 dev 和协议，最终确认需新的保留测试；缺 dev 时只进行固定预算的工程/机制探索，不选“获胜配置”。

| 顺序 | preset | 相对上一对照的变化 | 预算与用途 |
|---|---|---|---|
| 必做 A | legacy | v086 runner 中的原 encoder | 20 步 smoke；fresh 600→2000，作为同 runner 对照 |
| 必做 B | preln | 仅 PP 每层消息源 LN | 同上、同种子、同采样、同预算 |
| 条件 C | preln_dropedge | 在 B 上仅 PP DropEdge=0.1 | 有可信 dev 且需要检验正则化时再开；同预算 |

不自动把 schedule/lowlr/aligned 都续到 4000，不默认启动 C，不做 30–50 次搜索。600 是运行/梯度与开发指标检查点，不用已知独立测试的 600 排名淘汰慢启动的模型。没有训练错误或预先定义的资源停止条件时，A/B 均完成相同 2000 预算；如果 dev 表明值得检查日程后半程，再事先成对约定 4000，而不是只续测试集“赢家”。

已有两卡记录中，每组 2000 步约 85–109 分钟，peak allocated 约 1.57–1.59 GB。它们不是稳定速度基准，不能把组间时间差归因于 LR；新 encoder 的速度/显存由 smoke 和日志实测。首先记录每秒监督蛋白、采样节点/边、allocated/reserved 峰值（若有）及评估时间，再决定是否需要性能工程。

### 5.1 开发集与图特异性验证

可信 dev 应在读取结果前冻结，包含 Stage1 未见过、Stage2 不监督的 gold 蛋白。本包沿用的 `DevelopmentSetV085` 还要求它的蛋白 ID 与整个 Stage2 graph registry、完整 Stage1 训练 ID、最终测试 ID 隔离；不能直接拿图内 weak/core 改名作 dev。另需核对时间、序列/同源组重叠、候选缓存和图标签遮罩，避免通过邻居标签回流。同一 dev manifest 用于 A/B。原 core holdout 继续 monitor。

若当前没有这样的 dev，可以先运行 A/B 的工程可行性和固定预算训练，但**不得据 ind_test 的差异选 encoder、dropout、checkpoint 或扩展预算**。已有 ind_test 被多次查看，对后续自适应设计只能作为探索性诊断；最终确认应留新的、未参与这些决策的测试集，或明确报告这一局限。

在 dev 上先报告完整 G 与 E/M/B，再在固定 checkpoint 做 `pp_off/core_off/weak_off/graph_off/go_shuffle`。这些是推理干预，不是重训无图基线。`weak_off` 同时关闭多种 weak 输入，仍不能据它否定 weak–GO。

判断 encoder 图收益时，既看 `preln_full − legacy_full`，也比较各自 `full − pp_off`，并看 core/GO 干预方向。跨随机种子至少做一次预先固定的配对复验；在 dev 上用蛋白（有同源簇时按簇）配对重采样估计不确定性。每次重采样须重新计算相应 Fmax/PR 曲线，不能平均每蛋白独立最优 F1，也不能把 micro AP 当 PR-AUC。最终需要时再增加预先登记的无图重训对照；该训练模式本包未实现。

dynamic 完成后，纯采样对照是 **dynamic vs aligned**，因为两者同为 native_dropout=1、低 LR；dynamic vs lowlr 同时改变采样策略和缺失增强。不要因 dynamic 当前尚未返回就把 A/B 改为另一套采样。

## 6. 安装与运行

本包在已安装 v085 的项目上使用。先解压到单独目录；示例以项目绝对路径 `/path/to/latence-project` 表示服务器上的实际项目，请替换。

```bash
unzip latence_nbs_v086.zip -d /tmp/nbs_v086_review
python /tmp/nbs_v086_review/latence_nbs_v086/install_v086.py \
  --project-root /path/to/latence-project --check-only
python /tmp/nbs_v086_review/latence_nbs_v086/install_v086.py \
  --project-root /path/to/latence-project
cd /path/to/latence-project
```

安装器先检查包内文件和关键既有依赖的 SHA256，再写入仅 v086 文件。发现旧依赖不匹配会停止，应提供实际差异核对，不应删除检查绕过。重复安装相同内容不改文件；同名 v086 文件若内容不同也会停止，以免影响已开始的新实验。包内 `DEPENDENCIES_v086.json` 与 `package_files.json` 用于核对。

先单卡检查两个模型，smoke 目录与正式目录分离：

```bash
NUM_GPUS=1 SMOKE_STEPS=20 bash scripts/nbs/run_nbs_v086.sh smoke legacy
NUM_GPUS=1 SMOKE_STEPS=20 bash scripts/nbs/run_nbs_v086.sh smoke preln
```

如有已审计的独立开发集，将同一 `DEV_MANIFEST=/absolute/path/to/development_manifest.json` 加到以下每个训练/恢复命令；没有则不要伪造一个。

```bash
NUM_GPUS=2 bash scripts/nbs/run_nbs_v086.sh pilot legacy
NUM_GPUS=2 bash scripts/nbs/run_nbs_v086.sh pilot preln

NUM_GPUS=2 STEPS=2000 \
  RESUME=outputs/latence_nbs_experiments/v086/legacy/nbs_step600.pt \
  bash scripts/nbs/run_nbs_v086.sh train legacy
NUM_GPUS=2 STEPS=2000 \
  RESUME=outputs/latence_nbs_experiments/v086/preln/nbs_step600.pt \
  bash scripts/nbs/run_nbs_v086.sh train preln
```

也可以两个 preset 各自 fresh `STEPS=2000 ... train`，无需先 pilot；不要把 smoke checkpoint 当正式初始化。`STEPS=2000` 表示累计停止步，不是从恢复点再加 2000。训练不会自动执行独立测试。公共 cache 沿用现有路径；同机并发首次准备缓存前应先串行 prepare，避免重复构建。

固定协议后显式评估。下面默认使用原独立测试输入，因此是审计/最终固定报告命令，不是选择超参数的循环：

```bash
CHECKPOINT=outputs/latence_nbs_experiments/v086/legacy/nbs_step2000.pt \
  bash scripts/nbs/run_nbs_v086.sh evaluate legacy
CHECKPOINT=outputs/latence_nbs_experiments/v086/preln/nbs_step2000.pt \
  bash scripts/nbs/run_nbs_v086.sh evaluate preln

# 仅在已约定需要图干预报告时，对固定 checkpoint 执行。
CHECKPOINT=outputs/latence_nbs_experiments/v086/preln/nbs_step2000.pt \
  bash scripts/nbs/run_nbs_v086.sh diagnose preln
```

E/M/B/G 仍沿用原 reference 对齐、来源校验和指标定义。已有 v085 输出继续用 `run_nbs_v085.sh recompute`，无需重训；v086 使用自己的 evaluator。不要把两版 prediction manifest 混入同一个导出目录。

条件 C 若决定执行，独立开 fresh `preln_dropedge`，不能从 preln checkpoint 改 config 续训。多种子复验要为配对的两个 config 设置相同新 `full_task.seed`、保持采样 seed 规则一致，并使用独立 `WORK_DIR`；不覆盖第一次结果。

## 7. 若 LN 没有可重复图收益，下一步检查什么

优先做只读训练快照审计，再选择一个明确假设改动。不要连续堆叠 BN、sum、JK、GAT、fanout、decoder dropout 和 PU 系数。

1. **weak 输入来源。** 分开 sampled candidate、sampled binary pseudo、直接 candidate matcher 三条路径的贡献；保持其余分支和 checkpoint 不变。现有 `weak_off` 不能完成这种归因；本包没有把新细分干预伪装成已实现功能。
2. **难 gold 来源。** 报告去重 core 蛋白/GO 对、B 概率分层、频次分层、每种已知正例的 loss 与梯度。当前 Stage1 见过的 core 很容易，考虑未来真实时间外训练 gold 或严格交叉拟合的训练表征/预测；不能把 dev/test gold 倒灌，也不能擅用 weak 区域被禁止的 metadata gold。
3. **PU 冲突。** 区分 decoder 当前图支持 A、固定 loss top-8 的任意支持 F、经过相似度/邻居数/票数规则的结构支持 S。在 unknown 且不属于 alias exclusion 的集合 U 内，分别统计 `A∩¬F` 与 `A∩F∩{S=0}`，再与实际 hard-PU 集合相交。它们说明来源/规则缺口，不直接说明是假阴性。具体快照审计设计见包内 `PU_AUDIT_v086.md`；本轮未修改 loss。

确认上述证据后再选择一次最小实验，例如难 gold 的训练采样策略，或固定 encoder 下的 PU 支持规则；该实验需要单独版本和匹配控制。未确认可重复图特异性收益之前，不迁回原完整模型。

## 8. 需回传的最小材料与验证边界

- dynamic：600/2000 的 comparison，加 resolved_config、training_history；有 manifest 一并提供。
- 新 A/B：resolved_config、training_history、已按协议生成的 comparison，加训练启动命令、设备/卡数、实际 stop step、失败或恢复记录。600 的独立测试 comparison 仅在事先登记需要固定轨迹报告时生成；pilot 不自动评估测试。dev 报告如有，提供 manifest 的审计信息与固定选择规则。
- 图干预：固定 checkpoint 的各 comparison 或汇总；保留原预测矩阵在服务器，以便需要时做配对统计。
- 新的 dev/难 gold 来源：Stage1/Stage2 是否见过、时间截点、去重/同源组规则、core/weak 数量、已知正例数量及 GO 频次；无需上传全部大矩阵来完成下一次代码审查。

本地验证范围和实际通过项记录在包内 `VALIDATION_v086.md`。CPU 小图测试只能验证实现契约和恢复/评估逻辑，不能替代服务器 GPU、DDP、全规模 BP 的 smoke，更不能证明真实性能提升。
