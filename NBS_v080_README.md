# LATENCE NBS v0.8.0：目标蛋白、完整 GO 监督、共享图前向

本包基于本次上传的 `latence_nbs_v071_related_code.zip` 开发。实际改变训练主路径，同时保留预测 weak–GO 和 weak–core–GO 两条证据路径。它不是旧 checkpoint 的修补器，需要从 Stage1 产物初始化新的第二阶段模型。

**GO 映射修复已纳入本包**

已修正初始化时报 `task GO mapping must be one-to-one` 的错误检查。原项目允许 canonical GO ID 与 alt-ID 对应的多个 classifier 列共享一个 ontology 节点；模型现在接受这种多对一映射，仍检查一维、非空和索引范围，完整保留任务列顺序与数量。不要对任务输出列去重。该修复不改变模型参数结构或数据签名，现有 core 邻居缓存可复用。若上次运行在此初始化检查处退出，覆盖修复文件后直接重新运行 `pilot`，不需要 `RESUME`。

本次模型、训练/恢复/导出和原 GO 对齐规则的针对性测试：18 项通过，1 项 DDP 测试仍因本地 Gloo socket 限制跳过。新增用例检查共享几何时预测、候选证据、core vote 和 loss 仍按各任务列分别处理；实际导出测试也使用含重复映射的完整 CC 输出空间。

**安装与首次运行**

把 `latence_nbs_v080.zip` 放在 `latence-project` 的同一级目录，然后运行：

```bash
cd /home/dataset-local/data_local/shaojiangyi/latence-project
unzip -o ../latence_nbs_v080.zip
bash scripts/nbs/run_nbs_v080.sh pilot
```

压缩包内部直接是项目相对路径。解压位置必须是 `latence-project`；不需要另建补丁目录，也没有 `Base files differ` 安装步骤。交付文件全部是新增路径，不覆盖上传源码中的旧训练器、matcher、评估器。

`pilot` 自动完成一次 core 邻居准备，然后默认用两卡训练 600 次同步参数更新，保存第 100、300、600 步。它们是更新步数，不是三个完整数据 epoch。默认每卡 64 个 weak 和 16 个 core 目标，两个 rank 使用同一个全局乱序队列的不同片段。按实际完成的 weak 访问数记录 `weak_equivalent_passes`。

首次准备用 Torch GPU 分块计算归一化 cosine 的 core-only top-8，排除固定留出 core 和 core 目标自身。之后复用该缓存，不重跑 Stage1、不重建全量 PPI 图。首次准备本身也有计算成本；当前环境没有真实数据和 A100，不能给出已验证的运行分钟数或 GPU 峰值。训练每 25 步报告实际 sec/step 与显存。

若需要改路径或运行预算，主要修改以下两个文件：

| 文件 | 应修改的内容 |
|---|---|
| `scripts/nbs/run_nbs_v080.sh` | `CONFIG`、`WORK_DIR`、`NUM_GPUS`、`STEPS`、`INPUT_DIR`、`METADATA_FILE` |
| `nbs_models/nbs_protein_go/configs/bp_full_task_v0.8.0.json` | `data`、`go_boxsqel` 数据路径，以及 `full_task` 中的 batch、模型和 loss |

默认数据路径沿用本次源码的 full-task top-512 产物；默认独立测试输入目录为 `outputs/latence_nbs_eval/bp_shared_full_task_top512_v071_v2`。若您的实际正确缓存是其他目录，只改 `INPUT_DIR`。不会因为缓存缺失而静默构建 rare-first 输入。

**本次确认并处理的结构问题**

| 当前上传源码中的限制 | 新路径的实际改动 |
|---|---|
| `experiments/nbs/latence_nbs_components.py` 新建模型后调用 `freeze_go_geometry`；`nbs_pg/training.py` 的该函数冻结整个 `go_box_encoder` 和 `go_tower`，包括新初始化的神经网络参数 | 只固定原始 BoxSquaredEL center/offset；GO 投影和 ontology 关系参数参与训练，每步重算学习后的 GO embedding |
| `nbs_pg/model.py` 的外部蛋白前向主要使用 core 原始特征；其他关系 source 被置零，core 自身的 gold–GO 消息没有按训练路径进入外部 target | 训练和独立测试调用同一个 target-local graph forward；core 状态由蛋白特征和其 gold–GO 消息共同生成，再传给 target |
| 外部 weak–GO 预测证据主要进入 matcher 的标量 gate | 预测 GO embedding 按 Stage1 边属性聚合到 target，并保留具体 protein–GO 配对的候选属性；每个 GO query 决定如何使用这些信息 |
| query episode 先选小批 GO，再填充大量蛋白；单个 primary weak 的正监督受 query bank 限制 | 先选目标蛋白，输出全部 21,312 个 BP GO；每个目标的全部已有 gold/pseudo 正例进入 loss，不受 K=512 或旧 Q=64/144 限制 |
| 正负项、采样组成和 ASL 归一化使总 loss 的绝对值不易解释 | 正例、hard PU、background PU 分别按蛋白归一化；同批、同一 PU mask 计算 backbone objective 和 objective gain |
| 小残差在混合精度、多个小尺度门控下可能难以影响最终输出 | 默认训练使用 FP32；最终残差与 base 在 FP32 相加，只保留一个零初始化输出层 |

这些是源码确认的计算或优化限制；不能仅凭源码给出各项对原性能下降的因果贡献比例。旧三轮训练 loss 约下降 10%，因此不能说完全没有优化；问题是优化未转化为独立任务收益。新 loss 数值变大本身也不构成改善证据。

**新的图与 decoder**

所有 ontology GO 共享 BoxSquaredEL 输入和四种有向关系：is_a、has_child、part_of、has_part。稀疏消息传播训练 GO 表示。每个目标蛋白获得三个来源：自身 Stage1 表示、预测 weak–GO 邻域消息、core 邻居消息；core 消息包含该 core 的 GO 注释表示。

decoder 对每个 protein–GO 配对，使用该 GO 的 query 与三个来源匹配，再结合该配对的候选属性和 core 支持分数生成有正有负的 logit 修正。不存在只覆盖候选 GO 的输出空间，也没有额外的可训练逐类别分类器权重矩阵。候选外的已知正例同样参与 query/graph 参数优化。

这是目标蛋白局部的异构消息传播模型。当前精简主路径没有引入旧 PPI/core–core 多跳采样，也没有跨 target 的批内消息回流；不能称为“全部旧关系原样保留”。减少这些计算使两条核心路径可以在训练与测试使用一致定义。是否还需要 PPI 扩展，应在这条路径取得可靠收益后作为一个独立问题判断。

原始 GO 几何和静态稀疏邻接可缓存；学习后的 GO embedding 只允许在评估模式、同一 checkpoint 内缓存。`go_chunk` 只控制解码分块，不减少输出或正例监督；反向传播仍保留各块必要的 activation，不能声称训练总显存仅取决于 chunk。

**正例、PU 与标签隔离**

- core 正例取训练集 gold CSR；weak 正例取已审计的 modelout pseudo CSR，并保留 soft probability。原始 `>0.5` 筛选后存成 FP16 的值可能恰好等于 0.5，仍按 CSR 成员关系保留，避免再次丢标签。
- 所有已知正 GO 都进入正例 BCE，按蛋白内 confidence 归一化；weak/core 分别取均值，再按角色权重合成。
- hard PU 从完整输出中按 `max(current_probability, base_probability)` 选 top-64；background 从其余 unknown 中无放回抽取 64 个。两组互斥，均排除全部已知正例。未标注仍是 PU，并非确认的生物学负例。
- 默认 hard/background 权重为 0.2/0.05，negative ASL 使用 gamma=2、clip=0.05；其余 unknown 使用 baseline KL anchor。新正监督仍可能先促使分数整体上升，不能把训练 gain 当作已解决这一风险的证据。
- target 的 gold/pseudo 字段只供 loss 使用。weak pseudo 不进入图前向；core 目标自身不会作为自己的 annotation-bearing anchor。
- 固定 1,024 个 core 留作第二阶段验证，从训练目标和所有 anchor 中同时排除。训练/测试均以剩余 core 按同样 cosine 检索。独立测试的 1,800 行邻居在首次使用时按这一池重新计算一次，继续复用原 Stage1 表示、base 概率和候选数组。

这份 core 留出集只隔离第二阶段；Stage1 已经见过原 core 训练蛋白，因此不能当作最终独立泛化证据。它用于低成本筛选 checkpoint 和判断是否继续投入，最终主结果仍看独立测试。

**600 步之后怎么判断**

训练目录默认：`outputs/latence_nbs_experiments/v080/main`。

| 产物 | 用途 |
|---|---|
| `training_history.json` | 每 25 步的 loss、同批 base objective/gain、正/PU 数量、实际 logit 修正、耗时和显存 |
| `validation_history.json` | 第 0、100、300、600 步的完整任务 histogram micro-AUPRC、Fmax 与 P/R@10/50/100；最后一步包含 weak_off、core_off |
| `nbs_step100.pt`、`nbs_step300.pt`、`nbs_step600.pt` | 固定预算 checkpoint |
| `best.pt` | 留出集 micro-AUPRC 最好的已训练 checkpoint；它可能仍不如 backbone |
| `latest.pt`、`resolved_config.json` | 续训与实际运行配置；runtime 记录总更新预算和全局 batch |

按以下顺序作决定：

1. 看 100→300→600 步的留出 AUPRC 和 P/R@k，而不是要求随机 batch loss 单调下降。
2. 若训练 gain 增加，但留出 AUPRC/Top-k 不改善，不原样延长到完整多轮。现有结果已把问题缩小到目标函数/图证据的泛化，而不是继续猜 query 是否覆盖正例。
3. 若完整模型不如 weak_off，检查预测 weak–GO 信息带来的噪声或依赖；若不如 core_off，检查 core 检索和注释传播。两个开关均关闭该路径的“消息＋配对证据”，因此衡量整条路径作用，不能单独证明 GNN 聚合胜过简单标签投票。
4. 只有留出 AUPRC 有稳定收益、Top-k 没有明显倒退，再以 `best.pt` 做完整独立测试；通过后才扩大训练量。独立测试不用于每 100 步反复调权重。

运行最终独立评估：

```bash
bash scripts/nbs/run_nbs_v080.sh evaluate
```

默认复用原 Stage1 评估器，固定 P/R@10/50/100，输出到 `main/eval_step<实际步数>/full/metrics/`，包括 `nbs_ind_test_metrics.json` 与 `nbs_primary_comparison.tsv`。预测为 FP32，避免存储量化把小修正再次抹掉。若预测已完成、只在 metrics 阶段失败，修复评估器所需环境后重跑同一命令，会校验输入/预测身份并复用已有预测。

如确需检查另一条路径，无需重训：

```bash
ABLATION=weak_off bash scripts/nbs/run_nbs_v080.sh evaluate
ABLATION=core_off bash scripts/nbs/run_nbs_v080.sh evaluate
```

这两条并非首次必跑；600 步的 core 留出结果已包含它们。

确认值得延长后，可以保持配置和 GPU 数量不变续训：

```bash
STEPS=2000 RESUME=outputs/latence_nbs_experiments/v080/main/latest.pt \
  bash scripts/nbs/run_nbs_v080.sh train
```

`STEPS` 是累计总步数。续训保存模型、优化器、各 rank 的 RNG 和蛋白队列位置，使用 warmup 后恒定学习率，避免改变总步数时重建 OneCycle。续训须在原 work-dir，模型、loss、数据、batch 和优化器设置保持一致；修改这些设置应创建新的训练目录。

**验证范围**

CPU PyTorch 2.4.1 下，23 项测试通过。覆盖：全部正例与非候选正例监督；21,312 GO 下有效梯度不被全空间分母稀释；两条图路径和 GO 参数的学习；不读取 target labels；core 自标签与留出隔离；batch/chunk 一致性；BF16 环境下 FP32 残差；连续训练与断点恢复一致性；真实导出到现有评估脚本的 subprocess，以及 metrics-only 重试。

一个较强的 synthetic 测试令真实 GO 同时不在 candidate 和 core vote 中，仅通过不同 weak–GO 图上下文区分目标；训练后能恢复该小集合的全部 top-1。它证明代码具备所需表达和优化能力，不代表真实蛋白独立测试已经改善。

两 rank CPU DDP 测试已提供，但本执行环境禁止 Gloo TCP socket，故该项跳过。没有运行真实 A100/CUDA DDP、生产数据训练或 Stage1 正式指标后端；实际导出集成测试使用现有评估器的 `local_micro` 后端。真实环境默认仍使用 `stage1`，不会静默替换正式口径。

如本地需要复核代码测试：

```bash
PYTHONPATH="$PWD/nbs_models/nbs_protein_go:$PWD/nbs_models/nbs_protein_go/tests${PYTHONPATH:+:$PYTHONPATH}" \
  OMP_NUM_THREADS=1 python -m pytest -q \
  nbs_models/nbs_protein_go/tests/test_full_task_loss.py \
  nbs_models/nbs_protein_go/tests/test_full_task_model.py \
  nbs_models/nbs_protein_go/tests/test_full_task_data_v080.py \
  nbs_models/nbs_protein_go/tests/test_full_task_runner.py
```

实际新增的运行源码为三个模块 `full_task_model.py`、`full_task_data.py`、`full_task_loss.py`，一个 Python 入口、一个 shell 入口和一个配置。继续复用已有 mmap 产物、索引格式与正式评估器。配置中的 `episode` 是旧 store 构造器兼容字段，其 sampler 不参与新训练；实际 GO 数始终为完整 task vocabulary，不要通过修改该字段调整 v0.8.0 的 Q。

当前交付建立了可以检验 weak-to-strong 主张的训练路径；期刊级结论还必须来自真实独立数据上的显著收益、两条路径的机制证据和受控比较，不能由 loss 数值、synthetic 成功或一次架构修改替代。
