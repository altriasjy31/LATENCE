# LATENCE NBS v0.8.3：原生图上下文与全集单matching

本版以v081为基础，撤回v082的dense modelout蒸馏与classification/dual结构。目标仍是：图模型G在主Fmax、主AUPRC上超过expert E，同时明确其增益是否来自图信息。

这是现有LATENCE项目的代码更新包，包含模型、数据、训练/评估入口、配置、测试及安装器；不含数据和checkpoint。

## 1. 本次结果与问题定位

核对10份v082附件后，E/M/B、输入manifest和主评估helper一致。当前没有发现旧缓存混用。

| 输出 | steps | 主Fmax | 主AUPRC | exact micro-AP |
|---|---:|---:|---:|---:|
| B | — | 55.6317 | 32.0087 | 49.4508 |
| v081 graph | 4000 | 55.8969 | 36.9541 | 50.9435 |
| v081 graph | 8000 | 55.8189 | 36.9255 | 51.4998 |
| v082 teacher | 600 | 55.3704 | 35.7840 | 49.9277 |
| v082 classifier | 600 | 55.4102 | 35.0105 | 50.3158 |
| v082 dual | 600 | 55.4991 | 35.9123 | 50.5416 |
| v082 dual | 4000 | 55.3128 | 37.2909 | 51.0199 |
| E | — | 69.7811 | 56.4960 | 67.5547 |
| M | — | 69.7716 | 59.2118 | 68.7907 |

单位为百分数。dual4000相对v0814000：Fmax −0.5841个百分点，主AUPRC +0.3369；相对E仍差14.4684个Fmax百分点及19.2051个AUPRC百分点。双指标目标未实现，但不能称所有指标全面退步。未提供的classifier4000结果不作数值引用。

dual训练501–600与3501–4000窗口：总loss 0.09421→0.07773，最终G的全域可比目标0.06242→0.04975，teacher KL 0.10130→0.08329。说明优化确实发生，但没有转化为希望的精召率平衡。旧selected集合已覆盖约98.52%的训练高teacher位置，不能继续把这次问题直接解释成“正GO完全没采到”。

关于“modelout捷径”：v082完整M只用于loss，没有直接加到推理输出或输入encoder。当前证据支持撤回以拟合M为中心的目标，不能宣称已经证实M前向泄漏。v083进一步明确：原M>0.5 CSR仅提供离散成员关系，选中的目标为1；M概率不做软target或置信度权重，也不读取dense modelout文件。E/M仍保留为独立评估参考。

## 2. v081与原完整图的实际差距

v081读取真实训练蛋白、冻结表征、core gold、weak pseudo及完整GO本体；训练与归纳推理也调用同一forward。差距主要在如何使用这些数据：

- 原PPI、similar_to、weak_to_core虽被loader打开，v081的FullTaskData没有使用它们。
- 目标蛋白只连接重新按余弦相似度检索的8个core，以及自身512个GO候选。
- 每个core的全部GO先平均为一个128维向量；逐GO query只能关注这份混合摘要。精确GO关联主要留在最后的标量core_vote中。
- graph_aux输入包含own-protein分支，辅助loss下降可以来自蛋白自身信息，并不单独证明图有效。

因此，“相对于原完整异构图，当前模型省略了重要结构”有源码依据；“用了测试生成的训练数据”“训练与测试构图代码已经错位”没有当前证据支持。

现有full模型结果也不足以证明图贡献为零。v083增加直接检验这一问题的对照，而不是用总loss下降代替图贡献。

## 3. v083实际实现

### 3.1 原生P–P上下文进入core锚点

目标仍用相同方式检索8个core，保留训练/归纳的一致性。每个core锚点从原图读取三类真实入边：

- ppi
- similar_to
- weak_to_core

始终按edge_index的source→destination方向读取，不把CSR索引方向当作边方向，也不擅自反转边。原PPI若已双向导出，两个方向均按原始边处理。

默认每个core、每种关系最多4个邻居，按confidence、原score、reciprocal rank排序并去重。原始图只在prepare时分块扫描一次，生成小型缓存；不构建全体蛋白的巨大新邻接矩阵。

上下文节点输入为冻结蛋白表征和最多16个backbone候选GO。三种关系分别编码，再更新core锚点。上下文节点的gold/pseudo标签不进入这一步，从而避免core目标标签经两跳回流。监督core目标自己的gold边仍被排除；holdout core不进入锚点及PP上下文池。

训练和外部目标使用同一份core图上下文。无需为ind_test凭空生成真实PPI边。

### 3.2 每个GO query读取精确core–GO关系

保留原有一般core上下文，同时增加精确关联消息：对当前GO，仅从实际标注该GO的core锚点汇聚相应信息，并保留支持质量/权重。GO信息不再只有标签均值摘要和一个标量vote。

仍对全部21,312列进行单matching解码，没有classification塔，也没有256个query的输出限制。支持矩阵按GO chunk读取，避免构造巨大[B,GO,neighbor,hidden]张量。原GO别名列保持独立输出位置。

### 3.3 最终预测直接监督图通道

保留B加残差的初始化、全部已知正例、hard/background/medium PU、ranking及结构支持衰减。weak正例统一二值。旧graph_aux关闭，图参数通过最终预测目标训练。

v083仍保留own-protein和B上下文，这是必要的可学习参照。graph_off不会被人为强制成B；否则“关图下降”只是结构定义，不能成为有效证据。

原生图贡献仍需实测。这是有预算的图上下文接入，尚未恢复原NBS所有层、蛋白→GO反馈和整图端到端更新，不能称作完整异构图训练。

## 4. 安装

将ZIP解压在latence-project同级，例如：

- /path/to/latence-project
- /path/to/latence_nbs_v083

在项目根目录运行：

~~~bash
python ../latence_nbs_v083/install_v083.py --project-root .
~~~

也可从其他位置指定两个绝对路径。安装器核验包内文件并先备份，备份目录为outputs/nbs_code_backups/v083_时间戳；重复安装同一版本无副作用。

对原local_loader.py只应用一个局部补丁，允许明确标记为v083的binary_membership消费模式；其余旧版仍要求原有soft-target契约。安装器保留该文件的其他本地修改，不覆盖整份loader，也不要求它与旧版本整体哈希一致。若这个具体校验片段已被手工改写，安装器会在写入前明确指出位置。

不修改v081/v082实验结果。v083默认使用新的outputs/latence_nbs_experiments/v083目录。由于结构与目标发生变化，不能用v081/v082 checkpoint作RESUME；同一v083实验可以正常续训。

## 5. 先准备真实图，再做短训练

~~~bash
bash scripts/nbs/run_nbs_v083.sh prepare graph
~~~

prepare复用已有cosine core邻居缓存，并新建原生PP上下文缓存。默认位置：
outputs/latence_nbs_experiments/v083/shared_pp_context

实际路径以配置中的full_task.pp_context_cache为准。终端及pp_context_manifest.json给出每种关系的源边数、保留边数、覆盖core数和center_coverage。首先确认这些数字：若某类关系覆盖为0，它就没有提供相应消息，不能仅凭文件存在认为已经用上。

首次扫描原图是一次性成本，耗时取决于源边规模和存储速度；后续smoke/train自动核验并复用缓存。PP源文件记录路径、大小、mtime、shape，索引manifest及缓存数组记录哈希。未额外全量哈希所有庞大PP源数组。

三份配置：
nbs_models/nbs_protein_go/configs/bp_full_task_v0.8.3_{graph,local,no_graph}.json

| 训练变体 | 可用图输入 | 作用 |
|---|---|---|
| graph | weak→GO、core→GO、core原生PP上下文 | 主实验 |
| local | weak→GO、core→GO；关闭新增PP上下文 | 隔离原生PP上下文增量 |
| no_graph | 蛋白自身、GO编码、B；关闭weak/core邻域 | 验证图相对相同训练目标是否增加收益 |

三组使用同一模型参数结构、二值监督和损失配置。no_graph关闭的是蛋白邻域及候选GO输入；三组仍共享GO本体编码/传播，且采用同一结构支持衰减策略计算loss，避免顺带更换目标函数。它不等同于原始B，也不代表移除了全部GO本体先验。

先固定600步：

~~~bash
for variant in no_graph local graph; do
  bash scripts/nbs/run_nbs_v083.sh smoke "$variant"
  STEPS=600 bash scripts/nbs/run_nbs_v083.sh pilot "$variant"
  bash scripts/nbs/run_nbs_v083.sh evaluate "$variant"
done
~~~

正式pilot与smoke默认目录隔离；同一输出目录已有结果时，用RESUME续训或使用新WORK_DIR，不会静默覆盖。

默认2张GPU，单卡用NUM_GPUS=1 DEVICE=cuda:0。正式三组应保持相同GPU数、batch、seed及step数。pilot/train不指定STEPS时默认为4000；固定保存600/2000/4000、每1000步及最终step。

不建议先把一组延长到8000再补控制组。先比较同预算的图增量，确认方向后按计划到2000/4000。600步本身不是模型能力上限。

## 6. 一条命令检查图贡献

对已完成的graph checkpoint：

~~~bash
bash scripts/nbs/run_nbs_v083.sh diagnose graph
~~~

执行同一checkpoint的六种模式：

| 模式 | 改变 |
|---|---|
| full | 全部已实现图路径 |
| weak_off | 关闭目标及PP上下文的候选GO信息 |
| core_off | 关闭core消息及其PP上下文 |
| pp_off | 只关闭新增原生PP上下文 |
| graph_off | 关闭weak/core邻域，保留own/B学习通路 |
| go_shuffle | 固定置换候选GO与core注释的GO关联，输出GO列不变 |

go_shuffle打乱的是功能关联，保留P–P拓扑，不是蛋白边拓扑置换。其结果只支持“依赖GO关联身份”的诊断。

各模式仍使用同一E/M/B和原评估器。E/M自动复用v081参考缓存，仅供评估，不参与v083训练。默认INPUT_DIR、METADATA_FILE沿用上一版，可按真实路径设置：

~~~bash
export INPUT_DIR=/实际路径/bp_shared_full_task_top512_v071_v2
export STAGE1_REFERENCE_DIR="$INPUT_DIR/stage1_references_v081"
export METADATA_FILE=/实际路径/unidata_with_exp_train_pseudo.pkl
~~~

诊断主要结果：
- outputs/latence_nbs_experiments/v083/graph/eval_step600/nbs_graph_effects.json
- 同目录各模式的metrics/nbs_w2s_comparison.json
- 预测manifest绑定checkpoint、输入、实现代码、原图缓存、输出概率及干预模式

nbs_graph_effects.json中的full_minus_intervention为“full减干预后结果”，正值表示该干预使指标降低。源checkpoint或评估口径不一致时拒绝汇总。

需要指定checkpoint：

~~~bash
CHECKPOINT=outputs/latence_nbs_experiments/v083/graph/nbs_step2000.pt \
  bash scripts/nbs/run_nbs_v083.sh diagnose graph
~~~

只看full：
~~~bash
bash scripts/nbs/run_nbs_v083.sh evaluate graph
~~~

同一checkpoint的关闭/置换结果是依赖性证据；训练过的graph vs no_graph，以及graph vs local，更适合回答新增图信息是否提高泛化。两类比较应共同看，不能只用“关图下降”宣布图学习成功。

## 7. 判断顺序与后续动作

先看主Fmax和同口径主AUPRC，再看exact micro-AP及Top-k，不用后一种AP替换主AP。

- graph > local：支持新增原生PP上下文有增量。
- local > no_graph：支持weak/core邻域及精确core–GO读取有增量。
- full优于go_shuffle：模型依赖功能关联身份；还需训练对照区分有益信息与分布扰动。
- 关闭PP几乎不变且PP coverage/norm低：先核对边方向、有效覆盖和消息强度。
- 原图覆盖充分、PP消息非零，却graph≈local：考虑关系质量、过度聚合、有限邻居的代表性，再做fanout/关系控制。
- graph与no_graph相近：不能宣称图先验有效，应定位图监督矛盾或缺乏互补信息。
- 即使图对照有收益，G仍未超过E：只能称阶段性图增益，最终weak-to-strong目标仍未达成。

training_history新增：
- pp_edge_count、pp_message_norm：实际进入当前batch的PP上下文。
- weak_core_prior_positive_coverage、core_core_prior_positive_coverage：精确core标签支持了多少已知训练正位置。
- core_supported_unknown_probability_delta：core支持但未列入目标的位置，相对B的平均概率变化。
- binary_positive_targets：二值监督检查。

这些coverage是相对训练gold/pseudo，不是独立测试真标签召回。模型不会把图支持的未标注位置自动改成正例；这些位置仍可能受到PU压力，需结合诊断判断后续是否调整PU。避免无证据地增加负样本或继续扩大Q。

同一v083实验续到4000：

~~~bash
STEPS=4000 RESUME=outputs/latence_nbs_experiments/v083/graph/latest.pt \
  bash scripts/nbs/run_nbs_v083.sh train graph
~~~

local、no_graph同理。延长STEPS保留optimizer、每rank RNG及蛋白循环；改变模型、loss、原图、split或训练实现会拒绝原地resume。

修改fanout或缓存来源时，使用新配置和新pp_context_cache、WORK_ROOT。默认fanout4与PP候选16是首轮计算预算，不是已证实最优参数。优先看同预算控制，再决定扩大。

core holdout仍被Stage1见过，只作数值/训练监视。反复用ind_test决定结构会让它逐渐承担开发集角色；最终模型选择应放到Stage1未见过的独立标注开发集，ind_test用于最后确认。当前并未虚构已提供此开发集。

## 8. 验证与回传

本地43项测试通过。包括真实mmap输入、小型原生PP缓存、三份生产配置的二值契约、分块采样一致性、逐query标签边、标签隔离、两条图路径梯度、续训完全一致、真实CC全列评估及诊断汇总来源校验。

特别验证：在所有样本own特征与B完全相同、只有core邻域能区分标签的小任务中，新模型能优化；关图后损失增加。这证明实现能学习图信号，不证明真实BP数据上已有增益。

环境为CPU PyTorch 2.4.1；未验证用户真实BP数据、CUDA/NCCL及两卡性能。完整测试命令：

~~~bash
PYTHONPATH="$PWD:$PWD/nbs_models/nbs_protein_go:$PWD/nbs_models/nbs_protein_go/tests" \
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python -m pytest -q \
  nbs_models/nbs_protein_go/tests/test_full_task_data_v083.py \
  nbs_models/nbs_protein_go/tests/test_full_task_model_v083.py \
  nbs_models/nbs_protein_go/tests/test_full_task_runner_v083.py \
  tests/test_full_task_eval_v083.py tests/test_run_nbs_v083_shell.py
~~~

下一次优先回传：三组600步comparison与training_history、graph的nbs_graph_effects.json、一次pp_context_manifest.json。无需新增一套庞大审计或上传全部预测矩阵。
