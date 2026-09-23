# v086 只读训练目标审计设计（后续协议，未实现执行器）

依据实际 `full_task_loss_v081.py`、`full_task_evidence_v081.py`、`full_task_data_v084.py`、`train_nbs_full_task_v085.py` 和本轮12份JSON。本文定义后续机制诊断，不修改本版loss、runner、采样、目标或checkpoint；下述命令接口尚未实现，不是本包已运行的实验结果。

## 1. 本轮已经能证明什么

- 三组共用相同core监督曝光：64,000个protein occurrences、3,372,561个gold pair occurrences。B<0.5的gold为5,272（0.156320%）；这些不是去重数。该定义不覆盖B>=0.5但排序仍困难的gold。
- lowlr最后窗口：任意直接core支持均值为 `core_query_supported_fraction * 21312 = 242.653740` GO/蛋白；已知正例均值28.061249，alias排除0.018750。因此至少214.573741个eligible unknown有任意core支持。
- 固定loss support中S>0的eligible unknown仅3.721500个/蛋白。故至少210.852241个/蛋白有任意core支持且S=0；按曝光汇总的未保护比例下界为98.265631%。aligned对应224.152499个、98.366860%。这里由集合包含关系取保守下界，未假定C和S的重叠。
- 不能据此断言98%的PU为错误监督：C定义宽松，S有置信度、数量和票占比门槛；不知道其与hard64的交集，也不知道真实生物标签。
- 每行loss先做行内归一化，再分角色均衡：当前weak/core权重各0.5。每卡64 weak、16 core使每core行权重是每weak行4倍。原始pair数和loss量均不能直接解释参数梯度贡献。

## 2. 固定审计集及预算

先完成静态标签审计，不运行GNN：对训练允许的59476 core，只读取其gold CSR列及冻结B同位置的logit/probability。按唯一protein、唯一task-column pair报告B分位数、p<0.1/0.5/0.9的预设分层、每protein难gold数量分布。不要按本次测试最优阈值构造困难定义。原store的`base_logit_store.gather_matrix([protein], gold_go)`即可按行稀疏索引缓存，不必导出59476×21312的新矩阵。

另预先固定两份只用于机制审计的训练种子manifest：

1. 随机集：8个batch，每batch64 weak+16 core，ID无重复，排除holdout/development/test。用固定audit seed生成，和checkpoint/结果无关。
2. 困难集：从静态B<0.5的core中按固定seed最多选128个protein，分为8个core-only batch（不足则全取）。这是富集条件分析，不估计总体出现率；必须单独报告。

首轮仅对matched legacy/preln的step600/2000执行相同审计。8+8 batches ×2 checkpoint ×2 encoder=64次前向；无参数反向、无优化器step。若成本再压缩，困难集先只在step2000跑。若复核v085，则只用lowlr/aligned的同名checkpoint和同一manifest，不进行全preset扩展。

两个审计图协议应分别命名：

- `eval_deterministic`：model.eval，固定图选择；最容易跨模型重复，不等于真实训练batch。
- 如必须研究训练dropout：使用固定采样context(step/rank/training=True)及`torch.random.fork_rng`，在独立checkpoint副本上model.train前向，记录sampler/context/RNG。不能把它称为历史训练step的精确重放，因为本次JSON未记录原seed ID和RNG。

主训练不得调用新的随机采样或改变RNG流。读完保存的checkpoint独立运行是最稳妥的低成本方案。预先固定的audit train数据不承担模型选择职能。

## 3. 分清两个支持图和三种缺口

统一符号（均为[B,G]）：

- P：`positive_mask`，weak来自二值pseudo成员，core来自gold；不读取weak隐藏gold。
- X：`alias_pu_exclusions(P, task_to_ontology)`。
- U = ~P & ~X：真正PU eligible未知项。
- C：当前sampled第一跳core中任一个有该GO注释。依据`neighbor_index/anchor_go_edge`构造布尔存在性，不含当前batch监督种子和holdout标签。
- A：`output['core_support_mass'] > 0`；当前实现通常与C相同，但学习到的attention权重浮点下溢可能使A小于C，必须报告C xor A。A不是校准概率，不用固定阈值把其直接变成可信label。
- F0：固定`loss_*` top8中任一合法、finite、similarity>0 core有该GO标签。
- F1：F0再要求similarity>=0.5。
- S：调用原`structural_core_support(loss_support_batch(batch), G, config)`得到连续支持；S>0须至少2个不同core支持且相似度加权票占所有合格core相似度总和>=0.5。重复neighbor和重复annotation不得累加票数。
- a=1-0.75*S：当前PU衰减（floor=.25）；S=0时a=1。

对U、core/weak角色分别报告各集合计数、占U比例、每protein均值，以及以任意core支持项为条件的比例：

| 集合 | 可解释含义 |
|---|---|
| U & C & ~F0 | 当前图支持在固定top8中不存在：来源/截断差异 |
| U & C & F0 & ~F1 | 固定pool也有该标签，但支持邻居未达到similarity阈值 |
| U & C & F1 & (S==0) | 有合格邻居，但数量/票占比未过保护规则 |
| U & C & (S>0) | 两套支持一致且得到一定衰减 |
| U & ~C & (S>0) | loss有保护、当前前向直接core读出无该支持 |

前三项是互斥分解，合起来是U&C&S=0。C不包含二跳间接语义传播、weak candidate或pseudo信息，因此不能宣称覆盖“所有图证据”。

## 4. 与真正loss选择相交，不能从JSON推断hard-PU冲突

原代码：`mining=max(sigmoid(z), sigmoid(base))`（detach）；H是U中top64；ranking hard为U中top16；background是U\\H中由rand_like选64；ranking random从U\\rank_hard中用另一rand_like选16。medium是U排除H/background且mining>.05；anchor也在排除H/background后分high_base、positive_drift、background三层，可能与medium重叠。

必须新增的最小表格，每项分weak/core报告：

- |H & C & S==0| / |H| 和 / |H & C|；同样输出rank_hard/rank_random/medium版本。
- |H & C & S>0|及其中S/a的分位数。
- 上节三个缺口分别与H、ranking、medium相交的计数。
- 各集合的B、G概率分位数，mean(G-B)，以及G-B>0的比例。
- ASL active条件为G>.05；不能把入选H的所有项都算成产生有效ASL梯度。
- 已知正例、alias exclusion与各PU掩码的交集必须为0；计数以分子分母全batch累计后取比，不平均不等分母的batch比值。

复现边界：

1. 仅靠当前12份摘要，任何具体H、background、ranking和medium mask都不能重建。
2. 若已有完整z/base/P/X，则H及rank_hard可用原`_topk_mask`重算；精确tie selection需要相同设备/实现，不能用另一个排序函数悄悄替代。可额外报告边界tie数。
3. background、rank_random及依赖它们的medium/anchor需要进入原loss之前的Torch CPU和相关CUDA RNG state。保存前向之前的RNG不够，因为candidate dropout等会推进RNG。
4. 无历史RNG时，只能给固定audit seed的标准化诊断：在fork_rng里调用原loss/重建掩码，明确不是历史随机mask。
5. 最保险的审计快照保存实际mask压缩位图、mining/config/source hash和种子ID；或保存loss调用前RNG与z/base/P/X/S并验证checksum。不要保存或传播E/M连续矩阵。

## 5. 精确logit压力，而非用weighted loss数值充当梯度

低成本方案不做encoder反向：`z_probe=z.detach().float().requires_grad_(True)`，其余输入detach，调用原`full_task_loss_v081`后`autograd.grad(loss,z_probe)`。

- 这是所定义审计batch/掩码下L对最终logit的精确局部导数。
- 分量导数可在审计器中分别仅保留原配置一个loss coefficient，其他系数置0（role weights、K、support、margin等完全保留）；每次调用还原同一loss前RNG，确保所有分量使用相同选样。原代码的random mask抽样与这些系数无关，但实现前仍应断言mask一致。
- 最后验证原total gradient与各分量gradient之和一致；无需改生产loss。新代码若改变随机调用，应改为直接使用保存掩码，不能依赖旧调用顺序。
- 对U上ASL/ranking导数g>0表示梯度下降倾向压低logit；anchor的g正负皆可，分别报push_down/push_up。不要对正负梯度直接求和造成抵消。
- 输出每分量在`U&C&S==0`、受保护项及无core支持项的sum positive gradient mass、每pair均值、以及占该分量全部U梯度质量的比例；正例侧按core/weak、B<.5/.5-.9/>=.9单独报告向上梯度质量。
- 如果比较两个encoder，各自H会随z改变。先报告“各自真实诊断目标”，如需要纯表示比较再用明确的公共冻结掩码做附加表；附加表不得冒充实际训练loss。

局限：这些是dL/dz，不是dL/dencoder参数；经过decoder/encoder Jacobian后可能方向不同、尺度不同。不能从hard-PU占loss40%推导其占encoder梯度40%。真正参数梯度冲突须对相同参数子集计算各分量梯度向量及cosine，需额外encoder反向，首轮不做。

## 6. 最强但仍低成本的“模型内对抗”探针

仅在上述hard交集明显存在时，对预先固定的2个随机batch、step2000做full/core_off配对前向（eval、同一图、同一GO编码）。记d_core=sigmoid(z_full)-sigmoid(z_core_off)。

报告集合 `U & C & (S==0) & H & (d_core>0)` 的数量和原full loss向下梯度质量。这才直接说明“该模型的core证据在抬高这些位置，而当前目标在局部压低它们”。core_off会同时影响core注入/读出及传播，不能把d_core解释为单边因果效应。此探针仍不能确认未知项是真阳性，不能据此直接把它们加成正例。

## 7. 静态难gold审计的补充

在B<.5稀少时，应读B在全部训练gold上的预设分位数和多阈值分布，防止把calibration当困难程度。条件困难集可额外在每row找top16 unknown，用原ranking_margin=1记录gold与unknown的margin违反比例。这是模型当前监督下的排序困难，unknown不等于生物真负例。

按唯一蛋白/GO频率分层记录难gold是否集中在极少数protein或GO，避免总曝光5272实为小批样本重复。分角色positive loss仍使用原行内/角色归一化，不因本审计修改采样或重权重。

## 8. 最小产物与决策关联

最小产物：audit_manifest.json（checkpoint/source/ID/采样与模式/RNG identity）、pu_source_overlap.json、pu_logit_pressure.json、core_gold_difficulty.json；Markdown只是以上JSON的可读视图。

- 若主要是C无F0：后续可做“固定更广PU支持来源”的单因素实验；首先保留loss公式。
- 若主要有F1但S=0：说明保护规则更关键，不能先归咎top8；先量化真实支持强度，勿直接取消门槛。
- 若H与未保护C重叠极少：来源覆盖大差异未必构成主要loss冲突，优先分析weak融合/decoder表达。
- 若难gold梯度质量极低：强化encoder未必能解决纠错学习信号不足；下一项应是来源合规的困难gold/开发集设计，而不是再加深GNN。
- 所有结论用于机制解释和预先设计，不能以ind_test比较挑超参数。
