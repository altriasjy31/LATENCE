# LATENCE NBS v0.9.0：共享 protein–GO scorer

本版本保留异质 tunedGraphSAGE，将固定分类器替换为使用目标 GO 表示的共享打分器，加入监督隔离及分项 loss 日志，并将固定闭集评估合并为一条命令。所有 v088/v089 原文件保持字节一致。发布/模型版本为 0.9.0，数据和采样架构仍为 0.8.8；底层 Python 包元数据保留原值，不要求重新安装旧包。

**验证边界：CPU 小图测试验证代码正确性；新模型的外部 Fmax、A100 显存、速度和真实新 GO 功能预测均未验证。** 当前已测 v089 epoch30 Fmax=57.050449995685725，固定 baseline=57.72314936447243，完整门槛=60.72314936447243。v090 的对应分数均为空。

## 1. 主改造及理由

原输出为 `Linear(320, 21312)`，各 GO 分类器行独立训练。GO 几何/本体只作为稀疏证据参与蛋白表示，目标 GO 本身未直接参与最终打分。

v090 采用 `score(p,g)=(pW)g/sqrt(H)+b(g)`；`b(g)` 是共享的 GO-conditioned 线性函数，包含共享标量 bias，没有逐 GO 独立参数行或频次 bias。新目标只要有合法、坐标一致的 GO 表示，就可以通过同一组参数打分。默认仍按原 task_to_ontology 顺序输出完整 21312 列，包括原始 alias 列；映射到同一 canonical GO 的列同分。

主配置维持 H320、3 层 typed GraphSAGE、2 层 ontology encoder、RAW residual→typed BN→ReLU→dropout、input dropout0.1、层 dropout0.25、原三层 fanout 和监督权重。JK 使用 last，edge_dropout=0。可选 JK sum/训练期 DropEdge 的实现有正确性测试，但本次正式配置未启用，不把它们当成官方 GraphSAGE 的必要缺项。

在现行 feature_dim2048、GO geometry512维、4类本体关系下，参数数从11,423,680变为4,685,249；输出头从6,841,152变为102,721。参数减少不等于训练时间或显存按比例减少，实际资源须由新日志测量。

## 2. tunedGNN 官方证据

已通过 GitHub 插件固定官方 main 到 commit `23f9604e8b13a9a6d3faa2f691cd844006979153`。来源链接和参数清单见 `UPSTREAM_TUNEDGNN_v090.json`。

| 项目 | 论文 GraphSAGE proteins，Table9 | 官方 proteins.sh＋protein.py 实际命令 | v090 主配置 |
|---|---|---|---|
| 层数 | 6 | 6 | 3，控制改动与当前采样成本 |
| hidden | 512 | n_hidden80×n_heads4=320 | 320 |
| normalization | BN | BN | 按节点类型 BN |
| residual | 开启 | 前一 RAW 层输出残差 | 保留同一骨架 |
| 层 dropout | 0.3 | 0.25 | 0.25 |
| input dropout | 表中未单列 | 0.1 | 0.1 |
| JK | 正式附录已移除该搜索项 | shell 未开启 | last，即未启用聚合 |
| SAGE DropEdge | 非必要配置 | SAGE 分支不使用；edge_drop只作用于GAT | 0 |
| optimizer | 具体实现见代码 | AdamW，LR0.01，WD0 | AdamW，LR0.006，WD1e-4 |
| schedule | 表中未单列 | 验证ROC-AUC驱动ReduceLROnPlateau(.75,50) | 原定warmup2/hold10/cosine100，minLR0.0006 |
| epochs | 1000 | 1000 | 100 |
| train batch | 表中未单列 | ceil(Ntrain/10) | 每卡5120weak＋1280core |
| eval batch | 表中未单列 | 65536 | 16 |
| sampling | 表中未单列 | 训练fanout32，评估100，每层相同 | 保留当前异质关系fanout |

65536不是官方训练batch；官方proteins的112标签ROC-AUC任务也不能直接等同本任务21312标签protein-Fmax。论文的主要启发是认真保持normalization、dropout、residual与合适深度，并使用独立验证集调参。当前模型已有前三项；不能据平台曲线就断言缺JK或DropEdge。论文附录B.3明确充分调参后JK往往不必要。异质图不自动等于异配图，不能据“heterogeneous”就套用异配图加深结论。

本次共享scorer是针对LATENCE的标签语义和新GO需求的改造，**不是该论文已经验证的性能结论**。为判断其效果，主版本保留原编码器、loss角色权重及日程，不同时展开深度/损失/正则参数网格。

## 3. 安装：只增加新文件

将压缩包解压到现行项目之外，保留正在运行的 v089：

```bash
unzip latence_nbs_v090.zip -d /tmp
python3 /tmp/latence_nbs_v090/install_nbs_v090.py \
  --project-root /home/dataset-local/data_local/shaojiangyi/latence-project --dry-run
python3 /tmp/latence_nbs_v090/install_nbs_v090.py \
  --project-root /home/dataset-local/data_local/shaojiangyi/latence-project
```

安装器完整预检后仅写新文件/缺失依赖；已有相同文件不重写，未知不同字节拒绝。不要将压缩包直接覆盖现行目录。复用服务器PyTorch2.4环境、既有数据路径和缓存；新增默认闭集配置不要求2025 ontology或新标签资源。

## 4. 主实验启动

在可用GPU上使用独立WORK_DIR。默认fresh主实验用于控制变量，NBS参数从0初始化，冻结序列特征和GO资源继续复用。

```bash
cd /home/dataset-local/data_local/shaojiangyi/latence-project
export PYTHON_BIN=/opt/conda/envs/pytorch2.4/bin/python
export CUDA_VISIBLE_DEVICES=0,1 CUDA_DEVICE_ORDER=PCI_BUS_ID
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export CONFIG=nbs_models/nbs_protein_go/configs/bp_full_task_v0.9.0_mb12800_ddp_ep100.json
export NUM_GPUS=2 DEVICE=cuda:0
export WORK_DIR=outputs/latence_nbs_experiments/v090/mb12800_shared_go_fresh_ep100
unset INIT_FROM ENCODER_FROM RESUME STEPS STOP_STEP STOP_EPOCH CHECKPOINT DEV_MANIFEST PRESET
unset ABLATION METRIC_BACKEND AUPRC_MODE
bash scripts/nbs/run_nbs_v090.sh train ddp
```

每5轮保存固定checkpoint，默认48updates/轮，100轮4800updates；weak尾批按实际总体缩短。若先做2步服务器入口检查，可将最后一行改成 `SMOKE_STEPS=2 bash scripts/nbs/run_nbs_v090.sh smoke ddp`，之后同目录用RESUME继续，不重新fresh。

严格续训：

```bash
unset INIT_FROM ENCODER_FROM STOP_STEP STOP_EPOCH
RESUME="$WORK_DIR/latest.pt" bash scripts/nbs/run_nbs_v090.sh train ddp
```

可选的编码器迁移只作为另一种明确初始化，不与fresh混记：新目录启动前设置 `ENCODER_FROM=outputs/latence_nbs_experiments/v089/mb12800_fresh_ep100/nbs_epoch30.pt`。它验证旧来源后仅导入encoder及BN，重新初始化共享头/optimizer/RNG；新scorer epoch和曝光从0计数，旧encoder曝光单独记录。当前不接受将使用过unseen GO监督的旧编码器迁入label-holdout实验。`INIT_FROM`不用于v090。

单卡配置为 `bp_full_task_v0.9.0_mb6400_single_ep100.json`，`NUM_GPUS=1`，命令preset=`single`，保持每卡batch，总100轮9600updates。

## 5. 简化评估：一条命令

保留原固定闭集1800×21312，strict >、0.001阈值、不删零正例列、不做ontology传播，不融合B/E/M。只汇报Fmax、P/R、阈值、与baseline差及达标状态。

```bash
unset INIT_FROM ENCODER_FROM RESUME STEPS STOP_STEP STOP_EPOCH
CHECKPOINT="$WORK_DIR/nbs_epoch20.pt" bash scripts/nbs/run_nbs_v090.sh evaluate ddp
```

该命令使用单进程推理，自动导出固定checkpoint、计算原Fmax-only、写 `acceptance_epoch20.json`。fresh epoch20对应 `eval_step960/full/metrics_fmax_only/nbs_w2s_comparison.json`；epoch25/30为step1200/1440。无需手动拼接eval目录或再运行checker。重复执行会先核对缓存来源；来源变化或不完整缓存拒绝复用。

独立checker退出码仍为0=达标、1=未达标、2=未验证；整条evaluate在合法报告为“未达标”时正常完成，来源/计算错误仍报错。超过baseline与完整达标分别记录；完整门槛始终60.72314936447243%。固定epoch100为主终点，中间结果用于描述曲线，不用ind_test选择超参数或挑最佳checkpoint替代终点。

## 6. 当前v089平台：可选的一个缓存命令

```bash
"$PYTHON_BIN" scripts/nbs/compare_nbs_cached_v090.py \
  --work-dir outputs/latence_nbs_experiments/v089/mb12800_fresh_ep100 --epochs 25,30
```

只使用已导出的矩阵，不重跑模型。输出B与两个epoch的Fmax、P/R、阈值，以及相同recall45/50/55%下的网格precision上界。其定义为原0.001阈值网格中recall至少达到目标的最大precision，不插值，达不到则null。结果另存小JSON；不能据单个点断言过拟合或因果瓶颈，不计算AP/传播/融合。

## 7. 新GO接口与隔离

模型提供 `encode_proteins(batch)`、`encode_ontology()` 和 `target_ontology_ids`；支持动态目标顺序、重复canonical/alias列及GO分块推理，无需新增classifier参数行。GO字符串通过hash绑定的registry解析到**现有合法geometry行**。示例：

```bash
"$PYTHON_BIN" scripts/nbs/predict_nbs_go_v090.py \
  --checkpoint "$WORK_DIR/nbs_epoch20.pt" \
  --input-dir outputs/latence_nbs_eval/bp_shared_full_task_top512_v071_v2 \
  --target-go-file target_go_ids.txt --output-dir outputs/v090_target_go_ep20 --device cuda:0
```

`target_go_ids.txt`每行一个GO ID。输出单独的概率矩阵、目标列映射及来源记录，不读取gold、不自动声称zero-shot有效。未知GO/冲突alias/obsolete歧义拒绝；不随机补geometry，不允许把重新独立拟合的2025坐标直接替换旧ontology资源。目标namespace及真实新版本实验仍需其正式ontology快照契约；此接口首先支持现有ontology中有几何的标签。

默认 `label_split=all_seen` 为原闭集。标签留出可在**新实验开始前**给 `full_task.label_split` 指定manifest；seen/unseen按canonical alias整组划分，同时过滤loss列与负例、candidate/gold/pseudo证据和targets，没有逐GO频次prior。留出实验保留原列顺序，但必须用独立评测协议，闭集baseline gate会拒绝混用。

Stage2留出应标为 `stage2_label_holdout`；上游特征和teacher可能已见过目标GO。严格pipeline模式要求实际文件hash、时间、完整监督GO范围及上游依赖声明；缺失或存在目标监督污染则拒绝。它核对hash绑定声明，不能代替独立重训审计。当前现行数据不自动满足严格模式。

2022不存在、2025新增的GO仍需用户提供合法新节点及同坐标系geometry/跨版本映射、时点gold和独立协议。当前交付不包含这些数据，也没有新GO生物功能预测的实测结果。

## 8. 新日志及测试

`training_history.json`新增 `contrib_core_positive`、`contrib_core_negative`、`contrib_weak_positive`、`contrib_weak_negative`，加总应与总loss一致。按角色均值归一化，当前core/weak权重1/.25对应总目标约80%/20%，不是按batch数量计算监督占比。分项只用于诊断，不自动修改loss或学习率。

完整测试命令与实际结果见 `VALIDATION_NBS_v090.json`。测试包括非前缀block映射、动态GO/alias/顺序/chunk一致性、合法新增geometry节点、seen负梯度隔离、污染拒绝、旧encoder迁移、精确resume、真实mmap loader和shipped config入口、缓存复用及验收契约。测试只证明实现正确性。已实际尝试双进程Gloo；当前执行环境禁止创建TCP device，该项记录跳过，NCCL/A100仍需服务器验证。
