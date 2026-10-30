# v092：单卡从头训练，原始GO几何直连与随机特征对照

本包替代当前冻结v089→共享头500步→joint的改造方案，重新初始化**全部Stage2可训练参数**并联合训练。Stage1表示、伪标签、candidate和邻居池继续使用现有数据。不存在旧checkpoint热启动，也不宣称Stage1的信息被清除。

## 如何理解本轮判断

v089 epoch80：Fmax57.813881，比B高0.090731个百分点；AP51.385342，比B高1.934554个百分点。梯形PR-AUC为51.385757（B50.621882），它与非插值AP不同。epoch60→80的Fmax只增约0.0099，长训收益已进入平台。

已测热启动方案：共享头500步的Fmax52.238、AP42.259；joint3步后完整路径57.011/49.488，共享路径49.566/37.200。因此暂停当前改造协议合理，但不能由此证明共享GO评分或者原始几何方案均无效。

原训练history中epoch71–80平均约269.89秒/更新，没有直接显示提速。joint约123秒使用冻结GO/BN统计和不同优化路径，不能作为完整从头训练速度的等价对照。使用本包smoke10步的step10日志估计实测速度，不把CPU合成数据测试当GPU吞吐证据。

## 四个预设

| preset | 蛋白输入 | GO输入/读出 | 比较目的 |
|---|---|---|---|
| baseline | Stage1 x＋随机初始化共享线性投影 | 原v089本体证据＋独立逐GO分类器 | 新单卡优化条件下的工程对照 |
| ontology | 同上 | 本体编码＋共享protein–GO scorer | 与geometry隔离原始几何路径 |
| geometry | 同上 | 本体编码＋原始几何直连＋共享scorer | 主实验 |
| geometry_random | Stage1分支＋固定随机特征的共享投影 | 与geometry相同 | 单独考察随机特征分支 |

推荐先跑geometry的smoke和5epoch；需要判断raw路径贡献时，跑相同预算的ontology；geometry_random作为后续独立对照。不要把四组都同时放在一张卡上竞争资源。

所有预设：单GPU、3层SAGE、hidden320、global weak10240/core2560、accumulation2，每微批weak5120/core1280；core/weak损失权重1.0/0.25，原fanout，BN/dropout正常训练，seed1092；LR0.001、warmup2epoch、hold到10epoch、cosine至0.0001；固定总日程100epoch，每5epoch保存。

baseline的学习率、seed和单卡统计更新与历史v089双卡实验不同，不能称为原结果的精确复现。它与本包其他组匹配优化设置。第一轮5epoch是预算检查点，不是100epoch优化日程被缩成5epoch；早期达不到历史epoch80的57.8不自动代表失败。

## 原始几何如何接入

原始坐标 `[center, log(abs(offset).clamp_min(1e-8))]` 按所提供的本体几何逐维标准化，经过1024→512→320共享MLP，与原本体编码相加，再经过最终LayerNorm。最终融合向量同时供图中的GO证据聚合和共享评分使用。

ontology控制组具有相同最终LayerNorm，只有raw几何支路关闭。保留原GO-GO本体路径，不假定它破坏语义。全GO几何统计不使用训练标签频次、测试标签或测试蛋白特征。均值和尺度保存在checkpoint；GO几何及图关系仍由原本体契约绑定。

共享读出没有可训练逐GO分类器行和残差表，Stage2编码器与GO路径从头共同建立评分空间。沿用v090的动态目标GO行接口：可对已提供几何/关系的本体节点评分；接口可用不等于已验证严格零样本性能。

## 随机特征的准确含义

v089的node_encoder本来就是随机初始化、可训练的共享投影。新增可选分支为：

```
h_node = A(Stage1_x) + 0.1 * B(phi_R(Stage1_x)) + GO_evidence
```

`phi_R`为固定种子的Gaussian投影＋cos/sin，128维；随机矩阵和相位冻结并存入checkpoint，B为可训练共享线性投影。相同Stage1向量在任意batch/角色得到相同随机特征，新蛋白可直接计算，无需查找训练期蛋白ID向量。它是一个具体可测的非线性表示扩展，不新增独立生物信息。

本包没有默认增加“每个蛋白一个可训练随机ID向量”：它容易记忆训练身份，独立测试新蛋白没有受训ID参数。随机分支也不能消除Stage1已接触的监督信息。严格未见蛋白/未见GO评估仍需上游特征、伪标签、candidate及标签/时间切分的完整来源隔离。

## 安装

完整上传并解压ZIP，在解压目录运行：

```bash
python install_v092.py \
  --project-root /home/dataset-local/data_local/shaojiangyi/latence-project --apply
```

只新增v092模块、入口、配置、测试及本说明。复用现有v089/v090数据/训练/评估工具和已经验证过的P0 fast采样器；已有同名v092文件会备份。旧实验入口、配置和checkpoint不被替换。

## 运行：无需export或清理旧变量

每条命令在工程根目录执行，使用实际环境中的Python：

```bash
cd /home/dataset-local/data_local/shaojiangyi/latence-project

/opt/conda/envs/pytorch2.4/bin/python scripts/nbs/run_nbs_v092.py \
  smoke --preset geometry --gpu 0 --stop-step 10
```

smoke使用独立的`geometry_single12800_scratch_smoke`目录。包含真实数据构图、前向、反向、优化和checkpoint保存。日志的step10窗口仅含第6–10次更新，比启动总耗时更适合初步计时，并保存smoke_timing.json。它仍处于2epoch学习率warmup中，不代表稳态吞吐率。默认CPU线程数4，可用`--threads 1`覆盖，但改变线程数不能与旧history直接当采样器速度对照。

正式训练到5epoch：

```bash
/opt/conda/envs/pytorch2.4/bin/python scripts/nbs/run_nbs_v092.py \
  train --preset geometry --gpu 0 --stop-epoch 5
```

评估固定epoch5：

```bash
/opt/conda/envs/pytorch2.4/bin/python scripts/nbs/run_nbs_v092.py \
  evaluate --preset geometry --gpu 0 --epoch 5
```

续训至20epoch：

```bash
/opt/conda/envs/pytorch2.4/bin/python scripts/nbs/run_nbs_v092.py \
  resume --preset geometry --gpu 0 --stop-epoch 20
```

评估epoch20把`--epoch 5`改成`--epoch 20`。恢复到配置的100epoch可显式写`--stop-epoch 100`；不要修改配置epochs或把STOP_STEP当环境变量设置。

ontology、baseline和geometry_random的命令仅需更换`--preset`，各自输出目录自动隔离。常规运行无需`--work-dir`；自定义时请在该实验后续train/resume/evaluate中始终给出相同目录。

入口对子进程清除旧CONFIG/WORK_DIR/RESUME/INIT_FROM、旧GPU数量、DDP变量等，只采纳本条CLI参数。`--gpu 0`指物理卡0，子进程使用cuda:0。没有torchrun。新训练不会自动覆盖已有实验目录；已有目录请使用resume。

## 输出与评估

默认正式目录：

```
outputs/latence_nbs_experiments/v092/geometry_single12800_scratch
```

checkpoint：`nbs_epoch5.pt`、`nbs_epoch10.pt`等；`latest.pt`只用于严格同实验恢复。日志：`training_history.json`、`resolved_config.json`、`phase_manifest.json`。

原始评估产物保留在`eval_step*/full/`。额外的`metrics_v092/summary.json`报告整体protein-Fmax、精确非插值micro AP、按core频次0/1–5/1–9/10–99/100–999/≥1000的分层结果，以及B对照。1–5与1–9分档重叠，不能相加。额外报告重用已生成概率，不再运行GNN。

仍保持1800蛋白、21312 BP列、严格 `>`、0.001阈值网格，无概率融合，无GO分数传播。旧v090的主评估器是Fmax-only，所以AP需看额外报告；AP不是梯形PR-AUC。本包不自动搜索最佳测试checkpoint。正式选模应使用独立开发集；Stage2 core holdout仍仅是训练监控，不能自动称为Stage1未见验证。

## 版本与恢复约束

v092是新增结构与实验入口。为复用既有严格评估格式，baseline保留v089 checkpoint外层协议，三个共享头组保留v090外层协议；并非加载旧模型权重。真实结构由完整新model_config、`experiment_version=0.9.2`、architecture_extension及新模型/runner/fast采样器文件哈希绑定。

只能用本入口恢复自身checkpoint。不要用旧v089/v090/v091 checkpoint初始化或恢复新预设。不要在运行后修改配置或源码再期待严格resume继续通过。

## 本地验证范围

在PyTorch2.4.1+cpu下验证真实模型前向、反向、raw路径与本体路径梯度、随机分支的新输入支持、动态GO行、全seen损失与旧BCE一致、state恢复和参数更新；另验证原phase trainer的真实训练/断点恢复/固定checkpoint推理。源数据读取和外部CLI评估部分使用小型合成fixture。

这不代表实际数据、A100显存、训练速度或预测性能已验证。先运行上述smoke，再开始正式5epoch试验。包内测试可在安装后执行：

```bash
/opt/conda/envs/pytorch2.4/bin/python -m unittest discover -s tests -p 'test_v092*.py' -v
```

后续优先提供：smoke的training_history（计时和显存）、正式训练的training_history、epoch5的metrics_v092/summary.json。先比较ontology与geometry，再决定是否开展随机分支、candidate证据dropout或排名聚合；不要同时叠加后难以归因。
