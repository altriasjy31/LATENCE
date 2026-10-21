# LATENCE NBS v0.8.2 — 完整教师监督、全集分类与局部 matching

版本：0.8.2。适用于已运行 v0.8.0/v0.8.1 的现有 LATENCE 项目。
这是可安装的代码更新包，包含新模型、loss、数据读取、训练/评估入口、配置及测试；不包含数据和 checkpoint。

## 1. 8000步结果说明了什么

同一评估口径下，G 的结果为：

| 方法 | 主 Fmax | 主 AUPRC | exact micro-AP |
|---|---:|---:|---:|
| Backbone B | 55.6317 | 32.0087 | 49.4508 |
| G，600步 | 55.5232 | 34.6488 | 49.6483 |
| G，4000步 | 55.8969 | 36.9541 | 50.9435 |
| G，8000步 | 55.8189 | 36.9255 | 51.4998 |
| Expert E | 69.7811 | 56.4960 | 67.5547 |
| Stage1 modelout M | 69.7716 | 59.2118 | 68.7907 |

数值单位为百分数。4000→8000，主Fmax −0.0780个百分点、主AUPRC −0.0286；exact micro-AP +0.5563。应称为“主指标平台，部分排序仍改善”，不能说完全没有学习，也不能据此断言已经达到模型上限或发生过拟合。没有重复种子和差值置信区间，不将小幅下降解释为显著退化。

新增完整history中，3501–4000步与7501–8000步的窗口均值：

| 训练量 | 3501–4000 | 7501–8000 |
|---|---:|---:|
| comparable主目标 | 0.191291 | 0.185376 |
| 总loss | 0.202920 | 0.194888 |
| positive_target_kl | 0.039632 | 0.035712 |
| hard位置平均概率变化，相对B | −0.063002 | −0.083541 |
| 正例平均概率变化，相对B | −0.021077 | −0.019795 |
| 平均绝对logit修正 | 0.688019 | 0.771968 |
| 学习率 | 0.0003 | 0.0003 |

优化仍在进行，而这些训练收益未充分转化为主Fmax/AUPRC。8000步时G仍比E低约13.96个Fmax百分点、19.57个主AUPRC百分点。超过B只是中间进展；最终目标仍是图模型超过E，并同时报告与M的差距。

当前v081已经解码全部21,312个GO，也覆盖CSR内全部正例。不能再用旧版“每蛋白只监督1个正例”解释这次平台。源码中的更直接问题是：完整M被截断为>0.5的CSR；其余位置进入PU和B锚定路径。M=0.49的软信息因此消失，甚至可能受到向0、向B的压力。v082首先改变这一目标，再分别检验分类读出与matching增量。

训练快本身不能证明模型无效。当前使用冻结蛋白表征、缓存的8个core邻居及简化图编码；本版继续利用它做快速、有归因的验证，不因速度快而直接扩充图规模。

## 2. 实际修改

| 环节 | v082行为 |
|---|---|
| weak监督 | 从原训练weak manifest声明的dense modelout读取完整M；全部GO使用Bernoulli软目标KL |
| 监督归一化 | 每蛋白按M<0.05、0.05≤M<0.5、M≥0.5三层分别平均，再平均非空层 |
| weak负样本 | 移除旧weak向0的PU及默认B锚定；低概率M仍提供软负约束 |
| core监督 | gold正例＋谨慎PU，保留别名排除和结构支持衰减 |
| 蛋白覆盖 | 所有weak registry行均可进入训练，包括没有>0.5 pseudo的行；保留可恢复的随机无放回循环 |
| classification C | 类别专属全集读出；共享protein、weak→GO、weak→core→GO编码；以B为初始化残差起点，可自由修正 |
| matching G | 仅选中query做GO条件attention与残差；G=C+局部修正，未选GO逐元素等于C |
| 分支监督 | C对全集监督；额外refinement只在选中位置归一化，不将相同全集loss重复计算为两份 |
| 评估 | 分别导出C和G，与同一E/M/B比较；summary绑定真实预测、checkpoint、输入和分支 |
| 恢复 | 保存optimizer、蛋白队列、每rank RNG；拒绝用v081 checkpoint续训新目标 |

默认query预算256：96个分类高分、64个靠近0.5的不确定位置、48个weak候选、32个core支持、16个探索位置。顺序去重，不足名额按分类排名补齐。训练探索使用可恢复RNG，评估使用固定种子顺序，不依赖batch划分。教师概率和gold不进入query选择或模型forward；全集正例覆盖由C承担，不要求局部matching包含全部正例。

默认weak/core各按角色均值等权，因此64 weak＋16 core不意味着80%的梯度权重归weak。默认refine_weight=0.5。首轮不额外叠加ranking loss、旧graph auxiliary loss或扩大邻居数，避免同时引入更多难以归因的变化。

这是可移植的双分支读出验证，尚未恢复原NBS的多关系PPI整图编码。两条weak-to-strong路径保留；待分类和matching收益得到验证，再替换共享编码器，保留本版全集监督及局部matching接口。

## 3. 安装位置和命令

解压到 latence-project 同级目录即可，也可以放在其他位置。安装命令的 --project-root 指向真正项目根目录，解压目录不需要放进项目内部。

若目录为 /path/to/latence-project 和 /path/to/latence_nbs_v082：

~~~bash
cd /path/to
unzip latence_nbs_v082.zip
python latence_nbs_v082/install_v082.py --project-root /path/to/latence-project
cd /path/to/latence-project
~~~

安装器核验包内文件，备份发生变化的目标文件后安装；不使用旧版“base文件必须完全相同”的阻塞方式。备份位于项目 outputs/nbs_code_backups/v082_时间戳。重复安装相同文件无副作用。

v082默认输出至 outputs/latence_nbs_experiments/v082；旧v081 checkpoint和实验输出不改动。新目标需要重新初始化，不使用v081 latest.pt作为RESUME。真正同一v082实验可续训。

## 4. 训练数据检查

v082自动从配置中的 data.weak_graph_predictions_manifest 找到weak角色的 modelout_dense_file，通常为原导出目录的 modelout_weak_prob.f16.npy。原exporter默认保存dense modelout，多数情况下可以直接复用。

这是约489,222个训练weak蛋白的M，不是1800个独立测试蛋白的M；后者只用于评估，绝不能替代训练teacher。

~~~bash
bash scripts/nbs/run_nbs_v082.sh prepare dual
~~~

prepare打印teacher路径、shape、dtype、SHA256，然后核验/复用core邻居缓存。首次计算约20.9GB FP16教师文件的哈希会有顺序读取开销；训练按batch mmap读取，不把整份teacher放入GPU。若旧缓存遗漏无pseudo的weak行，会重新准备邻居；本次提供的运行数据原先weak eligible已为489,222，不能把这一通用修复宣称为本次主要瓶颈。

读取器核验蛋白ID顺序、registry、GO列、来源checkpoint及modelout语义，并在取batch时核对CSR已有正例的概率与dense M一致。缺少dense文件会在prepare报出明确文件路径，不会用CSR零填充。

若确实缺失：用原训练weak graph导出命令及相同checkpoint、registry和modelout参数重新导出，显式设置 --save-dense-modelout true。不要仅手改manifest添加文件名，也不要把backbone或expert重命名为modelout。原导出入口为 scripts/export_weak_graph_predictions.py / scripts/run_export_weak_graph_predictions.py；具体路径参数沿用已生成full_task_top512缓存的命令。

默认配置沿用v081的数据路径及BP词表。若本地位置不同，修改相应JSON的data路径，或使用 CONFIG=/实际配置.json。训练三组必须使用相同数据与split。

## 5. 建议运行顺序

三组命名如下：

| 变体 | 模型 | 主要对照问题 |
|---|---|---|
| teacher | v081 query-conditioned全GO模型＋新完整M目标 | 修正教师截断和监督目标是否有效 |
| classifier | 共享图编码＋全集类别专属分类头 | 全集分类读出是否比全GO matching易优化 |
| dual | classifier＋局部matching | 动态匹配是否在C基础上增加收益 |

teacher是完整监督控制组，不是额外训练第一阶段teacher。

先做每组20步smoke，输出自动隔离：

~~~bash
bash scripts/nbs/run_nbs_v082.sh smoke teacher
bash scripts/nbs/run_nbs_v082.sh smoke classifier
bash scripts/nbs/run_nbs_v082.sh smoke dual
~~~

默认2卡。单卡用 NUM_GPUS=1 DEVICE=cuda:0；改变卡数会改变global batch/覆盖，正式三组应保持一致。

先固定600步比较，比直接三组各跑8000步更节省预算：

~~~bash
STEPS=600 bash scripts/nbs/run_nbs_v082.sh pilot teacher
STEPS=600 bash scripts/nbs/run_nbs_v082.sh pilot classifier
STEPS=600 bash scripts/nbs/run_nbs_v082.sh pilot dual
~~~

pilot与train调用相同训练逻辑；未指定STEPS时默认4000，不会根据已见过Stage1的core holdout自动拒绝后续实验。默认保存600/2000/4000、每1000步及最终步checkpoint。

配置目录：
nbs_models/nbs_protein_go/configs/bp_full_task_v0.8.2_{teacher,classifier,dual}.json

用于一组的修改可以写入新的CONFIG路径，但应保留未修改控制组。新模型/损失/采样配置要用新WORK_ROOT；RESUME只用于同一配置延长总步数。

## 6. 评估：自动复用E、M，dual同时评估C和G

默认沿用v081已确认的独立测试输入目录和Stage1参考缓存。若位置不同：

~~~bash
export INPUT_DIR=/实际路径/bp_shared_full_task_top512_v071_v2
export STAGE1_REFERENCE_DIR="$INPUT_DIR/stage1_references_v081"
export METADATA_FILE=/实际路径/unidata_with_exp_train_pseudo.pkl
~~~

没有手工设置EXPERT_PROB和MODELOUT_PROB时，shell调用现有Stage1 reference helper，按实际decoder的modelout方式生成或核验E/M缓存；不会把B当M。它复用v081 reference helper，因为第一阶段计算定义未变。

~~~bash
bash scripts/nbs/run_nbs_v082.sh evaluate teacher
bash scripts/nbs/run_nbs_v082.sh evaluate classifier
bash scripts/nbs/run_nbs_v082.sh evaluate dual
~~~

dual默认先评估final G，再评估同一checkpoint的classification C。不要把C称作独立训练的classifier组：前者与matching共同训练，后者没有matching。

若需要两条图路径的消融：

~~~bash
ABLATIONS="full weak_off core_off" bash scripts/nbs/run_nbs_v082.sh evaluate dual
~~~

这是推理时关闭路径的依赖性诊断，不能等同于重新训练的结构消融。默认不自动把全部checkpoint乘上三种消融；先评估full，只有需要归因时才扩展。

仅评估已训练的600步：

~~~bash
CHECKPOINT_STEPS="600" bash scripts/nbs/run_nbs_v082.sh series dual
~~~

已有完整4000步时：

~~~bash
bash scripts/nbs/run_nbs_v082.sh series dual
~~~

默认series为600/2000/4000，缺少checkpoint会先报错。指定单个文件：

~~~bash
CHECKPOINT=outputs/latence_nbs_experiments/v082/dual/nbs_step2000.pt \
  bash scripts/nbs/run_nbs_v082.sh evaluate dual
~~~

最终输出：
- v082/dual/eval_step600/full/nbs_ind_test_prob.f32.npy：G
- v082/dual/eval_step600/full/classification/nbs_ind_test_prob.f32.npy：C
- 各目录内 nbs_full_task_prediction_manifest.json
- 各目录的 metrics/nbs_w2s_comparison.json 和 .tsv
- classification summary的symbol为C，差值为C_minus_E/M/B；final使用G_minus_E/M/B
- classification目录仍有兼容旧工具的NBS_final方法键，以symbol和branch区分C/G

summary记录预测矩阵SHA、checkpointSHA及路径、prediction manifest SHA、input SHA、step、variant、branch和模型实现哈希，解决旧compact summary没有绑定G来源的问题。缓存不匹配会拒绝复用；无需删除原结果，可设新的WORK_DIR。

E和M的外部GO列身份若只有原Stage1顺序声明，没有独立GO ID文件，原参考manifest仍会保留未独立核验标志，本版不会将其自动改成“已验证”。

## 7. 如何判断下一步

第一目标始终是主Fmax与同口径主AUPRC同时改善，并最终超过E。exact micro-AP、P/R@10/50/100用于解释变化，不能替代主AUPRC。不同loss定义的绝对值不能跨版本直接比较。

600步先检查：
1. teacher_pairs_per_weak_protein应等于21312；contrib_teacher和core均有限。
2. dual的selected_pairs_per_protein应为256，contrib_refine有限；分类和matching不是始终零更新。
3. teacher_kl各置信层是否改善，以及core_positive是否被严重牺牲。gain_vs_backbone_objective只是同位置训练目标改善。
4. C与G的主指标、Top-k以及G−C。只看到总loss下降不构成通过。

若T明显优于旧v081，支持完整监督改造；若classifier进一步优于T，支持分类读出；若dual的G进一步优于自身C与独立classifier，才支持matching带来额外收益。单次种子只提供初步证据，不能保证每一步或每个指标单调上升。

若600步仍未超过E，不应仅据此宣判失败；看同预算三组之间是否出现清楚方向，再按预定2000/4000步验证。若G不如C，先停用matching追求可复现的C收益，检查选中高教师概率位置覆盖、core支持和局部损失，不能用更多训练步掩盖有害分支。

同一v082实验从600续到4000：

~~~bash
STEPS=4000 RESUME=outputs/latence_nbs_experiments/v082/dual/latest.pt \
  bash scripts/nbs/run_nbs_v082.sh train dual
~~~

其余两组同理。续训保留模型、optimizer、RNG和蛋白循环；只延长STEPS不会重置学习率warmup。改变model/loss、数据或训练实现会拒绝原地resume，应开新实验。

core holdout约99.8%的结果仍只作为数值/训练监视，因为Stage1已经见过这些core蛋白。后续模型选择应使用Stage1未训练过的固定标注开发集，最终独立测试用于确认。当前接口沿用ind_test评估目录，不冒充已经实现新的独立开发集；频繁用同一ind_test结果设计模型会使其逐渐承担开发集角色。

待监督和双分支得到明确收益后，再对core检索池8→32/64、GO条件邻居聚合及原NBS多关系编码做单独对照。本版没有同时扩大邻居数，也没有恢复昂贵的GO-row filler流程。

## 8. 验证范围与复现

包内VALIDATION_v082.md记录本地测试结果。测试覆盖真实mmap小数据、多个任务列及别名、软教师、分支梯度、局部修正范围、断点恢复、真实评估器的C/G导出和来源绑定。测试环境为CPU PyTorch 2.4.1；没有用户完整数据和GPU，尚不能声称BP实际训练或双GPU性能已验证。

安装后可运行：

~~~bash
PYTHONPATH="$PWD:$PWD/nbs_models/nbs_protein_go:$PWD/nbs_models/nbs_protein_go/tests" \
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  python -m pytest -q \
  nbs_models/nbs_protein_go/tests/test_full_task_data_v082.py \
  nbs_models/nbs_protein_go/tests/test_full_task_model_v082.py \
  nbs_models/nbs_protein_go/tests/test_full_task_loss_v082.py \
  nbs_models/nbs_protein_go/tests/test_full_task_runner_v082.py \
  tests/test_full_task_eval_v082.py tests/test_run_nbs_v082_shell.py
~~~

如只回传一轮实验结果，优先提供三组resolved_config、training_history，以及固定步骤的comparison；dual同时提供final与classification两份comparison。无需新增整套分析目录或重复上传大概率矩阵。
