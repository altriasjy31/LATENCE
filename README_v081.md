# LATENCE NBS v0.8.1：图排序、结构支持与四组评估

本包用于已经可以运行 v080 pilot 的 LATENCE 项目。训练目标是利用图关系超过原始 expert probability，并以已经改进过的 Stage1 modelout 为强对照。训练期间不读取独立测试标签；expert 与 modelout 的独立测试矩阵只交给评估器。

## 1. 安装位置与安装命令

可以把压缩包解压到 `latence-project` 的同级目录。解压后目录名为 `latence_nbs_v081`，其中包含 `install_v081.py`。

在 **latence-project 根目录**运行：

```bash
python ../latence_nbs_v081/install_v081.py --project-root .
```

如果解压到了项目内部，则改为：

```bash
python latence_nbs_v081/install_v081.py --project-root .
```

安装器只复制 `package_files.json` 中列出的代码、配置和测试文件。现有文件有差异时会先备份再安装，**不会再因 Base files differ 而停止**。备份位于 `outputs/nbs_code_backups/v081_时间/`。不会修改模型权重、数据缓存内容或现有训练结果。包内新增的 v081 配置不会覆盖您调整过的 v080 配置。训练脚本和旧模型保留，便于对照。

依赖沿用当前环境：Python 3.11、PyTorch 2.4、NumPy 及项目现有依赖。精确 micro-AP 还需要 scikit-learn；缺少时评估器明确报错，不会静默换成 histogram AP。

## 2. 本次实质修改

| 配置 | 模型 | 监督 |
|---|---|---|
| `baseline` | 原 v080 模型 | 原 v080 loss |
| `loss` | 原 v080 模型 | 支持加权 PU、多正例排序、分层 KL |
| `graph` | GO 条件 weak/core 邻居读出、向量交互、辅助图分数 | 新 loss + 图分数辅助排序 |

三组均预测完整21,312个 BP GO，保留全部已有 modelout 正例。`baseline` 在相同 toy 数据和种子上已验证与 v080 的参数更新一致。

- weak 的软标签来自 expert-assisted **Stage1 modelout**，不是原始 expert probability。正例保留原有 CSR 成员及置信度。
- 结构支持来自其他训练 core 的金标和固定检索相似度，不读取当前模型分数或可学习 gate。默认至少2个邻居、相似度至少0.5、加权注释票数比例至少0.5，才产生支持权重。这些是待验证的保守启发式，不是生物学正确率。
- 有支持的未知位置降低 ASL、ranking 和 KL 权重，最低为原来的0.25；按位置数量归一化，避免权重衰减被分母抵消。**不将其直接改成正例。** 已知正例的同映射 GO 别名从 PU 中排除，但所有任务列仍保留。
- 除hard64和background64外，对剩余高于ASL clip的未知位置加入medium PU。KL按高base、明显升分和其余位置分别归一化。
- 所有已有正例对top16+random16个PU训练相对排序，支持权重同样生效。
- `graph` 组用当前GO query分别读取weak候选GO和core邻居，保留向量交互。辅助图分数不直接读取base logit、candidate pair或core vote标量；弱边置信度仍参与图编码。
- 训练时随机丢弃15%的候选边，规则与目标标签无关，同时删除对应的直接候选证据；评估时关闭。最终残差层零初始化，初始输出等于backbone；辅助图头从第一步学习。
- GO分块和activation checkpoint控制训练中间激活；最终加残差和loss为FP32。GPU吞吐和显存仍须在您的A100上实测。

## 3. 现在先运行什么

以下命令都在项目根目录运行，默认两卡。已有的 v080 core 邻居准备缓存继续复用。

```bash
bash scripts/nbs/run_nbs_v081.sh smoke graph
bash scripts/nbs/run_nbs_v081.sh pilot loss
bash scripts/nbs/run_nbs_v081.sh pilot graph
```

`smoke` 运行20步，单独保存到 `outputs/latence_nbs_experiments/v081/smoke_graph`，不会把这20步权重带入pilot。两组pilot各从头训练600步，保存100、300、600步。需要一张卡时设置 `NUM_GPUS=1`；需要限定物理卡时使用现有 `CUDA_VISIBLE_DEVICES`。

当前 v080 的600步可以作为初始基线。需要完全相同入口的对照时，再运行：

```bash
bash scripts/nbs/run_nbs_v081.sh pilot baseline
```

不要把 v080 checkpoint resume 到新loss或新graph配置中。三组的改变是有意的，需要各自初始化。已有训练目录会拒绝覆盖；重跑请设置新的 `WORK_ROOT` 或 `WORK_DIR`。

## 4. 自动重算第一阶段参照，再统一评估 E/M/B/G

本次更新不改变训练模型或loss，已有v080/v081检查点可以继续使用。**只为补齐参照，不需要重新运行 `pilot baseline`。** 安装命令仍是第1节的命令。

已按本次提供的第一阶段脚本核对：

| 符号 | 实际输出 | 自动化处理 |
|---|---|---|
| E | 第一阶段的 `external_only`，也称expert probability | 读取同一external源并对齐蛋白/GO |
| M | 第一阶段完整 `modelout` | 重跑完整decoder，直接取 `sigmoid(qout["logits"].float())` |
| B | 第一阶段backbone | 继续使用NBS输入缓存的B；另外重算B核对一致性 |
| G | 第二阶段图模型 | 按指定检查点输出完整任务概率 |

上传的第一阶段诊断代码计算了M，但只保存指标，没有保存M矩阵。此前NBS输入准备代码只导出了B。因此，已有NBS中标为 `backbone_base` 的结果确实是B；这不能证明第一阶段原评估没有计算M，也不能将G超过B解释为超过E或M。

自动参照入口复用项目已有的Stage1模型与严格checkpoint加载函数，不覆盖第一阶段的诊断脚本。默认M对应弱监督manifest记录的 `modelout::mix_expert_base_anchor::decoderprob::expert`，保留实际checkpoint的anchor模式、混合系数、selector和learned gate。不会用 `base_logits + raw_delta` 代替最终modelout，不扫描独立测试指标来选混合参数，也不运行IC融合分析。若manifest记录的是其他decoder输入源，入口会明确报出不支持的键。

**通常直接运行以下命令即可：**

```bash
# 如果曾设置上一版示例路径，先清除；已提供真实的成对参照时可保留手动方式。
unset EXPERT_PROB MODELOUT_PROB

# 可选：先单独生成参照，核对输出来源。
bash scripts/nbs/run_nbs_v081.sh references

# 自动生成/复用参照后，评估固定100、300、600步。
bash scripts/nbs/run_nbs_v081.sh series graph
# 已训练loss组时：
bash scripts/nbs/run_nbs_v081.sh series loss
```

`evaluate` / `series` 默认自动准备参照，**不要求提前运行 `references`**。系列评估只准备一次；loss、graph、baseline和各检查点共用缓存。Stage1进程退出、释放显存后才启动NBS推理。训练动作不会触发独立测试参照生成。

E默认来源与您上传的第一阶段runner一致：

```text
data/external_probs/esm2_3b/bp_predictions.prop.float16.npy
```

它是第一阶段实际使用的external文件，包含文件名所表示的prop版本；代码不会自动改用无`.prop`的另一个文件。若第一阶段设置过不同来源，只需沿用那个路径：

```bash
export EXTERNAL_PROB_PATH=/实际路径/bp_predictions.prop.float16.npy
bash scripts/nbs/run_nbs_v081.sh references
bash scripts/nbs/run_nbs_v081.sh series graph
```

Stage1 checkpoint默认从当前 `INPUT_DIR` 的weak graph manifest读取，并核对其与独立输入缓存的checkpoint哈希。MSA默认 `data/ind_MSA_bin/index.pkl`，模型结构从checkpoint/相邻args.json恢复。需要覆盖路径时使用第5节中的 `STAGE1_*` 变量；不会继承训练集MSA索引。

默认输出在：

```text
INPUT_DIR/stage1_references_v081/
  expert_prob.f32.npy
  stage1_modelout.f32.npy
  backbone_recomputed.f32.npy
  protein_ids.txt
  go_ids.txt
  stage1_reference_manifest.json
```

manifest记录external/checkpoint/代码来源、解析后的模型参数、ID契约、输出哈希及B差异。四组比较JSON内也会包含这份生成记录。缓存默认 `reuse`：匹配则复用，不匹配则重新生成；`require`只允许复用已有兼容缓存，首次生成请保留默认值。文件准备失败时不发布半成品。

**行列对齐的边界：** 第一阶段 `PseudoProbDataset` 默认假定external行序与metadata的ind_test蛋白同序，列序与原任务分类器同序；本入口继承该契约，再按NBS输入ID重排。若原external文件本身的排序不明，仅凭矩阵数值无法自动证明语义一致。可提供 `STAGE1_EXTERNAL_PROTEIN_IDS` / `STAGE1_EXTERNAL_GO_IDS`，按真实ID核验；无GO sidecar时，manifest明确标记该列序来自第一阶段契约，未独立验证。任务中的alt-ID列保留，不合并成canonical列。

MSA预处理沿用当前NBS输入准备的零worker流程；历史第一阶段runner默认4 workers，采样流可能不同，因此不承诺与任意旧诊断逐值相同。重算B的比较用于核对当前参照与NBS输入是否一致。

与第一阶段一致，E进入decoder前经float16再转float32；M直接保存float32。旧诊断默认把输出暂存为float16，因此新旧数字可能有轻微量化差异。E/M/B/G现在由同一个评估器、同一标签及同一列顺序计算：正式主表用 `stage1`，另算exact micro-AP和P/R@10/50/100。旧诊断的histogram micro-AP不能直接当作此主表AUPRC。

B校验默认要求最大概率差≤0.002，并记录均值与超差位置数。这是数值一致性容差，不是性能阈值。超差时先检查checkpoint、MSA采样、模型配置和AMP，不要仅扩大容差；已有B和NBS图输入不会被替换。

只评估一个检查点，或补评已有v080结果：

```bash
CHECKPOINT=outputs/latence_nbs_experiments/v081/graph/nbs_step600.pt \
  bash scripts/nbs/run_nbs_v081.sh evaluate graph

CHECKPOINT=outputs/latence_nbs_experiments/v080/main/nbs_step600.pt \
  bash scripts/nbs/run_nbs_v081.sh evaluate baseline
```

后者写到v081/baseline目录，不修改v080结果；要求数据、GO映射及core留出划分与检查点一致。现有G预测通过哈希校验后可以复用，只重新计算参照指标。

手动提供已有参照仍受支持：同时设置 `EXPERT_PROB` 与 `MODELOUT_PROB`，并分别提供 `EXPERT_PROTEIN_IDS`、`MODELOUT_PROTEIN_IDS`、`EXPERT_GO_IDS`、`MODELOUT_GO_IDS`；或在确已核实同序时显式设置 `REFERENCES_ALIGNED=1`。只设置一份参照会提示补齐或清除两者，避免混用不同来源。`EXPERT_PROB`是导出参照，`EXTERNAL_PROB_PATH`是Stage1原始输入源，两者不要混淆。

仅需G/B诊断时，可显式使用 `ALLOW_MISSING_REFERENCES=1` 跳过自动参照生成；报告会标记 `incomplete_references`，不能据此判断是否超过E/M。

## 5. 配置和输出位置

默认主入口设置如下，可通过环境变量覆盖，也可直接编辑shell顶部：

| 变量 | 默认值/用途 |
|---|---|
| `NUM_GPUS` | `2` |
| `STEPS` | `600`，smoke使用`SMOKE_STEPS=20` |
| `WORK_ROOT` | `outputs/latence_nbs_experiments/v081` |
| `WORK_DIR` | `WORK_ROOT/variant`，单次实验目录 |
| `INPUT_DIR` | `outputs/latence_nbs_eval/bp_shared_full_task_top512_v071_v2` |
| `METADATA_FILE` | 沿用现有`unidata_with_exp_train_pseudo.pkl`绝对路径 |
| `CONFIG` | `bp_full_task_v0.8.1_baseline/loss/graph.json`中的对应文件 |
| `CHECKPOINT_STEPS` | `100 300 600`，供`series`使用 |
| `ABLATION` | `full`、`weak_off`、`core_off` |
| `METRIC_BACKEND` | `stage1`，正式主表沿用第一阶段评估器 |
| `AUPRC_MODE` | `exact`，另算精确micro-AP；不会替换Stage1主口径 |
| `AUTO_REFERENCES` | `1`，评估自动导出/复用E/M；成对手动参照优先 |
| `EXTERNAL_PROB_PATH` | 第一阶段原external输入；默认`data/external_probs/esm2_3b/bp_predictions.prop.float16.npy` |
| `STAGE1_EXTERNAL_PROB_PATH` | 可单独覆盖上述来源，优先级更高 |
| `STAGE1_EXTERNAL_PROTEIN_IDS` / `STAGE1_EXTERNAL_GO_IDS` | 原external矩阵的ID文件；文本每行一个ID或一维`.npy` |
| `STAGE1_CHECKPOINT` | 默认从当前输入weak graph manifest自动解析并核验 |
| `STAGE1_MSA_INDEX` | 默认`data/ind_MSA_bin/index.pkl` |
| `STAGE1_MODEL_CONFIG` / `STAGE1_TRAIN_ARGS_JSON` | 可覆盖模型配置或args.json路径；args默认自动查找 |
| `STAGE1_DIAGNOSTIC_SCRIPT` | 默认`experiments/eval_weak_ind_test_detr_diagnostics.py`，仅复用参数定义与模型模块 |
| `STAGE1_REFERENCE_DIR` | 默认`INPUT_DIR/stage1_references_v081` |
| `STAGE1_REFERENCE_CACHE_POLICY` | `reuse`；可设`require`或`refresh` |
| `STAGE1_EVAL_DEVICE` / `STAGE1_EVAL_BATCH_SIZE` | 默认`DEVICE`及`8` |
| `STAGE1_NO_AMP` / `STAGE1_AMP_DTYPE` | 默认`0`及`bfloat16`，用于与已有B计算口径匹配 |

模型和loss参数在JSON的 `full_task.model`、`full_task.loss`、`full_task.structural_support` 内。其余配置段用于接入已有数据与本体；旧 `episode.num_queries` 不控制新模型的全GO输出数量。

最重要的文件：

```text
outputs/latence_nbs_experiments/v081/graph/
  resolved_config.json
  training_history.json
  validation_history.json
  nbs_step100.pt / nbs_step300.pt / nbs_step600.pt
  latest.pt
  best_core_holdout.pt
  eval_step600/full/metrics/
    nbs_w2s_comparison.json
    nbs_w2s_comparison.tsv
    nbs_ind_test_metrics.json
    evaluation_details.log
```

`best_core_holdout.pt`仅是Stage1已见core上的监测结果。默认评估`latest.pt`或您指定的固定检查点，不根据99.8附近的core AP自动判断是否继续，不用独立测试选择最佳epoch。

`nbs_w2s_comparison`同时报告E/M/B/G及G−E、G−M、G−B。`evaluation_goal_complete`表示四组正式比较是否齐全，**不是性能成功标记**。应查看预先确定的主指标差值及Top-k变化，而非把“比较齐全”当“超过专家”。本包不宣称统计显著性。

## 6. 新日志如何看

```text
loss = fit + aux
fit = comparable_objective
aux = contrib_graph_aux
 gain = base_objective - fit
```

辅助图排序没有对应的backbone目标，因此不计入`gain`；它仍真实进入总loss与梯度。不同组总loss绝对值不可直接比较。

主要关注 `contrib_ranking`、graph组的 `contrib_graph_aux`、`supported_unknown_pairs_per_protein`、`hard_pu_mean_attenuation`、`positive_target_kl`，以及独立评估的G−E/G−M。支持数为0不会报错：这说明当前邻域没有满足该规则，应结合结果判断阈值与图关系，而不是强行制造正标签。

## 7. 后续训练与路径消融

确认短程方向后，同一组延长到4,000步，保留原配置和优化器状态：

```bash
STEPS=4000 RESUME=outputs/latence_nbs_experiments/v081/graph/latest.pt \
  bash scripts/nbs/run_nbs_v081.sh train graph
```

修改模型、loss、结构支持规则后应使用新目录从头训练。代码会拒绝不兼容的resume；记录backbone概率文件身份及clip，防止换了残差基座却继续旧训练。

固定600步做推理通道消融：

```bash
CHECKPOINT=outputs/latence_nbs_experiments/v081/graph/nbs_step600.pt \
  ABLATION=weak_off bash scripts/nbs/run_nbs_v081.sh evaluate graph
CHECKPOINT=outputs/latence_nbs_experiments/v081/graph/nbs_step600.pt \
  ABLATION=core_off bash scripts/nbs/run_nbs_v081.sh evaluate graph
```

这检查模型对通道的依赖，不等同于分别从头训练的机制消融。

## 8. 验证范围

CPU PyTorch 2.4.1测试覆盖多正例与PU梯度、支持权重、alias、两路径辅助梯度、标签隔离、分块重算一致性、断点恢复、基线对照、真实2,903列导出及四组评估。最终测试结果见包内 `VALIDATION_v081.md`。

本环境没有GPU及您的完整checkpoint/数据，未验证A100吞吐、真实DDP训练或完整Stage1前向。新增自动参照已做CPU契约和数值路径测试，真实数据由 `references` 的B一致性校验确认。若只是安装本次评估更新，可直接补评已有检查点；只有首次训练新graph/loss方案时，才需要先做20步GPU smoke。效果以同口径E/M/B/G结果为准。
