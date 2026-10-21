# LATENCE NBS v085（训练/评估 0.8.5）

本包在已安装的 v084 上增补评估与训练策略。保留 v084 的两层关系 SAGE、全 GO matcher、二值 weak pseudo 监督和 loss，不加入 Modelout/Expert 概率到图前向或 dense teacher loss。**原 v084 运行文件不修改，旧 checkpoint 仍用原入口运行。** v085 runner/evaluator 版本为0.8.5，复用模型/数据实现仍为0.8.4；配置中 `stage.name=nbs_v084_*` 表示复用的数据监督协议，不代表安装失败。

详细分析见 `ANALYSIS_v085.md`；新对话可直接使用 `NEW_CHAT_PROMPT_v085.md`。

## 1. 安装

解压目录可与项目同级，也可放在别处；`--project-root` 指向真正的项目目录。在项目目录中：

```bash
unzip /实际下载位置/latence_nbs_v085.zip -d ..
python ../latence_nbs_v085/install_v085.py --project-root .
```

安装器检查 v084 依赖，只安装新增文件，必要时备份同名 v085 文件；重复安装相同包会跳过。无需覆盖整个项目，无需修改 `helper.py`。本包不含模型权重、真实数据或新的开发集。环境沿用现有 PyTorch/NumPy/scikit-learn。

## 2. 第一件事：重算已有 v084 预测，不重训

确认当前工作目录为 `latence-project`。原比较 JSON 已保存 E/M/B/G 数组位置，入口自动复用它们：

```bash
COMPARISON_PATH=outputs/latence_nbs_experiments/v084/fixed/eval_step2000/full/metrics/nbs_w2s_comparison.json \
  bash scripts/nbs/run_nbs_v085.sh recompute
```

默认 metadata 为本次已有的 `/home/dataset-assist-0/datafile/latence-dataset/unidata_with_exp_train_pseudo.pkl`，需要时通过 `METADATA_FILE` 覆盖。默认输出在原 metrics 目录下的新子目录 `metric_audit_v085/`：

- `metric_audit_v085.json`
- `metric_audit_v085.tsv`

该步骤只读取缓存概率、标签和身份文件，校验哈希/行列顺序，不加载模型、不运行图推理、不重导出 Stage1，也不覆盖原比较结果。仅上传的摘要JSON不足以重算，必须在保存概率矩阵的服务器运行。

一次重算四份 full 输出：

```bash
for variant in fixed dynamic; do
  for step in 600 2000; do
    COMPARISON_PATH="outputs/latence_nbs_experiments/v084/$variant/eval_step$step/full/metrics/nbs_w2s_comparison.json" \
      bash scripts/nbs/run_nbs_v085.sh recompute
  done
done
```

重算已有 step2000 的图干预输出（仅对已生成的目录运行）：

```bash
for variant in fixed dynamic; do
  for ablation in weak_off core_off pp_off graph_off go_shuffle; do
    COMPARISON_PATH="outputs/latence_nbs_experiments/v084/$variant/eval_step2000/$ablation/metrics/nbs_w2s_comparison.json" \
      bash scripts/nbs/run_nbs_v085.sh recompute
  done
done
```

若磁盘目录迁移，shell提供单条路径前缀映射 `PATH_MAP='/旧前缀=/新前缀'`；直接Python入口可以重复指定 `--path-map`。仅改变查找位置，仍校验原文件哈希，不放宽身份检查。

## 3. 新旧指标怎样读

历史 `methods.*.primary` 原样保留，便于复核；本次提供的 `helper.py` 中该 Fmax 是101阈值 micro-Fmax，AUPRC是101阈值 PR点梯形积分。不能把它们称为逐蛋白Fmax和精确PR面积。

新增 `methods.*.standardized`：

| 字段 | 含义 |
|---|---|
| `standard_protein_fmax` | 逐蛋白平均P/R后计算F，默认阈值步长0.001，严格 `p>t` |
| `standard_protein_fmax_grid_0p01` | 同口径0.01网格，检查阈值分辨率影响 |
| `standard_micro_ap` | 所有protein–GO位置的精确、非插值average precision |
| `standard_micro_pr_auc` | 按所有不同预测分数构造micro PR曲线的精确梯形面积 |
| `standard_micro_fmax_exact` | 所有不同分数组上的最佳micro-F1，含并列分数整体处理 |

分数均为百分数，阈值仍为0–1量纲。Exact micro-Fmax为包含概率0的预测可能报告略低于0的诊断阈值，属于完整排名扫描，不是推荐部署阈值。逐蛋白Fmax排除无gold的蛋白；precision在有预测的合格蛋白上平均，recall在全部有gold蛋白上平均。Micro指标包含所有行和列。

AP与PR-AUC不混用；新增指标不自动执行GO概率传播、筛除无阳性GO或校准，也不宣称完整CAFA评估。E/M/B/G始终同口径；`deltas.G_minus_E/M/B.standardized`保存差值。新实验预先关注逐蛋白Fmax与exact micro PR-AUC，同时报告exact AP和历史指标。指标改变不是性能提升。

## 4. 有限的训练对照

| preset | sampler | max LR → min LR | 日程 | native_dropout |
|---|---|---|---|---:|
| constant | fixed | 3e-4（warmup后恒定） | 对照，horizon=4000 |0.25|
| schedule | fixed |3e-4 → 3e-5|25步warmup+cosine，4000步|0.25|
| lowlr | fixed |1e-4 → 1e-5|同上|0.25|
| aligned | fixed |1e-4 → 1e-5|同上|1.0|
| dynamic | dynamic |1e-4 → 1e-5|同上|1.0|

`aligned`与`lowlr`配对，仅改变监督种子的第一跳原生关系dropout；第二跳原生关系和weak–GO仍保留。`dynamic`与`aligned`配对，仅改变邻居选择方式，不能与schedule同时归因学习率和采样。若schedule明显优于lowlr，需要把对齐实验复制为schedule日程并使用新的CONFIG/WORK_DIR，再做配对比较；不要将预置aligned当作自适应选择。

推荐先重算现有预测，再运行schedule/lowlr；随后测试aligned；dynamic最后考虑。不必默认把五组全部长训。现有v084 fixed可作历史constant参照，但新增可信开发集时应对constant补同口径评估。

```bash
bash scripts/nbs/run_nbs_v085.sh smoke schedule
STEPS=2000 bash scripts/nbs/run_nbs_v085.sh train schedule

bash scripts/nbs/run_nbs_v085.sh smoke lowlr
STEPS=2000 bash scripts/nbs/run_nbs_v085.sh train lowlr
```

默认2卡；单卡设 `NUM_GPUS=1`。smoke默认20步，单独目录，不继承到正式训练。默认checkpoint在600、1200、2000、4000及1000步间隔保存，最终停止步也保存。`pilot`默认600步，但**不能因此宣布充分收敛**。

有依据继续时，在原目录完成事先确定的4000步日程：

```bash
STEPS=4000 RESUME=outputs/latence_nbs_experiments/v085/schedule/latest.pt \
  bash scripts/nbs/run_nbs_v085.sh train schedule
```

`STEPS`是本次停止预算；`full_task.scheduler.horizon_steps=4000`是固定日程终点。2000暂停再续至4000不改变前2000步的LR曲线。不能把horizon改为2000再延长，不能通过RESUME改变LR、模型、loss、采样、角色batch或开发集身份，也不能用v084 checkpoint续训v085。改变策略请新建实验；超过既定horizon需要另行设计日程，当前入口拒绝隐式延长。

新日志记录scheduler、weak/core等效遍历量，以及难gold窗口累积计数和覆盖率。`difficult_gold_core_coverage`在v085改为累积分子/分母；分母为0时为null，不再把空难例批当作覆盖率0。

## 5. 可选：接入真正独立的开发集

现有1024个core holdout曾被Stage1训练过，继续仅作monitor。没有开发集也能运行预设训练预算，但不会生成可信开发集best checkpoint，不输出自动“值得迁移”结论。

若已拥有独立开发集，请准备：

1. 与现有inductive输入同格式的准备目录（文件名仍为 `ind_test_*`），其中蛋白是开发集，具有对应特征、B概率和候选；对应的v081 E/M reference目录。
2. 二值gold `labels.npy`、其protein IDs和原始task GO IDs文件。允许行列重新排列，由IDs显式对齐。
3. 完整Stage1训练蛋白ID列表（包括训练过的各角色）和最终测试ID列表，均一行一个ID。

工具只绑定已有数据，不生成标注、不把旧test自动切分为开发集。准备输入/Stage1 references仍使用项目原有导出流程，并确保其metadata确实描述开发集；可使用仅供该导出流程读取的开发集metadata视图，不能改变真实最终测试数据的身份。

```bash
python scripts/nbs/prepare_nbs_development_v085.py \
  --input-dir /实际路径/dev_inputs \
  --reference-dir /实际路径/dev_inputs/stage1_references_v081 \
  --labels /实际路径/dev_gold.npy \
  --label-protein-ids /实际路径/dev_label_protein_ids.txt \
  --label-go-ids /实际路径/task_go_ids.txt \
  --stage1-training-ids /实际路径/stage1_training_protein_ids.txt \
  --final-test-ids /实际路径/final_test_protein_ids.txt \
  --output /实际路径/dev_manifest_v085.json

DEV_MANIFEST=/实际路径/dev_manifest_v085.json STEPS=2000 \
  bash scripts/nbs/run_nbs_v085.sh train schedule
```

开发集不能与Stage1训练ID、最终test ID或当前Stage2图的任意蛋白ID重叠。该检查依赖提供的训练ID列表完整性，不能替代同源簇/时间划分。标签与E/M仅参与开发评估，绝不进入模型batch。准备目录、标签、行列IDs、E/M来源均绑定哈希。

续训必须提供同一个DEV_MANIFEST（也可固定写入CONFIG的 `full_task.development.manifest`）。输出 `development_history.json`；仅在开发集逐蛋白Fmax不低于B且exact PR-AUC超过B时，按exact PR-AUC保存 `best_development.pt`。这只是预设的开发checkpoint选择，不代表已超过E/M、显著提升或可以自动进入大规模长训。

## 6. 新模型评估

独立测试只作预先约定的报告，不参与checkpoint选择：

```bash
CHECKPOINT=outputs/latence_nbs_experiments/v085/schedule/nbs_step2000.pt \
  bash scripts/nbs/run_nbs_v085.sh evaluate schedule

CHECKPOINT=outputs/latence_nbs_experiments/v085/schedule/nbs_step2000.pt \
  bash scripts/nbs/run_nbs_v085.sh diagnose schedule
```

仍自动获取或复用已有E/M参考。Stage1 checkpoint默认保留正确的 `outputs/weak_exp_train_detr/.../weak_detr_decoder_epoch100.pt` 路径，可用STAGE1_CHECKPOINT覆盖。所有旧入口保持原状；对v084新生成预测仍用 `run_nbs_v084.sh`，之后通过v085 recompute补新指标。

输出重点：新旧指标同时存在的 `nbs_w2s_comparison.json`、`nbs_graph_effects.json`、training_history、development_history（如有）、resolved_config。优先回传这些小文件，无需上传权重和整张预测矩阵。

## 7. 验证边界与后续决策

CPU验证覆盖指标对照、并列分数与端点、历史网格反例、固定日程恢复、真实小图训练/归纳导出、缓存重算与来源检查、开发集隔离。具体结果见 `VALIDATION_v085.md`。本地没有用户完整BP数据或GPU，未执行A100/DDP端到端实验，不保证性能提升。

下一阶段若仍无改善：优先拆分种子candidate/邻居candidate/邻居pseudo的输入贡献、检查难gold监督来源，以及固定top8 PU保护与较大图证据范围的冲突；保留weak–GO和weak–core–GO两条路径。只有简化模型显示可重复的图特异性收益后，再把成功机制迁回原完整模型；不默认继续堆大fanout、训练轮数或dual decoder。
