# LATENCE NBS v093：统一 weak 标签、rare 保护与未见 GO 预测

本版把功能接入同一条可运行流程：重建 weak 数据 → 从头训练 Stage2 → 评估整体/低频/未见 GO → 按 GO ID 请求预测。原 Stage1 表征、蛋白图、BoxSquaredEL 几何及可兼容的蛋白邻居缓存继续使用。默认单卡、3 层、hidden=320、全局 weak/core=10240/2560、累积2次，总学习率 horizon 从起点固定为200 epochs，每5 epochs保存。

## 安装

在解压后的包目录运行：

```bash
/opt/conda/envs/pytorch2.4/bin/python install_v093_unified.py \
  --project-root /home/dataset-local/data_local/shaojiangyi/latence-project --apply
```

仅新增文件；不同内容的同名本地文件会被拒绝覆盖。本包需要已安装的 v090、P0 fast sampler 和 v092 源文件，安装器会检查依赖。它不会停止当前训练。

## 主运行流程

以下命令都在项目目录运行，不需要配置环境变量：

```bash
cd /home/dataset-local/data_local/shaojiangyi/latence-project

# 在现有 PyTorch 环境运行小型 CPU 合约/模型/损失/训练续训测试。
/opt/conda/envs/pytorch2.4/bin/python scripts/nbs/run_nbs_v093.py check

# CPU：用已有 core Backbone 概率拟合分组阈值，再重建 weak 标签与候选。
/opt/conda/envs/pytorch2.4/bin/python scripts/nbs/run_nbs_v093.py build --calibrate-backbone

# 真实数据3步检查，使用独立 smoke 目录。
/opt/conda/envs/pytorch2.4/bin/python scripts/nbs/run_nbs_v093.py smoke --gpu 0

# 新 Stage2 主实验；所有功能从第1步起同时启用。
/opt/conda/envs/pytorch2.4/bin/python scripts/nbs/run_nbs_v093.py train --gpu 0 --stop-epoch 20
```

主实验目录为 `outputs/latence_nbs_experiments/v093/unified_single12800`；新数据目录为 `outputs/latence_nbs_experiments/v093/unified_dataset`。训练上限与调度已固定为200，首次停在20不会重新定义调度；继续到200：

```bash
python scripts/nbs/run_nbs_v093.py resume --gpu 0 --stop-epoch 200
```

可以从首次就使用 `train --stop-epoch 200`。本地没有真实训练数据和GPU，不提供未经测量的训练耗时或性能保证。新 weak 证据规模、两条输出头及辅助损失可能改变速度和显存；先看 smoke 的实际日志。

在另一张卡上评估已完成的固定 epoch：

```bash
python scripts/nbs/run_nbs_v093.py evaluate --gpu 1 --epoch 5
```

结果为 `eval_step<实际step>/full/metrics_v093/summary.json`。step 不再假定永远是48×epoch，而由新训练 weak 有效行数决定。报告保留原始 Backbone 对照、加载器裁剪后的 Backbone 对照、NBS_final；包含整体与分频 micro/macro AP、protein Fmax，以及整体最优阈值和 Backbone 阈值下的分组 P/R。seen/unseen 分开报告，无阳性组返回 `null`，不伪造 zero-shot 指标。

训练与评估使用不同锁，评估与已修复的 v092 共用缓存构建锁，支持单卡训练时在另一张卡评估。所有输入仍应使用现有 full-task top512 推理缓存；从已截断16个候选的缓存无法恢复分频候选。

## Weak 标签如何重建

训练 core 金标准计数排除原1024个蛋白 holdout，保留全部21312个任务列和原始顺序。默认 count≤999 使用 Backbone，count≥1000 使用原 Stage1 modelout。modelout 是 decoder 输出，不是原始 expert 分数。每个新成员保存原始教师分数与来源编码；core 金标准不替换。

`build --calibrate-backbone` 使用原训练 core 的 Backbone 概率，按 1–5、6–9、10–99、100–999 分组，在0.001网格上以严格 `>` 最大化 pooled micro-F1。高频 modelout 保留原0.5规则。其校准范围明确为 **Stage1 已见训练 core，训练内拟合**，不是独立验证，不保证在 weak 或测试集最优；从未读取独立测试标签。校准文件记录每组阈值、TP、P/R/F1、数据身份与 seen mask。

若已有独立校准阈值，使用：

```bash
python scripts/nbs/run_nbs_v093.py build --thresholds-json my_thresholds.json \
  --dataset-dir outputs/latence_nbs_experiments/v093/independently_calibrated_dataset
```

支持的 JSON：

```json
{
  "thresholds": {"1-5": 0.5, "6-9": 0.5, "10-99": 0.5, "100-999": 0.5, "1000+": 0.5},
  "independently_calibrated": false,
  "provenance": {"scope": "replace with actual calibration source"}
}
```

例中数值只是格式示例。裸 `build` 则使用固定0.5，作为来源重构的简单对照，明确不称为校准。更改策略使用新 dataset-dir 和新 work-dir，禁止把旧模型直接 resume 到新数据。

候选总预算保持16：最多4个频次1–9、4个频次10–999，再由原全局排名补足；去重并在选择前过滤未见标签。该配额是待验证的设计，不代表加入的候选都是真实功能。新训练候选直接读独立缓存，外部查询采用同一选择函数。保留 `--candidate-mode global` 对照入口。

数据 manifest 中的 `frequency_strata` 给出旧/新 weak 正例数与覆盖列数。不要仅凭总标签数增加判定改善；也不要按每GO配额强造伪正例。

## Rare 保护与训练目标

已见 GO：`loaded_backbone_logits + shared_residual`。残差评分器全部为共享参数，并且全零初始化，因此训练开始时已见列的 logits 与读取器提供的 Backbone logits 完全相等。旧读取器默认将概率裁剪到约 `[1e-4,1-1e-4]`，不能宣称原始缓存中的0/1端点逐值不变。

未见 GO：使用单独的共享通用评分头，完全不读取其 Backbone 输出列。通用头在已见标签上训练，供未见任务列、ontology-only GO 和合法新节点使用。模型没有逐GO可训练分类器行。

默认目标为角色均值归一化的主二值 BCE，加0.25共享头 BCE、0.1低频 Backbone KL、0.1已选中 weak 正例置信度项。连续概率由明确的辅助项读取，不再只是写文件。训练 history 同时记录这些分量。weak 非成员仍沿用二值目标；其0不等于生物学功能已证实不存在。本版没有同时加入新的PU损失。

这是将保护机制与新数据一起落地的主实验，不是已完成的因果消融。应以整体性能和 rare 保留共同判断；1–5、1–9、10–99 micro/macro AP 以及固定工作点 P/R 必须单独检查。

## Zero 功能的具体范围

训练 core 中没有正例的 canonical GO 组，自动从所有监督项的正、负部分排除；候选、gold/pseudo 消息也过滤这些组。别名作为整体处理。支持额外指定 Stage2 canonical label holdout，从数据、阈值拟合、监督和输入证据中一起剔除：

```bash
python scripts/nbs/run_nbs_v093.py build --calibrate-backbone --label-holdout-fraction 0.1 \
  --dataset-dir outputs/latence_nbs_experiments/v093/zero_holdout_dataset

python scripts/nbs/run_nbs_v093.py train --gpu 0 --stop-epoch 20 \
  --dataset-dir outputs/latence_nbs_experiments/v093/zero_holdout_dataset \
  --work-dir outputs/latence_nbs_experiments/v093/zero_holdout_single12800
```

该入口用于有测试阳性的 Stage2 未见标签评估；它是可选的独立数据版本，不要求主实验先完成这组对照。原 Stage1 特征、教师与候选排序可能见过这些标签，故不能声称严格端到端 zero-shot 已验证。holdout 的阈值拟合只使用 seen 列。

按目标 GO 身份预测：

```bash
python scripts/nbs/run_nbs_v093.py predict-go --gpu 1 --epoch 5 \
  --go-ids-file requested_go_v093.example.txt
```

每行一个 `GO:xxxxxxx`，目标顺序保留；输出概率、蛋白ID、GO映射、seen/unseen 状态和来源。示例文件是已见BP根功能的接口检查，请换成所需BP目标。GO在完整ontology中但不在任务分类列时可以直接评分，不需要增加分类器权重。

如果GO连训练时完整ontology都未收录，必须提供**同坐标空间的新几何与关系**，通过 `--ontology-file updated_ontology.pt` 传入：

```python
{
    "ontology": {
        "center": ...,                 # float32 [updated ontology rows, box dimension]
        "offset": ...,                 # 原始offset，不是log-offset
        "task_to_ontology": ...,       # 原任务列映射到更新后的行
        "edges": {"is_a": ..., ...}    # 与训练时相同的关系名称，整数[2,E]
    },
    "old_ontology_rows": ...,           # 每个旧ontology行在新表中的行号
    "registry_records": [...]           # 完整更新GO注册表，保留所有旧GO/alias身份
}
```

程序严格验证所有旧几何坐标与GO身份锚定，保留训练时 raw geometry mean/scale，仅重算GO编码，不添加可训练分类器行。更新关系可能改变已有GO编码。它**不会仅凭一个新GO ID自动生成 BoxSquaredEL 特征**；原BoxSquaredEL的逐节点嵌入表也不具备这一现成能力。新节点几何的拟合/归纳生成是输入准备问题，本包不将随意放置的box伪装成训练好的BoxSquaredEL输出。

## 验证状态

参见 `VALIDATION_V093_UNIFIED.txt`。当前开发环境能执行 NumPy/CLI/指标与构建测试，但没有PyTorch，下载也不可用。模型、loss、真实优化器/续训/导出探针已写成测试，未执行者明确标为skip。请先在现有项目环境运行 `check`；它要求真实PyTorch并执行这些小型CPU检查，再运行真实数据 smoke。
