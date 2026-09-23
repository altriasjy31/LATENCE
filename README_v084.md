# NBS v0.8.4：多源邻域与两层关系 SAGE

本次版本统一命名 v084。在已安装 v083 的 LATENCE 项目上更新；代码包不是完整项目，也不包含数据、权重或第一阶段模型。

## 这一版解决什么

v083 的种子主要读取固定 cosine top-8 core，原生蛋白关系只补充这些 core 的上下文。v084 从种子直接建立多源邻域，先注入功能证据，再进行两层关系 SAGE。提供 `fixed` 与 `dynamic` 两组，其模型、监督、loss、邻居池、批次规模和缺边增强设置相同，只改变训练时的邻居选择。

本版先验证图构造和采样，不同时加入困难正例加权、新的 PU 保护、双塔、dense modelout 蒸馏或 ego-net Transformer。原主表指标保留；此前提出的 CAFA 聚合维度/AUPRC 积分疑问，尚需实际 Stage1 helper.py 才能核查，不能据此声称已修复。

## 安装

将 `latence_nbs_v084.zip` 解压，与 `latence-project` 放在同一级。进入项目根目录：

```bash
cd /home/dataset-local/data_local/shaojiangyi/latence-project
python ../latence_nbs_v084/install_v084.py --project-root .
```

也可把解压目录放在其他位置，只需给出 `install_v084.py` 的真实路径。不要把整个更新目录复制成项目中的子模块。

安装器校验包内文件，备份将修改的文件，再安装新模块；重复安装可识别已有版本。它仅给旧 `local_loader.py` 的二值监督许可增加 `nbs_v084_`，不要求旧文件全文完全一致。已有 v083 运行代码的其他本地改动不会被整文件覆盖。备份位置打印在终端。

v084 模型结构不同，必须新建训练，不能从 v081/v082/v083 checkpoint 续训。v084 内部续训保持同组、同配置和同设备数量。旧版本对源码有严格哈希检查；如还要续训旧 v083，请使用安装前备份中的旧 loader 或单独保留旧项目副本。

## 最短运行流程

第一阶段独立集输入、Expert 和 Modelout 参考结果仍复用原流程；无需重新训练第一阶段。

```bash
# 先验证 Stage1 参考输入，避免训练结束后才发现路径问题。
bash scripts/nbs/run_nbs_v084.sh references fixed

# 只需为两组共同准备一次 v084 邻居池。
bash scripts/nbs/run_nbs_v084.sh prepare fixed

# 每组 smoke 使用独立目录，pilot 从新模型开始。
for variant in fixed dynamic; do
  bash scripts/nbs/run_nbs_v084.sh smoke "$variant"
  bash scripts/nbs/run_nbs_v084.sh pilot "$variant"
  bash scripts/nbs/run_nbs_v084.sh evaluate "$variant"
done
```

默认：2 GPUs；smoke 20 步；pilot 600 步；train 4000 步。单卡运行在命令前加 `NUM_GPUS=1`。`pilot/train` 只负责训练，`evaluate` 单独运行，不会因参考结果错误而丢失已训练 checkpoint。

首次准备会扫描原生边并生成较大的 cosine 候选池；后续 fixed/dynamic 共用缓存。准备阶段耗时不等于每轮训练耗时。检查终端 relation coverage 与 cosine 进度；实际耗时、CPU 内存和 GPU 显存需以您的数据为准。

### 已修正的 Stage1 checkpoint 路径

默认值：

```text
/home/dataset-assist-0/datafile/latence-dataset/outputs/weak_exp_train_detr/bp_weak_detr_v3_expert_prob_warmstart340_to400/weak_detr_decoder_epoch100.pt
```

可显式覆盖：

```bash
export STAGE1_CHECKPOINT=/实际路径/weak_detr_decoder_epoch100.pt
```

保留原有 checkpoint SHA256 校验，允许文件搬迁但不混用其他权重。Expert 和 Modelout 自动生成逻辑仍来自 Stage1 评估；它们用于 E/M/B/G 比较，不作为 v084 的 dense 监督或输出混合项。

## 对照设计与图路径

| 项目 | fixed | dynamic |
|---|---|---|
| 首跳总预算 | 32 个不同蛋白 | 同左 |
| 首跳邻居的后续预算 | 每蛋白 4 个 | 同左 |
| 原生邻居池 | 每关系、每接收节点最多 64 个 | 同一缓存 |
| cosine 候选池 | 64 个训练 core | 同一缓存 |
| 训练选择 | 关系轮流取高排名邻居 | 前半稳定，后半按排名温和加权无放回抽取 |
| 推理选择 | 确定性选择 | 同样确定性选择 |
| 种子原生关系缺失增强 | 25% 概率只用检索补充 | 相同步骤/种子规则 |
| 输出与 loss | 全部 21,312 个 BP GO；全部已知正例 | 完全相同 |

32/4/64 是首轮可控预算，不是经过性能搜索的最优值。`fixed` 的固定是邻居排名选择；两组都存在相同的训练缺边增强，以覆盖独立蛋白没有原生关系的场景。新增邻居比例日志比较的是当前选择与同一池的固定选择，**不是跨 epoch 累计覆盖率**。

五类消息明确按“证据来源→接收蛋白”表示：`ppi`、`similar_to`、`weak_to_core`、显式反向 `core_to_weak`、`cosine`。原生关系方向按存储的实际边处理，不把所有边无条件双向化。

GO 信息分三路进入蛋白：候选 GO（B/selector 属性）、其他 core 的 gold 注释、其他 weak 的二值 pseudo-GO。随后两层蛋白关系 SAGE 覆盖 weak–core–GO 与 weak–core–core–GO；直接 weak–GO 证据继续保留，GO–GO 使用已有 ontology encoder。

模型仍是单一全 GO matcher。种子使用两层后的状态，直接 core query 读取第一层状态，防止查询偷偷超出两跳采样范围或随其他 batch 的邻域改变。没有新增 classification 分支。

当前监督批次全部种子的 gold/pseudo 标签在聚合前移除；holdout 标签全局不可见。邻居 weak 的二值标签是图输入证据，目标 weak 的二值标签只用于监督。模型不读取 dense M/E 概率。两组保留旧 cosine-only structural support 来计算相同的 PU loss，避免邻域选择同时暗中改变优化目标。

独立测试种子目前没有原生 PP 身份，使用同规则的 cosine 候选池连接训练 core，再读取已知邻居的原生关系。训练的原生关系缺失增强覆盖这种情况，但并不能证明训练/测试分布完全一致。

## 查看结果与是否延长

默认目录：

```text
outputs/latence_nbs_experiments/v084/fixed/
outputs/latence_nbs_experiments/v084/dynamic/
```

优先回传两组：

- `training_history.json`
- `resolved_config.json`
- `eval_step600/full/metrics/nbs_w2s_comparison.json`

history 新增实际采样节点数、每关系边数、检索占比、固定选择之外的新增邻居比例、经过缺边增强后的原生关系不可用比例（含人为 dropout），以及 B<0.5 的 gold 正例图支持覆盖。该困难正例统计是诊断，不改变 loss。图支持覆盖统计当前针对精确 core–GO 通道，不应称为全部 weak/PP 证据的召回。

先比较 fixed/dynamic 的 E/M/B/G 全任务 Fmax、主表 AUPRC、精确 micro-AP、Top-k 与时间。训练 loss 下降或梯度非零不能代替性能。最终目标仍是同时超过 Expert 的主表 Fmax/AUPRC，而不是只超过 B。600 步是短探针，不构成统计显著性证明。

需要确认路径贡献时，针对选定的固定 checkpoint：

```bash
CHECKPOINT=outputs/latence_nbs_experiments/v084/dynamic/nbs_step600.pt   bash scripts/nbs/run_nbs_v084.sh diagnose dynamic
```

输出 `eval_step600/nbs_graph_effects.json`。`weak_off` 关闭候选与二值 pseudo 功能证据，`core_off` 关闭 gold 和直接 core 读出（仍保留蛋白关系传播），`pp_off` 关闭两层 SAGE，`graph_off` 关闭以上三路，`go_shuffle` 打乱 protein–GO 对应关系。关闭实验衡量推理依赖，不等于独立训练的因果贡献。

如短训练结果值得继续，先固定延长至 2000 步：

```bash
STEPS=2000 RESUME=outputs/latence_nbs_experiments/v084/dynamic/latest.pt   bash scripts/nbs/run_nbs_v084.sh train dynamic
bash scripts/nbs/run_nbs_v084.sh evaluate dynamic
```

fixed 组将命令中的 dynamic 换成 fixed。断点恢复包含模型、优化器、蛋白队列及每个 rank 的 Torch RNG；邻域随机性由 seed、step、rank、蛋白 ID 确定。不能把 fixed 权重作为 dynamic 的等价续训。

## 可修改的设置

两份配置位于：

```text
nbs_models/nbs_protein_go/configs/bp_full_task_v0.8.4_fixed.json
nbs_models/nbs_protein_go/configs/bp_full_task_v0.8.4_dynamic.json
```

预算与采样设置在 `full_task.sampler`。修改时同时更新两组共同设置，只保留 `mode` 差异。修改模型、loss、邻居池来源或采样策略后使用新输出目录；缓存身份不一致时使用新的 `sampler.cache_dir`。

常用入口环境变量：`STEPS`、`NUM_GPUS`、`DEVICE`、`CONFIG`、`WORK_ROOT`、`WORK_DIR`、`INPUT_DIR`、`METADATA_FILE`、`CHECKPOINT`、`RESUME`、`STAGE1_CHECKPOINT`。自定义训练目录时，评估使用同一设置。路径可以包含空格。

## 验证范围

随包提供 CPU 测试，覆盖实际小图传播学习、图方向、标签隔离、全 GO 输出、固定/动态对照、采样复现、跨轮种子唯一性、断点恢复、预测分组不变性和评估来源绑定。具体最终检查结果见 `VALIDATION_v084.md`。

本地没有您的完整 BP 数据与 GPU 环境，未声称在 A100/DDP 上完成验证，也不保证真实性能提升。现有 core holdout 曾被第一阶段见过，只作运行监控；可靠模型选择应使用独立于第一阶段训练的有标注开发集。
