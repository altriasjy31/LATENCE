# LATENCE NBS v0.8.9.1：loader 兼容修复与大 batch 长训练阶段

本包交付完整 NBS 源码、单卡/双卡配置、启动器、固定口径 Fmax 评估器、验收检查器、安装器和测试。发布版本与 Python 包版本均为 **0.8.9.1**；图模型与数据架构仍是 **0.8.8 hetero_tuned**。不包含数据、训练权重或实验输出。

## 修复内容与立即恢复

原 v0.8.9 的 runner 将新 stage/release 配置交给了只接受 v0.8.8 的 loader，导致 `prepare` 在读取检查点前失败；此前的小图 runner 测试绕过了这条入口。本补丁新增显式 `local_loader_v089.py`，允许 `nbs_v089_long_phase` 和 release 0.8.9.1，并保持原监督来源、binary membership、阈值、gold/weak manifest 和列顺序校验。新 loader SHA 纳入实现 provenance。原 v088 loader/runner/data/model 保持字节一致，兼容已有 epoch 10 检查点。

保留现有 `train_nbs_full_task_v089.py`、`run_nbs_v089.sh` 和两份 `0.8.9_...json` 的文件名，内容版本升级到 0.8.9.1，所以启动入口无需重命名。请安装此修复包后重新初始化；此次报错发生在 data builder 内，未产生训练检查点，无需 `RESUME`。使用新的 `mb12800_from_ep10` 输出目录。来源目录里的 `ep20` 是原实验名，不要求来源检查点已经训练到 20；真正的 epoch 由 `nbs_epoch10.pt` 内容校验。

## 1. 本轮方案及判断边界

本轮执行已确认的主方案：沿用当前异质图 tunedGNN 和监督目标，从一个已完成 weak epoch 的检查点继承模型参数及 BN buffers，开启新的大 batch 长训练阶段，累计训练到 100 weak epochs。不是从零重训，也不是把新运行的 90 epochs 误记成 100 epochs。

当前证据支持继续检验“优化更新不足/训练时间不足”的假设，但不能证明它是唯一瓶颈。已提供的大 batch epoch 10 Fmax 为 56.568183，仍低于基线 57.723149；小 batch epoch 5 为 56.066803。两者都未达到原验收目标。更长训练、双卡吞吐和 v0.8.9.1 Fmax 均未验证；BCE 下降不能替代外部 Fmax。

| 配置 | 每卡 weak/core | 全局 query batch | 梯度累积 | 每个 weak epoch 更新数* |
|---|---:|---:|---:|---:|
| 单卡 mb6400 | 5120 / 1280 | 6400 | 1 | 96 |
| 双卡 mb12800，主方案 | 5120 / 1280 | 12800 | 1 | 48 |

\* 对应当前训练总体；代码从实际 weak population 推导步数，尾批可能较短。epoch 指 weak 样本的一次完整遍历，不代表 core 只访问一次。来源、阶段内和累计的 weak/core exposures 与 optimizer updates 分别记录。

共同优化配置：AdamW；峰值 LR=0.006；weight decay=0.0001；grad clip=5；新阶段 warmup 2 epochs，维持峰值到新阶段 epoch 10，之后 cosine 降至阶段终点，min LR=0.0006。新训练随机种子 1089，原 sampler seed=8088、holdout seed=8080 不变。双卡采用本地 BN 统计并同步 rank 0 的推理 buffers，不宣称使用 SyncBatchNorm。

大 batch 的首要目的为提高吞吐；当前仍是角色归一化 BCE/PU 监督，没有加入对比损失或 batch 内负样本。因此不能照搬 SimCLR 的 batch—负样本收益，也不盲目线性放大 AdamW 学习率。官方 tunedGNN proteins 脚本中的 65536 是评估 batch；其训练 batch 为约训练节点数的十分之一，不能将该数字当作本任务训练设置的直接证据。

研究依据：[tunedGNN proteins 固定源码](https://github.com/LUOyk1999/tunedGNN/blob/23f9604e8b13a9a6d3faa2f691cd844006979153/large_graph/protein.py)、[SimCLR 原论文](https://arxiv.org/abs/2002.05709)。本配置是针对已有 BCE 实验的待验证训练方案，不是 tunedGNN 官方 hyperparameters 的逐项复制。

## 2. 本次代码改变

- 新增 `full_task_phase_v089.py`：明确来源 epoch、阶段步数、累计 epoch、真实样本曝光次数；按累计 epoch 保存固定检查点。
- 新训练器支持 `--init-from` 和严格 `--resume`。初始化校验来源代码、数据、模型、loss、holdout、sampler、种子流及计数；所有 rank 核验同一个来源 SHA 和阶段契约。提高 sampler 的 `max_nodes` 失败保护阈值允许兼容，它不截断采样结果。
- `--init-from` 继承模型及 BN buffers，重置 optimizer、LR schedule、训练 RNG 和阶段历史；保留来源真实累计 exposures/updates。普通 `--resume` 恢复 optimizer、schedule、RNG 和 seed stream，拒绝悄悄改变 batch、world size、horizon、保存周期或来源。
- 保持原图编码、异质关系、序列特征、ontology、训练目标和权重；模型不读取 dense backbone logits，也不融合 expert/modelout 输出。来源模型的 classifier prior 被保留，不重新覆盖已训练 classifier。
- 默认关闭内部 core monitor，保留原 holdout 排除契约。`core/query_candidate/pp` 独立消融继续暂停；完整模型的图输入仍启用。
- `training_runtime.json` 报告近期训练窗口的 seconds/update、小时/epoch 和剩余时间估计。它排除本阶段前 10 updates，不包含加载、准备、保存和评估耗时；rank 0 allocated 显存也不等于整卡显存占用。双卡实际加速必须实测。

原 v0.8.8 trainer/evaluator/launcher/loader/data 文件与本轮提供的最新源文件逐字节一致，保留用于跨版本校验。旧实验源码仅为依赖和兼容材料，启动器不自动执行它们。

## 3. 安装

把 ZIP 解压到项目之外，例如 `/tmp/latence_nbs_v0891`。不要直接将压缩包覆盖到正在训练的目录。

```bash
unzip latence_nbs_v0891.zip -d /tmp
python3 /tmp/latence_nbs_v0891/install_nbs_v089.py \
  --project-root /home/dataset-local/data_local/shaojiangyi/latence-project --dry-run
python3 /tmp/latence_nbs_v0891/install_nbs_v089.py \
  --project-root /home/dataset-local/data_local/shaojiangyi/latence-project
```

安装器先校验整个清单，再写入。相同旧文件不重写；未知的不同运行代码会阻止安装，以保护本地修改。安装器允许原 v0.8.9 包中已知 SHA 的 v089 入口、阶段 helper、配置和说明文件升级，另将 Python 包版本元数据升级到 0.8.9.1；未知本地修改仍拒绝覆盖，原 v088 文件不升级。若报文件不同，需要核对该具体文件与当前运行来源；不要强行覆盖。包内旧配置/旧测试/历史启动器并非都安装，完整源码仍可直接从解压目录审阅和测试。

沿用现有 PyTorch 2.4 环境、数据和缓存即可，不要求重装环境。依赖说明见 `nbs_models/nbs_protein_go/requirements.txt`；测试额外需要 pytest。

## 4. 双卡主方案：先测试，再累计训练到 15

必须使用**已完成**的固定来源检查点。本次示例使用已经完成的 `nbs_epoch10.pt`；原 mb6400 运行可继续向 epoch 20 训练。不能把 20 步 probe 的 `latest.pt` 当作 epoch 10。来源必须处于完整 weak epoch 边界；新训练器根据检查点内容判断，不相信文件名。

以下命令假设来源是 epoch 10；请在可用 GPU 上执行，不会自动停止现有任务。

```bash
cd /home/dataset-local/data_local/shaojiangyi/latence-project
export PYTHON_BIN=/opt/conda/envs/pytorch2.4/bin/python
unset STEPS STOP_STEP STOP_EPOCH INIT_FROM RESUME CHECKPOINT DEV_MANIFEST PRESET
unset ABLATION METRIC_BACKEND AUPRC_MODE
export CUDA_VISIBLE_DEVICES=0,1 CUDA_DEVICE_ORDER=PCI_BUS_ID
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export CONFIG=nbs_models/nbs_protein_go/configs/bp_full_task_v0.8.9_mb12800_ddp_ep100.json
export NUM_GPUS=2 DEVICE=cuda:0
export WORK_DIR=outputs/latence_nbs_experiments/v089/mb12800_from_ep10
export INPUT_DIR=outputs/latence_nbs_eval/bp_shared_full_task_top512_v071_v2
export METADATA_FILE=/home/dataset-assist-0/datafile/latence-dataset/unidata_with_exp_train_pseudo.pkl
export INIT_FROM=outputs/latence_nbs_experiments/v088_mb6400_ep20_single/hetero_tuned/nbs_epoch10.pt
SMOKE_STEPS=20 bash scripts/nbs/run_nbs_v089.sh smoke ddp
```

确认无 OOM/NaN、双卡 rank 都完成、`phase_manifest.json` 来源 epoch=10，以及日志的 batch/world/阶段总步数正确。来源 epoch 10 时，目标累计 100 = **新增 90 epochs / 4320 双卡更新**。先用 `training_runtime.json` 判断速度和预算；20 步不提供 Fmax 验证。

确认后，保留同一 `WORK_DIR` 和配置，续训至累计 epoch 15；上述 20 步计入正式训练，不丢弃：

```bash
unset INIT_FROM
export RESUME="$WORK_DIR/latest.pt"
STOP_EPOCH=15 bash scripts/nbs/run_nbs_v089.sh train ddp
```

本阶段的 smoke 是一次新的运行；已有 `latest.pt/phase_manifest.json/training_history.json` 时不得重新 smoke。断点续训使用 `train` 与 `RESUME`，不要再次 `INIT_FROM`。同目录文件锁防止两个启动器同时训练。

## 5. 固定 epoch 外部 Fmax 与继续训练

默认每 5 个**累计** epochs 保存 `nbs_epoch15.pt`、`nbs_epoch20.pt`……`nbs_epoch100.pt`；短暂 `STOP_STEP` 也保存 `latest.pt` 以便 resume。只评估固定 epoch 文件，避免运行中变化的 latest。

```bash
unset INIT_FROM RESUME STOP_STEP STOP_EPOCH
CHECKPOINT="$WORK_DIR/nbs_epoch15.pt" bash scripts/nbs/run_nbs_v089.sh evaluate ddp
"$PYTHON_BIN" scripts/nbs/check_nbs_v089_acceptance.py \
  --baseline-comparison BASELINE_CONTRACT_v088.json \
  --comparison "$WORK_DIR/eval_step240/full/metrics/nbs_w2s_comparison.json" \
  --output "$WORK_DIR/acceptance_epoch15.json"
```

评估 `ddp` 配置仍为单进程 `cuda:0` 推理，无需两卡评估。只计算完整 G/B 蛋白 Fmax，不运行消融、E/M 比较、输出融合、AP/PR 排名或 ontology 概率传播。基线文件中的历史 E/M 只是固定评估契约，不会参与新模型输出。

协议保持 **1800 proteins × 21312 BP GO**、原蛋白/GO 顺序和 metadata、严格 `>`、阈值间隔 0.001。检查器核对来源与指标定义、输出、输入、检查点 hash，报告 `new_Fmax`、`delta_Fmax` 和独立的 `baseline_exceeded`。

- 超过基线小目标：Fmax > **57.72314936447243**。
- 原完整验收：gain ≥ 3 且 Fmax ≥ 60，因此有效门槛为 **60.72314936447243**。
- 仅超过基线而不足 +3，依然写“未达标”；没有真实结果写“未验证”。

检查器退出码为：0=达标，1=未达标，2=未验证。**不要用 `&&` 把检查器与续训串联**，以免正常的“未达标”退出码阻断训练。它也不会覆盖已有验收 JSON；重复评估请换输出文件名。

若按计划继续到累计 epoch 100：

```bash
unset INIT_FROM STOP_STEP STOP_EPOCH
RESUME="$WORK_DIR/latest.pt" STOP_EPOCH=100 bash scripts/nbs/run_nbs_v089.sh train ddp
```

也可以分段 STOP_EPOCH=30/40/50；它只设暂停点，不改 100 epochs 的固定调度终点。不得在 resume 时延长 horizon 或改变配置。评估时再次清除训练变量，显式传入新的固定检查点。

来源 epoch 10、双卡时，`phase_step=(累计 epoch−10)×48`：epoch 15/20/50/100 对应 240/480/1920/4320。因此评估目录依次为 `eval_step240/480/1920/4320`。换来源 epoch 20 或单卡后应按实际 manifest 的 `steps_per_epoch` 计算；不要照搬 240。

如果希望每 10 epochs 保存一次，请在**新阶段开始前**复制配置并将 `full_task.checkpoint_every_epochs` 从 5 改成 10，再设置 CONFIG。新阶段开始后的 resume 不允许改变该字段。最终 epoch 100 始终保存。

## 6. 单卡替代

在开始新阶段前改用以下设置，其余初始化/续训/评估流程相同，但所有启动命令的 preset 改为 `single`：

```bash
export CUDA_VISIBLE_DEVICES=1 NUM_GPUS=1 DEVICE=cuda:0
export CONFIG=nbs_models/nbs_protein_go/configs/bp_full_task_v0.8.9_mb6400_single_ep100.json
export WORK_DIR=outputs/latence_nbs_experiments/v089/mb6400_from_ep10
```

单卡来源 epoch 10 到累计 100 为新增 90 epochs / 8640 updates；累计 epoch 15 的评估目录为 `eval_step480/full/metrics`。不能从双卡阶段直接 strict-resume 成单卡；改变 world size 需要在已完成 epoch 边界，用固定检查点 `INIT_FROM` 开启另一新目录，并明确记录新的优化阶段。

## 7. 验证与资源边界

本地已验证小规模 CPU 图上的跨版本 weights-only 初始化、classifier/BN 保留、优化器重置、精确 resume、dropout/RNG/尾批/计数、阶段链式继承、来源契约、完整轴导出、缓存与验收拒绝错误 provenance，以及启动器/安装器。双卡损失梯度全局归一化的数学检查通过。

实际双进程 Gloo 测试因当前环境创建通信设备时报 `Operation not permitted` 而跳过；真实 NCCL/GPU/DDP、A100 显存、双卡速度和外部 Fmax **未验证**。服务器 20 步测试是必要下一步。保存耗时、磁盘/主机内存、加载和评估均未在本地真实总体上测量；两个 GPU 的 epoch 时间不保证减半。固定检查点将保存完整训练状态，请检查可用磁盘，不要以当前 11.59GB allocated 推断全部资源预算。

本次补充了读取两份真实配置、实际 mmap loader 和四个 CLI 阶段共同入口的 64 项测试，直接覆盖此次漏检。精确测试结果与命令见 `VALIDATION_NBS_v089.json`。可在解压源码目录执行双进程 CPU 检查（不使用生产数据）：

```bash
cd /tmp/latence_nbs_v0891
PYTHONPATH=.:nbs_models/nbs_protein_go:nbs_models/nbs_protein_go/tests \
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
/opt/conda/envs/pytorch2.4/bin/python -m pytest -q \
  nbs_models/nbs_protein_go/tests/test_full_task_ddp_v089.py
```

若后续固定 epoch Fmax 持续停滞或下降，应据此复核优化不足这一核心假设；本包没有自动叠加同类补丁、修改评测任务或宣称 100 epochs 必然超过基线。
