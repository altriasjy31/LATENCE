# LATENCE NBS v0.8.7：直接图预测、真实 epoch 与证据来源控制

本包是安装于现有 **v085 + v086 工程**上的增量更新。只新增带 v087 / v0.8.7 的文件；旧版脚本、模型和实验目录保留。新模型需要从头训练，不能接续 v086 checkpoint。

## 1. 为什么这一轮改变训练目标的实现方式

上传的 legacy step 4000 已呈现平台：标准 protein Fmax 从 step 2000 的 **57.976971** 变成 **57.930862**；exact micro PR-AUC 从 **52.080257** 变成 **52.072281**。同一批参考的 B 为 **57.723149 / 50.621882**，E 为 **71.690220 / 67.808017**，M 为 **71.644822 / 68.811595**。这里所有数值采用 0–100 标度。

step 4000 的 `full − weak_off` 为 **−0.358764 / −0.230653**，`full − pp_off` 为 **−0.075717 / −0.036799**；core 路径的对应差值为 **+0.372149 / +1.282934**。这些干预结果提示现有来源使用方式值得重做，不能单凭它们断言删掉某一路训练必然更好。

因此这版同时检验两个问题：给图模型完整的多轮优化预算后，残差形式是否仍受限；取消 B 的直接输出通道后，图模型能否学出更强的绝对预测。第三组再独立检验来源遮罩，避免把它和输出形式混为一个原因。

## 2. 三组预置配置

| preset | 最终输出 | B anchor | 三类 source dropout | 用途 |
|---|---|---:|---:|---|
| `residual_epoch` | `B + delta` | 0.05 | 0 | 相同 epoch 预算下的残差对照 |
| `direct` | 图模型直接输出全部 GO logits | 0 | 0 | 本轮主要结构改动 |
| `direct_source_dropout` | 图模型直接输出全部 GO logits | 0 | 每类 0.20 | 检验对弱证据来源的依赖 |

三类来源分别是 query candidate、neighbor candidate、neighbor weak pseudo。遮罩按本地蛋白节点和来源独立产生，丢弃该节点这一来源的全部关联；query candidate 的同一个遮罩同时作用于 encoder 注入和 decoder 入口。训练随机性由 seed/step/rank/source 确定，评估时停用随机遮罩。原有候选级 dropout=0.15 仍在三组中保持一致。

`direct` 移除了最终 `B + delta`，同时删除 decoder 内的 `tanh(B/4)` 和 `vote − sigmoid(B)` 两项数值输入。GO 逐列 bias 初始为可学习的 −4，最后读出层采用小随机初始化，保证图分支首步获得梯度。−4 是初始化选择，不是从测试标签估计的先验。

**direct 仍使用 B 生成的稀疏候选；默认 PU mining 仍为 `max(current, B)`。** 因而它并非完全不依赖 B。配置支持 `loss.mining_source=current`，但本轮三组都固定为 `current_base`，控制 PU 选择规则。E/M 密集概率不进入图模型前向，也不作为 soft target；M 只贡献既有的二元 pseudo membership。旧的 own-feature graph auxiliary 仍关闭（`graph_weight=0`）。

三组统一采用 legacy encoder、两层关系 SAGE、32/4 两跳预算、fixed sampler、相同 weak/core 二元监督和结构性 PU 保护。v086 的 preln/DropEdge 支持仍可用，但不在本轮三组中同时改变。改变输出形式与移除 B anchor 共同构成 direct 方案，不能把组间差值解释成仅一个标量开关的效应。

## 3. epoch 与 tunedGNN 的对齐范围

这次对齐的是完整监督集合遍历、多轮优化和明确的学习率日程；没有声称复现 tunedGNN 的数据、架构或 protein benchmark 超参数。

一个 epoch 定义为 weak 训练集合恰好遍历一次。每轮使用新的随机排列；最后一批不补齐、不丢弃、不重复。core 采用独立的连续循环，跨 weak epoch 不重置，同一全局 batch 内无重复 core。训练标签在**全部 rank 的本次监督种子**范围内隐藏，覆盖 sampled gold/pseudo、decoder anchor 和固定 PU 支持图；候选边和 PP 拓扑保留。

默认每卡 weak=64、core=16，2 卡时：

| 项目 | 数量 |
|---|---:|
| weak population（当前数据） | 489,222 |
| global weak batch | 128 |
| global core batch | 32 |
| 每 weak epoch 更新数 | `ceil(489222 / 128) = 3823` |
| epoch 1 / 3 / 5 终点 | 3823 / 11469 / 19115 steps |
| 5 epoch weak 曝光 | 2,446,110 |
| 5 epoch core 曝光 | 611,680（59476 个 core 约 10.28 次） |

最后一批只有 6 个 weak；某 rank 可以没有 weak，但仍有 core。loss 按全局 weak/core 数归一化，DDP 平均梯度后保持指定的角色权重，不让尾批比例悄悄改变目标。

学习率：前 **0.1 epoch** warmup 到 **1e-4**，保持至 **epoch 3**，随后 cosine 降至 **epoch 5 的 1e-5**。日程按更新位置折算 epoch；每轮边界与实际 weak 遍历一致。5 epoch horizon 从建实验时固定；`STOP_STEP` / `STOP_EPOCH` 只控制这次执行停在哪里，不重写日程。日志同时给出 epoch、该轮 step、累计 step、实际 weak/core 曝光量。

普通比例诊断是 rank/step 均值，在不等尾批时不是严格总体均值；loss 是正确的全局角色归一。来源遮罩还额外记录全局节点计数和 `global_removed_fraction`。

## 4. 安装

先把 ZIP 解压到工程外的临时目录，在已经部署 v085 和 v086 的项目上检查后安装：

```bash
unzip latence_nbs_v087.zip -d /tmp/nbs_v087_release
python /tmp/nbs_v087_release/latence_nbs_v087/install_v087.py \
  --project-root /path/to/latence-project --check-only
python /tmp/nbs_v087_release/latence_nbs_v087/install_v087.py \
  --project-root /path/to/latence-project
cd /path/to/latence-project
```

安装器校验已有依赖和新增文件的 SHA256，拒绝覆盖内容不同的文件；重复安装相同文件可安全跳过。若报告旧依赖不同，请先核对本地版本，不能用强制覆盖解决。数据路径沿用现有工程；邻居缓存沿用 v084，首次 `prepare` 会检查其契约。

新增核心文件：

- `nbs_pg/full_task_model_v087.py`：绝对预测与三来源遮罩。
- `nbs_pg/full_task_data_v087.py`：全局训练 batch 标签排除。
- `nbs_pg/full_task_epochs_v087.py`、`full_task_schedule_v087.py`：真实 epoch、独立 core 循环、固定日程恢复。
- `nbs_pg/full_task_loss_v087.py`：全局角色归一、分角色损失诊断。
- `nbs_pg/local_loader_v087.py`：明确的 v087 数据与监督契约。
- `scripts/nbs/train_nbs_full_task_v087.py`、`eval_nbs_full_task_v087.py`、`run_nbs_v087.sh`。
- 三份 `configs/bp_full_task_v0.8.7_*.json` 以及新增测试。

以上 `nbs_pg/` 与 `configs/` 都位于 `nbs_models/nbs_protein_go/` 下。

## 5. 建议执行顺序

下面在项目根目录执行；默认用 2 卡，可用 `CUDA_VISIBLE_DEVICES` 指定卡。不要在各组之间改变卡数或 batch。若有已经验证且与训练/独立测试排除的开发集契约，在第一条 smoke/正式训练命令前设置 `DEV_MANIFEST=/absolute/path/development_manifest.json`，随后所有恢复保持同一份契约。没有开发集也能执行固定预算训练。

### 5.1 三组各做 20 步 smoke

```bash
for preset in residual_epoch direct direct_source_dropout; do
  NUM_GPUS=2 bash scripts/nbs/run_nbs_v087.sh smoke "$preset"
done
```

smoke 使用 `outputs/latence_nbs_experiments/v087/smoke_<preset>`，formal 使用 `.../v087/<preset>`；不要把 smoke checkpoint 作为正式起点。检查 loss/梯度有限、全部 GO 列参与、`contrib_graph_aux=0`、direct 的 `contrib_anchor=0`。source-dropout 组的来源移除率在足够样本下应接近 0.20；另两组为零。smoke 只验证运行，direct 初始预测不继承 B，不能凭 20 步性能判断方案。

### 5.2 三组都从头跑到 epoch 1

```bash
for preset in residual_epoch direct direct_source_dropout; do
  NUM_GPUS=2 STOP_EPOCH=1 bash scripts/nbs/run_nbs_v087.sh train "$preset"
done
```

自动先检查/准备图采样缓存，然后训练。每组保留 `resolved_config.json`、`training_history.json`、`validation_history.json`、`latest.pt`、`nbs_step3823.pt` 与 `nbs_epoch1.pt`。这些 step 数基于当前数据和 2 卡配置，实际以 resolved config 为准。

### 5.3 在原目录恢复到 epoch 3，再到 epoch 5

```bash
for endpoint in 3 5; do
  for preset in residual_epoch direct direct_source_dropout; do
    NUM_GPUS=2 STOP_EPOCH="$endpoint" \
      RESUME="outputs/latence_nbs_experiments/v087/$preset/latest.pt" \
      bash scripts/nbs/run_nbs_v087.sh train "$preset"
  done
done
```

也可以不设置 STOP 参数，一次跑完 5 epoch。提前中断后，仍以上一已保存 checkpoint 恢复；未保存的更新会重新执行。恢复要求代码、数据、种子、world size、模型/loss、日程一致。`EPOCHS=5` 是配置一致性断言；不能用 `EPOCHS=1` 做短跑。旧 `STEPS` 参数被拒绝。日程之外的续训需另立实验，本版不允许恢复时延长 horizon。

每完成一轮默认保存 `nbs_epochN.pt`。`best_core_holdout.pt` 仅是 Stage2 core 留出监视器，因为 Stage1 曾见过这些蛋白，不能据此宣称超过 E/M。只有绑定有效外部开发集时才可能产生 `best_development.pt`。

## 6. 导出、评估与消融

训练命令不自动运行独立测试。建议先完成预先固定的 1/3/5 epoch 预算；用开发集选择方案。若在现有独立测试上多次比较，这些结果属于探索性对照，应保留 checkpoint 选择记录。

### 6.1 只导出概率

```bash
CHECKPOINT=outputs/latence_nbs_experiments/v087/direct/nbs_epoch5.pt \
  bash scripts/nbs/run_nbs_v087.sh export direct
```

`export` 不需要 E/M 参考概率或 metadata；它只使用既有推理输入和 checkpoint。输出仍用实际 optimizer step 目录，例如 `direct/eval_step19115/full/`；epoch 是训练数据遍历单位，保留 step 目录便于和旧评估工具对接。

### 6.2 比较三个固定终点

```bash
for preset in residual_epoch direct direct_source_dropout; do
  CHECKPOINT_EPOCHS="1 3 5" \
    bash scripts/nbs/run_nbs_v087.sh series "$preset"
done
```

默认读取既有 `INPUT_DIR`、`METADATA_FILE` 与 Stage1 reference 设置，必要时自动准备 E/M 参考并先退出该进程，再加载 NBS。可通过环境变量覆盖路径。已有 E/M `.npy` 可同时设置 `EXPERT_PROB`、`MODELOUT_PROB`，并按已有行/GO 契约提供 ID 文件或 reference manifest。不要把 E 与 M 指向同一个未经核验的数组。

主比较表从 `methods.*.standardized` 读取：

- `standard_protein_fmax`（默认阈值步长 0.001）；
- `standard_micro_pr_auc`（exact 排序后 PR 曲线积分）；
- `standard_micro_ap` 单独记录，不与 PR-AUC 混用。

B/E/M/G 必须用相同蛋白、相同全部任务 GO 列和相同 metric contract。是否超过 B 与是否超过 E/M 是不同的实验结论。历史 legacy step 4000 仅作为背景，不代替本轮相同训练预算的 residual 对照。

### 6.3 epoch 5 运行完整来源诊断

```bash
for preset in residual_epoch direct direct_source_dropout; do
  CHECKPOINT="outputs/latence_nbs_experiments/v087/$preset/nbs_epoch5.pt" \
    bash scripts/nbs/run_nbs_v087.sh effects "$preset"
done
```

该命令评估 9 个分支：`full`、`weak_off`、`core_off`、`pp_off`、`graph_off`、`go_shuffle`、`query_candidate_off`、`neighbor_candidate_off`、`neighbor_pseudo_off`。旧的五项 `full_minus_intervention` 字段保留；三项细来源差值单独写入 `fine_source_effects`。

`weak_off` 同时关闭三类候选/pseudo 图证据，不改变训练时 weak 监督的含义；`graph_off` 关闭这些图证据及 core/PP，仍保留蛋白自身特征。`go_shuffle` 打乱 GO 关联，不是 PP 拓扑打乱。消融改变推理输入分布，其差值用于识别依赖，不等价于重新训练的因果收益。

## 7. 下一轮应回传什么

每组回传 resolved config、training history、validation/development history，以及 epoch 1/3/5 full 的 `nbs_w2s_comparison.json`；epoch 5 再回传 `nbs_graph_effects.json`。无需上传大 checkpoint 或概率矩阵，除非需要定位导出问题。

重点判断：

1. `direct` 相对同预算 `residual_epoch` 的两项主指标和学习曲线是否改善，尤其是 epoch 3→5 是否仍有进展。
2. `direct_source_dropout` 是否进一步改善 full，同时降低某一单独来源关闭造成的退化；不能只追求消融差值变大。
3. core/PP 关闭与 GO shuffle 的表现是否支持模型使用了图信息，而不只是依赖 query candidate。
4. 若直接预测持续落后，检查分角色正例/PU 贡献、初始化及训练收敛，再决定是否修改监督或 PU mining；本轮不同时加入更深 encoder、全新采样和新的教师蒸馏。

## 8. 验证范围

交付前执行新增单元/集成测试、旧模型回归和安装器测试。覆盖 direct 对 dense B 数值的前向不变性、图梯度、三来源隔离、weak 精确覆盖、跨卡尾批、断点恢复、全局监督种子排除、概率导出与评估身份校验。具体通过数量写入包内验证报告。

已通过 194 项 CPU 单元/集成及旧模型回归测试，安装器另有 15 项测试通过。真实两进程 Gloo 测试因当前环境禁止初始化而跳过（1 skipped），因此实际跨卡运行尚未验证；包内保留该测试供服务器执行。未在你的真实大图、GPU / NCCL 或完整 5 epoch 上运行；20 步 smoke 是服务器上的下一步运行检查。代码检查不能提前证明精度会提高。
