# v091 R1：从 v089 epoch60 建立共享评分路径并联合微调

本包提供完整的四阶段入口：冻结表示缓存、共享头预训练、完整模型转换与输出校验、联合微调。使用原三层 v089 的 `nbs_epoch60.pt`；这是独立训练阶段，有新优化器和更新计数，不是旧长训的 epoch61，也不是六层 v090。

## 本轮设置

| 阶段 | 配置 |
|---|---|
| 来源 | `outputs/latence_nbs_experiments/v089/mb12800_fresh_ep100/nbs_epoch60.pt` 与同目录 `resolved_config.json` |
| 冻结缓存 | 编码器、GO 编码器、BN 全部 eval；每个训练蛋白一个固定合法视图；完整 global batch 排除 |
| 共享输入 | 原始 Box center + log(abs(offset))，1024 维；按训练 GO 坐标标准化 |
| 共享头 | MLP 1024→512→321：320 维分类权重 + 1 个 bias；无可训练逐 GO 嵌入表 |
| 预训练 | 500 次更新，weak2048/core512，LR=1e-3，warmup20，之后恒定；每100步保存 |
| 监督 | 继承 v089 core=1.0、weak=0.25；binary CSR 全 GO BCE；保留 alias PU 排除 |
| 联合训练 | 首轮100次更新，global weak10240/core2560，分2个 microbatch，每个6400；每25步保存 |
| 联合 LR | 蛋白编码器3e-4、共享头1e-3、逐 GO 残差增量1e-4；warmup10，之后恒定 |
| 联合目标 | `L_full + 1.0 × L_shared`；共享辅助项默认 detach 蛋白表示，仅训练共享头；完整路径损失更新编码器、共享头和残差 |
| 首轮稳定设置 | GO 编码器继续冻结；蛋白编码器允许梯度，但 BN running statistics 与 dropout 保持 eval；BN affine 可训练 |
| 残差 | 无 dropout，无额外残差惩罚，残差参数组 weight_decay=0 |

这些是首轮实验参数，不是诊断证明的最优参数。共享辅助项系数1.0使共享路径持续得到监督，之后可依据两个路径的表现调整。联合训练始终从真实图重新计算表示；不会错误复用冻结缓存来训练编码器。

共享参数和残差采用数值稳定的等价写法：

```
q, c = shared_head(geometry)
q0, c0 = 转换时保存的共享输出
W_effective = W_old + ((q - q0) + delta_r)
b_effective = b_old + ((c - c0) + delta_s)
delta_r = 0, delta_s = 0   # 初始化
```

这等价于 `r = W_old - q0 + delta_r`、`s = b_old - c0 + delta_s`，但避免了先减去大数再加回造成的初始舍入损失。同一计算后端上初始有效权重就是原 W、b。转换实际运行真实模型 forward 和最多4096个缓存表示检查，失败时不生成 `hybrid_init.pt`。它不读独立测试标签。

## 安装

完整上传 ZIP 后在服务器解压；不要分文件覆盖旧解压目录，以免清单与文件混用。在新解压目录运行：

```bash
python install_v091_shared_ep60.py \
  --project-root /home/dataset-local/data_local/shaojiangyi/latence-project --apply
```

只安装本包新增 v091 文件到原 `scripts/nbs`、`nbs_models/nbs_protein_go/nbs_pg`、configs 和 tests。若已有同名 v091 文件会先备份；旧 v089/v090 入口、P0 采样器和源 checkpoint 均作为依赖使用。

需要此前已安装并验证的 `full_task_sampler_fast.py` 和 `scripts/nbs/diagnose_nbs_knn_v090.py`。源模型、数据、GO、配置和相关原代码继续按 v089 checkpoint 的契约检查。安装器无 `--apply` 时只做预检。

## 1. 设置环境并缓存冻结表示

```bash
cd /home/dataset-local/data_local/shaojiangyi/latence-project
export PYTHON_BIN=/opt/conda/envs/pytorch2.4/bin/python
export CUDA_VISIBLE_DEVICES=0
export DEVICE=cuda:0 NUM_GPUS=1
export CONFIG=nbs_models/nbs_protein_go/configs/bp_shared_head_v091_from_v089_ep60.json
export WORK_DIR=outputs/latence_nbs_experiments/v091/from_v089_ep60_shared_head
export JOINT_WORK_DIR=outputs/latence_nbs_experiments/v091/from_v089_ep60_joint
export INPUT_DIR=outputs/latence_nbs_eval/bp_shared_full_task_top512_v071_v2
export METADATA_FILE=/home/dataset-assist-0/datafile/latence-dataset/unidata_with_exp_train_pseudo.pkl
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
unset INIT_FROM ENCODER_FROM RESUME STEPS STOP_STEP STOP_EPOCH CHECKPOINT
unset HEAD_CHECKPOINT HYBRID_INIT JOINT_CHECKPOINT DEV_MANIFEST PRESET ABLATION
unset METRIC_BACKEND AUPRC_MODE EVAL_DIR CONVERT_DIR CACHE_QUERY_BATCH

bash scripts/nbs/run_nbs_shared_head_v091.sh cache
```

物理 GPU 由 CUDA_VISIBLE_DEVICES 选择，进程内部仍用 cuda:0。使用单 GPU 普通 Python，无需 torchrun。脚本会清理继承的 DDP 环境变量。

缓存位置：

```
outputs/latence_nbs_experiments/v091/from_v089_ep60_shared_cache
```

每个 macro 最多10240个 weak、2560个 core；默认分两次6400查询构图，每次都排除整个 macro 的监督种子。缓存每个训练蛋白一次，尾部可以缺某种角色；预训练时每次更新重新按2048/512抽取。此缓存是固定训练视图，不是旧 DDP 的 RNG 或梯度流重放。

缓存跳过原分类器的大矩阵乘法，直接取得其输入 h。每完成一个 macro 才提交进度与该段哈希；中断后重跑同一条 cache 命令，会核对并接着生成，未完成的 macro 会重算。完整缓存重跑会校验并复用。训练入口拒绝读取 incomplete 缓存。

若要在首次缓存前降低构图规模，可设置 `CACHE_QUERY_BATCH=1280` 或直接修改配置中的 cache.query_batch；之后的续建必须保持相同缓存设置。已生成部分缓存后改变参数时，请在配置中指定新的 cache_dir，保留旧目录。

`candidate_quality_train.json` 记录训练 core/weak 的实际输入 top16 candidate precision、recall、正确数直方图和置信度分档。weak 的正确性以伪标签成员关系计，不代表实验标注准确率。

## 2. 预训练共享头

建议先在正式共享头目录运行3次更新，检查后从同一阶段恢复到500次：

```bash
STOP_STEP=3 bash scripts/nbs/run_nbs_shared_head_v091.sh train

RESUME="$WORK_DIR/shared_head_step3.pt" \
  bash scripts/nbs/run_nbs_shared_head_v091.sh train
```

也可直接执行一次完整 `bash scripts/nbs/run_nbs_shared_head_v091.sh train`，两种方式选一。

`STOP_STEP` 是该新阶段的绝对更新终点，不是增加多少步。不要把旧 `nbs_epoch60.pt` 传给 RESUME；来源已固定在配置中。中断恢复用本阶段自己的 `latest.pt`：

```bash
RESUME="$WORK_DIR/latest.pt" bash scripts/nbs/run_nbs_shared_head_v091.sh train
```

恢复要求配置、缓存、来源和训练代码一致。不要为了从3步继续而把配置 train.steps 从3改为500；配置始终500，用 STOP_STEP 提前停止。

输出包含 `shared_head_step100.pt` 等和最终 `shared_head_step500.pt`，以及 `training_history.json`。该阶段只更新共享 MLP，不反传 GNN。

## 3. 评估共享路径并转换完整模型

```bash
HEAD_CHECKPOINT="$WORK_DIR/shared_head_step500.pt" \
  bash scripts/nbs/run_nbs_shared_head_v091.sh evaluate

HEAD_CHECKPOINT="$WORK_DIR/shared_head_step500.pt" \
  bash scripts/nbs/run_nbs_shared_head_v091.sh convert
```

评估使用与原诊断相同的严格 `>`、0.001阈值网格 protein-Fmax，以及非插值、同分组 micro AP；单位都是百分数。保持全部21312列，无概率融合、无 GO 传播。输出比较：

- 原 epoch60 的 W、b；
- 仅共享 W、b；
- 完整残差初始化；
- 共享 W + 原 b；
- 原 W + 共享 b。

同时给出 core 频次分层指标和独立测试 top16 candidate 质量。低频统计仍沿用0、1–9、10–99、100–999、≥1000分档，便于与R3对照。

评估目录：`$WORK_DIR/eval_shared_head_step500/`。查看 `shared_head_metrics.json` 与 `evaluation_manifest.json`。转换目录：`$WORK_DIR/converted_shared_head_step500/`，内含 `conversion_check.json` 和完整 `hybrid_init.pt`。

转换与评估是独立入口；转换不需要独立测试文件。共享路径表现很差时，可先检查训练和评估结果，再决定是否投入联合微调预算。完整初始化保持源性能是验收条件，不代表已经提高性能。不得把共享路径的已见 GO 指标称为严格零样本。

## 4. 联合微调：先3步检查，再恢复至100步

```bash
HYBRID_INIT="$WORK_DIR/converted_shared_head_step500/hybrid_init.pt" \
STOP_STEP=3 bash scripts/nbs/run_nbs_shared_head_v091.sh joint

RESUME="$JOINT_WORK_DIR/hybrid_step3.pt" \
  bash scripts/nbs/run_nbs_shared_head_v091.sh joint
```

源编码器与新共享头均从转换 checkpoint 加载，建立新优化器。不复用 v089 优化器、旧 LR 日程或旧 epoch 计数。此阶段日志的 `joint_weak_epoch` 只计新增图训练暴露，不把共享头的500步算作原模型的500次训练。

默认 global12800、2个 microbatch，loss 直接使用整批 weak/core 分母并求和，没有再除以2。若需要更小图微批，请在开始任何阶段前调整 joint.accumulation_steps；改动配置之后不能作为同一实验直接恢复旧 v091 checkpoint。

训练输出 `hybrid_step25.pt` 等、最终 `hybrid_step100.pt`、`training_history.json` 和 `joint_contract.json`。中断后：

```bash
RESUME="$JOINT_WORK_DIR/latest.pt" bash scripts/nbs/run_nbs_shared_head_v091.sh joint
```

固定 checkpoint 评估完整路径和共享路径：

```bash
JOINT_CHECKPOINT="$JOINT_WORK_DIR/hybrid_step100.pt" \
  bash scripts/nbs/run_nbs_shared_head_v091.sh evaluate-joint
```

也可用 `JOINT_CHECKPOINT=.../hybrid_init.pt` 检查转换后模型的实际推理。evaluate-joint 支持固定名称的文件；不使用会被续训覆盖的 latest.pt。

## 本地验证及服务器验收

本包在 PyTorch **2.4.1+cpu** 下使用原三层 FullTaskGraphModelV088、原 full_task_loss_v088 和真实 AdamW，完成合成数据的缓存→训练→转换→联合训练→评估测试。覆盖：全局监督排除、缓存中断续建、共享/完整参数梯度、冻结GO和BN统计、原输出保持、角色损失累积、两种训练的断点恢复一致性、分频次评估以及启动器路由。

这不是实际数据或GPU的端到端结果。本地转换 logit 差为0、恢复参数逐位一致仅限所测CPU案例。服务器仍需验证真实缓存、显存、耗时、epoch60初始指标和后续性能。

安装后可运行全部包内测试（测试使用临时合成数据，不读取服务器训练数据）：

```bash
"$PYTHON_BIN" -m unittest discover -s tests -p 'test_*_v091.py' -v
```

建议带回：缓存的 `candidate_quality_train.json`、共享头 `training_history.json`、共享评估 `shared_head_metrics.json`、`conversion_check.json`；若执行联合微调，再附该阶段的 training_history 和评估文件。
