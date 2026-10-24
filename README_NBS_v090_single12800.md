# NBS v0.9.0：单卡 12800、六层候选配置

本补丁基于用户上传的双卡配置，提供可运行的单卡配置及必要的六层采样支持。
这组参数尚未完成生产数据 GPU 验证，不能声称优于 v089。

## 参数

| 项目 | 设置 |
|---|---|
| 配置文件 | `nbs_models/nbs_protein_go/configs/bp_full_task_v0.9.0_mb12800_single_ep100_6layer_lr001_weak.json` |
| GPU / 进程数 | 1 / 1，不使用 torchrun |
| 常规 optimizer update 的监督蛋白 | 10240 weak + 2560 core = 12800 |
| accumulation_steps | 1，默认单次前后向直接使用 12800 |
| 当前数据的步数 | ceil(489222 / 10240) = 48 updates/weak epoch；100 epoch = 4800 updates |
| 模型 | heterogeneous SAGE 6 层，H=320；独立 GO 图编码 2 层；共享 protein–GO scorer |
| dropout | input=0.10，hidden=0.30，ontology=0，edge=0；JK=last |
| loss | core_weight=0.65，weak_weight=1.0 |
| 优化器 | AdamW，峰值 LR=0.01，WD=0.0001，grad_clip=5 |
| 调度 | warmup 2 epoch，保持至 epoch 10，然后 cosine 至 epoch 100，min_lr=0.001 |
| 保存 | 每 5 epoch 保存固定 checkpoint |
| 六层 fanout：outer→inner | 前四层每种关系各 1；第五层每种关系各 2；第六层仅 cosine=32 |

五种蛋白关系为 ppi、similar_to、weak_to_core、core_to_weak、cosine。
新增的三个远层采用稀疏预算；六个消息块独立采样并严格首尾衔接，不重复使用三个旧块。
ghost 查询每层仍只从 cosine 支持邻居接收消息，全局监督蛋白和 holdout 排除规则保留。
fanout=1 且 stable_fraction=0.5 时固定份额取整为 0，训练使用动态抽取，推理使用确定性 top1。

每个 weak epoch 的尾批保留实际不足的 weak 数量，不填充重复样本；12800 指常规宏批量。

core/weak 是各自均值损失的权重，归一化系数约 39.4%/60.6%，不代表实际梯度贡献占比。
NBS 保持二值全列 BCE，提高 weak 权重同时影响其正例和负例；没有把一阶段软标签 ASL 偷换进来。
单卡与旧双卡保持宏批量及更新数一致，但 BN 统计、支持节点去重、随机采样和 dropout 路径不同，不能称逐位等价。

## 安装

先把 zip 放到项目根目录，从项目根目录执行。解压目录放在项目之外：

```bash
unzip latence_nbs_v090_single12800.zip -d /tmp
python3 /tmp/latence_nbs_v090_single12800/install_nbs_v090_single12800.py \
  --project-root "$PWD"
```

可追加 `--dry-run` 先查看安装计划。该包包含所需 v090 代码，因此无需先手工安装上一份 v090 包。
安装器会先检查全部依赖及冲突，只升级已知原版 `full_task_data_v090.py` 并备份；相同文件保留，未知本地修改会被拒绝。
不覆盖用户改过的双卡配置，也不改 v088/v089 源文件。
补丁会改变 v090 的预测源码哈希：应在启动这个新实验前安装，不适用于正在运行的旧 v090 训练。

## 首次运行：短停后继续同一训练

在独立终端、项目根目录执行：

```bash
export PYTHON_BIN=/opt/conda/envs/pytorch2.4/bin/python
export CUDA_VISIBLE_DEVICES=0 CUDA_DEVICE_ORDER=PCI_BUS_ID
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export CONFIG=nbs_models/nbs_protein_go/configs/bp_full_task_v0.9.0_mb12800_single_ep100_6layer_lr001_weak.json
export NUM_GPUS=1 DEVICE=cuda:0
export WORK_DIR=outputs/latence_nbs_experiments/v090/mb12800_single_6layer_lr001_weak_ep100

unset INIT_FROM ENCODER_FROM RESUME STEPS STOP_STEP STOP_EPOCH CHECKPOINT DEV_MANIFEST PRESET
unset ABLATION METRIC_BACKEND AUPRC_MODE

STOP_STEP=3 bash scripts/nbs/run_nbs_v090_single12800.sh train
```

六层显存/主存峰值尚未实测。先确认三次更新完成，查看日志的 peak_allocated_gb、max_micro_nodes、max_micro_edges 和 seconds_per_step。
这个短停不改变 100 epoch 的调度；无需重新从零开始。正常后继续：

```bash
RESUME="$WORK_DIR/latest.pt" bash scripts/nbs/run_nbs_v090_single12800.sh train
```

若希望不短停，第一条训练命令去掉 `STOP_STEP=3` 即从零完整训练。
不添加 `single` 或 `ddp` 第二个参数；专用脚本固定单卡。
CUDA_VISIBLE_DEVICES=1 可选择物理 GPU 1，进程内 DEVICE 仍为 cuda:0。
六层不能直接从三层 v089 checkpoint 做 encoder transfer，推荐 fresh。

如果单次 12800 前后向 OOM，可在尚未正式开始的新配置中把 accumulation_steps 改成 2，每个 microbatch 为 5120 weak+1280 core，宏批量仍为 12800、每 epoch 仍为 48 次更新。
此时 BN 按 6400 的 microbatch 统计，不能声称等价于默认单次 12800；使用新 WORK_DIR，不能改配置后 RESUME 已有训练。

当前 v089 双卡 DDP 若继续运行，新单卡仍会共享物理 GPU/CPU/主存，并不等于资源隔离。
先做资源探测并比较旧实验平均约 258 秒/step 的耗时，不能只看瞬时 GPU 0%。不新增第二组双卡 NCCL。
完整匹配的采样池可只读复用，勿强制重建共享缓存。

## 简单评估

epoch 5、10 等固定 checkpoint 生成后，保持同一配置和 WORK_DIR：

```bash
CHECKPOINT="$WORK_DIR/nbs_epoch5.pt" \
bash scripts/nbs/run_nbs_v090_single12800.sh evaluate
```

沿用原始 1800 蛋白×21312 GO、严格 >、0.001 阈值网格的 protein-centric Fmax，以及固定 B 的验收检查。
不加概率融合或本体传播。训练期间可评估已完成的固定 checkpoint；不要用 latest.pt 代替固定 epoch。
尚未提供生产 Fmax、GPU/NCCL 或新增 GO 泛化收益的实验证据。
