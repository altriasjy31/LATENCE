# NBS v0.8.8.1：独立单卡大微批实验

本包是 v0.8.8 的新增实验配置与启动工具。模型、训练器、数据算法和评测器版本仍为 0.8.8；没有修改它们，也没有改变 Fmax 定义。`release_version=0.8.8` 是现有 loader 的兼容要求，启动工具版本为 `0.8.8.1`。

## 已知结果与本轮目的

| 实验 | 同口径 Fmax | 相对 B | 状态 |
|---|---:|---:|---|
| B | 57.723149 | — | 用户基线 |
| v088 epoch 1，step 3823 | 53.322322 | -4.400827 | 未达标 |
| 本包 mb160 单卡实验 | 未验证 | 未验证 | 未运行真实训练 |

当前比较仍是 BP 的 1800 个蛋白、21312 个 GO 列，标准 protein-centric Fmax，阈值间隔 0.001，无新增概率传播或输出融合。原版与本次上传结果的指标合同、输入身份、B 概率及指标源码哈希一致。报表中的 primary/micro/top_k 等为空是此前 Fmax-only 评测的设计，不是这次 Fmax 下降的原因。

按“至少增加 3 个百分点且达到 60”同时验收，门槛为 **60.723149**。epoch 1 真正退步；BCE 下降只能说明所用训练目标下降，不能证明 Fmax 改善。现有一个评测点不足以认定最终收敛性能，也没有证据保证再训练四轮可以补回差距。本包只检验大微批的训练效率和随后同口径 Fmax，不宣称解决了 Fmax 瓶颈。

上传 history 已含 resume 后 step 3875；epoch 1 的实际训练耗时为 127899.59 秒（35.53 小时），平均 33.46 秒/更新。resume 后 elapsed_seconds 重新计时，不能除以总 step。若保持该速度，再四轮约 142 小时，未计评估和启动开销。

## 唯一新实验

| 配置 | 当前续训 | 新实验 |
|---|---:|---:|
| GPU / 训练进程 | 2 / 2，DDP | 1 / 1，plain CUDA，默认 GPU 1 |
| 每进程 weak / core | 64 / 16 | 128 / 32 |
| accumulation_steps | 4 | 1 |
| 每次前向的查询蛋白数 | 20 / 卡 | 160 |
| 每次更新全局 weak / core | 128 / 32 | 128 / 32 |
| 每轮更新数 | 3823 | 3823 |
| 初始最大学习率 | 0.003 | 0.003 |
| 总计划 / 保存 epoch | 5 / 1、5 | 5 / 1、5 |
| sampler.max_nodes | 100000 | 400000 |
| eval_batch | 16 | 16 |

这是放大实际微批，不是增大全局优化 batch。完整 GO 编码与前后向由原来每卡四次变为新任务一次，单张大图也能合并重复支持节点。原采样器存在逐节点、逐关系扫描64候选以及读取/排序候选 GO 的 CPU 工作，扩大微批不会将这些工作自动移到 GPU。新进程每次更新承担的查询数是旧单个 rank 的两倍，速度必须实测，不能从微批八倍推导八倍加速。

新旧全局样本顺序、曝光数、学习率日程和同一 logits 下的 BCE 归一化可以对齐；模型轨迹不同：query BatchNorm 一次看到20→160个查询、每次更新的统计更新次数改变，dropout、rank 采样种子与支持节点去重也改变。因此从头开始，禁止把旧 `latest.pt` 复制过来或关闭 resume 检查。以后只能恢复本实验自己产生的 checkpoint。

`max_nodes=400000` 仅为超限报错的保护阈值，不预分配400000个节点，不增加 fanout，不截断图。默认 fanout 下160个查询的保守节点上界为 `160×33×11×6=348480`，因此调高该阈值以容纳大微批。该节点上界不能保证显存一定足够，GO 证据、激活和图对象也占资源。

## 应用与运行

在已有、已成功运行 v088 的项目根目录解压此新增包：

```bash
unzip -n /path/to/latence_nbs_v0881_batch.zip -d .
```

包内仅含新增配置、启动工具、对应测试和本说明；没有旧版训练/模型文件。不要修改当前正在使用的 v088 配置。原训练继续使用原来的 work-dir。

新配置路径：`nbs_models/nbs_protein_go/configs/bp_full_task_v0.8.8_mb160_single.json`。

新任务默认目录：`outputs/latence_nbs_experiments/v088_mb160_single/hetero_tuned`。

先运行50步测速；命令中的 Python 就用目前训练环境的 Python：

```bash
/opt/conda/envs/pytorch2.4/bin/python scripts/nbs/run_nbs_v088_mb160.py probe --gpu 1
```

它调用原 v088 prepare 和 train，复用已验证的不可变缓存；不会强制重建池。新任务不调用 torchrun、不创建第二个 NCCL 组，清除继承的分布式启动状态并固定为 world_size=1。默认数值库线程数为1。`--dry-run` 可先检查命令且不创建实验目录。

测速结束会打印最近有效窗口的秒/更新、PyTorch 峰值 allocated 显存与不含评估的训练时间估计。前10个更新不用于稳定速度估计；这些步骤已经训练，后面不会重复。

确认速度与并行影响可接受后，从本实验 step 50 继续到 epoch 1：

```bash
/opt/conda/envs/pytorch2.4/bin/python scripts/nbs/run_nbs_v088_mb160.py train --gpu 1 --stop-epoch 1
```

新实验完成 epoch 1 后按原条件评估：

```bash
/opt/conda/envs/pytorch2.4/bin/python scripts/nbs/run_nbs_v088_mb160.py evaluate --gpu 1 --epoch 1
```

评估使用现有默认输入和 metadata 路径；若服务器之前覆盖了路径，请传入同一 `--input-dir`、`--metadata-file`。不运行 core、query_candidate、pp 单独消融，不增加 E/M 融合。

```bash
/opt/conda/envs/pytorch2.4/bin/python scripts/nbs/check_nbs_v088_acceptance.py \
  --baseline-comparison BASELINE_CONTRACT_v088.json \
  --comparison outputs/latence_nbs_experiments/v088_mb160_single/hetero_tuned/eval_step3823/full/metrics/nbs_w2s_comparison.json \
  --output outputs/latence_nbs_experiments/v088_mb160_single/hetero_tuned/acceptance_epoch1.json
```

验收器输出 `未达标` 时退出码为1，`未验证` 为2，这是既定语义。不要用内部 holdout monitor 替代外部 Fmax 验收。

若继续该新实验到 epoch 5：

```bash
/opt/conda/envs/pytorch2.4/bin/python scripts/nbs/run_nbs_v088_mb160.py train --gpu 1 --stop-epoch 5
/opt/conda/envs/pytorch2.4/bin/python scripts/nbs/run_nbs_v088_mb160.py evaluate --gpu 1 --epoch 5
```

其中 epoch 5 对应 step 19115，验收时相应改用 `eval_step19115/full/metrics/nbs_w2s_comparison.json` 和新的输出文件名。训练器始终保留5轮的学习率计划；先在50步或epoch1停止，不会压缩学习率周期。

## 并行运行怎么判断

两个现有 PID 是同一项双卡 DDP 训练的两个 rank。再开本包只增加一个训练进程，占用 GPU 1；它与原任务仍共享 GPU、CPU、内存带宽和数据缓存。独立目录与文件锁保护实验输出，不提供计算资源隔离。

PyTorch 对 NCCL 进程共享 GPU 有明确死锁警示，故不推荐在这两张卡上再叠加一组双卡 DDP。改为单卡任务只是避免第二个 NCCL 组，仍不能保证原任务不减速或不中断。需要保证原任务独占时，应等它结束或使用另一张独立 GPU。

测速前后观察原任务连续数个25步日志窗口，同时观察新任务的稳定速度。现在原任务约34.4～34.5秒/更新；若并行后出现持续明显减速、停滞或 OOM，停止新增任务，不自动改动原任务。空余显存不能代替这项检查。

```bash
nvidia-smi --query-gpu=index,memory.used,memory.free,utilization.gpu,utilization.memory --format=csv -l 1
/opt/conda/envs/pytorch2.4/bin/python scripts/nbs/run_nbs_v088_mb160.py status
```

两个任务更新的全局查询数都为160，可用 `160/旧任务秒每步 + 160/新任务秒每步` 观察合计查询吞吐。这个数只衡量资源使用效率，不是模型收益，也不能替代各自完成同样 epoch 后的 Fmax。

额外成本包括：一个训练进程、一套模型/优化器状态、更大的激活/图及数据内存，和一次新的完整训练。GPU 显存、CPU RAM、并行延迟及实际加速均需50步测速确认。日志中的1.82GB只是 PyTorch rank0 的峰值 allocated，不是整卡总显存或持续利用率。

## 本地验证边界

实际 A100、大数据训练和新实验 Fmax：**未验证**。CPU 检查使用实际样本数量，验证全部19115步的全局查询 ID、曝光量与学习率序列相同；同一 logits 下完整 batch 与原两 rank × 四 micro 的损失/梯度归一化吻合（浮点误差量级），含每轮剩余6个 weak 的尾批。这不意味着包含 BatchNorm、dropout 的真实模型轨迹相同。启动工具专项测试 **31项通过**，覆盖目录隔离、同目录恢复、分布式环境清理、锁、评估调用与 resume 后计时；在合并现有源码的独立验证目录中也已通过。命令 dry-run 确认为单进程调用原训练器。这些检查不能证明 Fmax 会提升。

参考：PyTorch 2.4 distributed documentation，`init_process_group` 的 GPU 独占说明：https://docs.pytorch.org/docs/2.4/distributed.html
