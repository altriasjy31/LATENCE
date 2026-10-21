# NBS v0.8.8.2：全局 batch 五倍、学习率两倍

本版按用户澄清，将每次更新的全局 weak/core 从128/32增至640/160，单卡一次前向800个查询，accumulation_steps=1。与 v0.8.8.1 只调整微批不同，本版确实减少完成一轮训练所需的参数更新次数。模型、数据算法和评测器仍为 v0.8.8；启动工具版本为0.8.8.2，配置的 release_version=0.8.8 用于兼容既有 loader。

## 配置

| 项目 | 原 v088 双卡 | v0.8.8.1 单卡 | 本版单卡 |
|---|---:|---:|---:|
| 全局 weak/core 每次更新 | 128/32 | 128/32 | **640/160** |
| 单卡每次前向查询数 | 20 | 160 | **800** |
| 每卡梯度累积 | 4 | 1 | **1** |
| 每 epoch 更新数 | 3823 | 3823 | **765** |
| 五轮总更新数 | 19115 | 19115 | **3825** |
| 最大学习率 | 0.003 | 0.003 | **0.006** |
| 末端学习率 | 0.0003 | 0.0003 | **0.0006** |
| warmup / hold截止 | 0.1 / 3 epoch | 相同 | 相同 |
| eval_batch | 16 | 16 | 16 |
| max_nodes | 100000 | 400000 | 1800000 |

这里学习率按要求加倍，不按batch倍数自动放大到0.015。第77次更新达到0.006，第2295步仍为0.006，第3825步降到0.0006。仍按完整五轮计划计算学习率，测速或先停epoch1都不会压缩日程。

489222个weak每轮完整遍历，末批为262 weak＋160 core，共422个查询，不丢尾批。core因每轮更新数向上取整，每轮曝光122400次，比原来的122336次多64次（约0.0523%）；不宣称新旧曝光数完全相同。

保持相同模型、loss权重、fanout和评测条件。全局batch、学习率及更新次数改变了优化过程；监督种子排除集合从160增至800，BatchNorm和采样分组也改变。必须fresh run，不能恢复旧 v088 或mb160的checkpoint。训练开始后只能续接本实验自己保存的checkpoint。

max_nodes只是超限报错阈值，不预分配、不增加邻居数、不截断图。800查询的保守fanout上界为800×33×11×6=1742400，因此设为1800000；实际图会去重且受蛋白总数限制。节点阈值不构成显存保证。

## 应用

在已有可运行v088的项目根目录执行：

```bash
unzip -n /path/to/latence_nbs_v0882_batch.zip -d .
```

仅新增配置、启动工具、专项测试和本说明，不覆盖正在使用的训练/模型源码。若已启动mb160，可以在其终端Ctrl+C；不要停止原双卡v088的两个rank。本版默认输出目录是 `outputs/latence_nbs_experiments/v088_mb800_single/hetero_tuned`。

## 测速与训练

使用当前pytorch2.4环境，默认GPU1，先跑20步：

```bash
/opt/conda/envs/pytorch2.4/bin/python scripts/nbs/run_nbs_v088_mb800.py probe --gpu 1
```

前10步不进入稳定速度估计，默认测速只剩10个更新的初步测量。20步尚未到达最高学习率，不能当作0.006已稳定的证据。若第一次启动就希望覆盖warmup，可将上述命令改为：

```bash
/opt/conda/envs/pytorch2.4/bin/python scripts/nbs/run_nbs_v088_mb800.py probe --gpu 1 --probe-steps 100
```

两条probe命令选其一，不要在同目录重复运行probe。新配置不自动缩batch或降低学习率；OOM或错误会中止并保留原任务。可以加 `--dry-run` 检查启动命令而不创建目录。

测速可接受后，直接接着测速checkpoint训练到epoch1：

```bash
/opt/conda/envs/pytorch2.4/bin/python scripts/nbs/run_nbs_v088_mb800.py train --gpu 1 --stop-epoch 1
```

原训练器继续检查有限loss/梯度；第77步以后自然进入0.006阶段。epoch1完成后同口径评估：

```bash
/opt/conda/envs/pytorch2.4/bin/python scripts/nbs/run_nbs_v088_mb800.py evaluate --gpu 1 --epoch 1
/opt/conda/envs/pytorch2.4/bin/python scripts/nbs/check_nbs_v088_acceptance.py \
  --baseline-comparison BASELINE_CONTRACT_v088.json \
  --comparison outputs/latence_nbs_experiments/v088_mb800_single/hetero_tuned/eval_step765/full/metrics/nbs_w2s_comparison.json \
  --output outputs/latence_nbs_experiments/v088_mb800_single/hetero_tuned/acceptance_epoch1.json
```

评测默认采用之前相同的input-dir、metadata-file；若之前使用了其他路径，请向evaluate传入对应 `--input-dir`、`--metadata-file`。不运行暂停的消融，不融合E/M。

后续继续至epoch5：

```bash
/opt/conda/envs/pytorch2.4/bin/python scripts/nbs/run_nbs_v088_mb800.py train --gpu 1 --stop-epoch 5
/opt/conda/envs/pytorch2.4/bin/python scripts/nbs/run_nbs_v088_mb800.py evaluate --gpu 1 --epoch 5
```

epoch5验收时，comparison路径改为 `eval_step3825/full/metrics/nbs_w2s_comparison.json`，验收输出改为新的epoch5文件名。epoch5的3825步与旧版epoch2的3825步不同，比较时看epoch/weak曝光数，不按相同步数比较。

## 如何判断速度

原v088 epoch1实际训练耗时127899.59秒（35.53小时，不含随后评估）。本版每轮765个更新，估计每轮耗时为 `765×稳定秒每步`。

| 本版稳定秒/步 | 外推一轮训练耗时 | 相对原epoch1外推加速 |
|---|---:|---:|
| 33.44 | 7.11小时 | 约5倍 |
| 55.73 | 11.84小时 | 约3倍 |
| 83.59 | 17.76小时 | 约2倍 |
| 167.19 | 35.53小时 | 基本持平 |

这些是代数换算，不是性能预测。当前mb160日志约42秒/步不能代替mb800测量。status会打印按当前速度外推的每轮耗时和加速比，并明确它不是完成整轮后的实测速比：

```bash
/opt/conda/envs/pytorch2.4/bin/python scripts/nbs/run_nbs_v088_mb800.py status
```

CPU采样随图规模增长，减少更新次数不保证等比例缩短训练；共享GPU时还需观察原任务速度。额外显存、主机内存和真实时间成本尚未验证。减少卡数和并行负载不同，也意味着该速度比是实际运行方案的耗时比较，而非受控的单因素硬件基准。

## 验证与验收状态

- 新启动工具专项测试36项通过，覆盖配置、20/100步probe、同目录恢复、765步epoch约束和评估调用。
- 用实际样本数量核对epoch/尾批和学习率边界；CPU验证800查询与422查询尾批的BCE角色归一化正确、梯度有限。
- 未在A100和真实数据上运行本版，真实速度、峰值资源和Fmax均为 **未验证**。
- 原v088 epoch1 Fmax=53.322322，B=57.723149，差值-4.400827，**未达标**。新实验继续使用相同1800蛋白、21312 GO列和标准protein Fmax；同时满足提高至少3且达到60的门槛为60.723149。大batch/更高LR不构成Fmax改善的保证。
