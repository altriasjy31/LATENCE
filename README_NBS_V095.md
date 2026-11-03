# v095：ASL、选中弱监督与 GO 层级约束联合训练

本包是已有 v094 工程的增量更新，只新增 v095 文件。直接在 latence-project 根目录解压。复用 v094 已构建的 backbone hard / modelout soft 数据，无须重建标签、重新校准阈值或替换旧实验文件。

## 推荐运行

在已激活 PyTorch 环境的工程根目录执行。默认使用物理 GPU 1；启动器会清理历史 CONFIG/WORK_DIR/RESUME 等变量并重新设置 CUDA_VISIBLE_DEVICES。

```bash
unzip LATENCE_nbs_v095_asl_hierarchy_r1.zip -d .
python scripts/nbs/run_nbs_v095.py check
python scripts/nbs/run_nbs_v095.py train --gpu 1 \
  --init-from outputs/latence_nbs_experiments/v094/weak_teacher_single12800/nbs_epoch20.pt
```

推荐使用已完成的 v094 epoch 20 权重以节省重复学习。初始化会核对模型配置、数据和本体身份、GO 轴、seen 掩码及 checkpoint 完整性；允许的架构配置变化只有 JK last→sum。新 optimizer、学习率日程和 epoch 计数从零开始。v095 epoch 5 表示新目标训练了 5 轮，并非全程累计 5 轮。JK 改变聚合方式，初始化后输出不保证与 v094 完全一致。

如果明确希望从头训练，省略 `--init-from`。两种方式共用默认输出目录，因此请选择一种启动；另开实验需另设 `--work-dir`。

默认目录：

- 来源：`outputs/latence_nbs_experiments/v094/weak_teacher_single12800`
- 新实验：`outputs/latence_nbs_experiments/v095/asl_hierarchy_single12800`
- 冻结配置：新实验目录旁的 `asl_hierarchy_single12800_config/training_config.json`

来源路径不同，在首次 train 时提供 `--source-work-dir /实际/v094/工作目录`，并相应调整 `--init-from`。数据路径从来源 resolved_config.json 读取。

可选的 3-step GPU 冒烟测试：将主命令的 `train` 改成 `smoke`，其他参数相同。输出进入单独的 `_smoke` 目录；该操作用于检查实际设备的内存与执行，不参与正式训练。

中断后续训：

```bash
python scripts/nbs/run_nbs_v095.py resume --gpu 1
```

在空闲 GPU 上并行评估固定完成的 epoch：

```bash
python scripts/nbs/run_nbs_v095.py evaluate --gpu 2 --epoch 5
```

`--gpu 2` 需换成实际空闲卡。训练锁和评估锁独立；各版本评估仍共用推理缓存锁，避免同时构建共享缓存。评估输出保持原 exact Fmax / Micro AP / Macro AP 与分频率口径，不在推理端融合 backbone。

已有 GO / 本体新增 GO 的请求式预测仍由同一共享评分器执行：

```bash
python scripts/nbs/run_nbs_v095.py predict-go --gpu 2 --epoch 5 \
  --go-ids-file requested_go_v094.example.txt
```

本体中尚不存在的新 GO 仍需提供锚定的更新本体及几何特征，接口为 `--ontology-file`。本包不会自动生成新 GO 的 BoxSquaredEL 特征。zero-shot 保持 Stage2 监督排除范围，不能据此声称 Stage1 从未见过对应功能。

## 固定训练方案

| 项目 | v095 默认值 |
|---|---|
| 有效 batch | 12800 = weak 10240 + core 2560 |
| 梯度累积 | 2 次，每次 6400 个种子；按完整 batch 角色数量归一化 |
| 编码器 | 3 层 heterogeneous tuned SAGE，hidden 320 |
| 蛋白质聚合 | 上一层 raw residual → typed BN → ReLU → dropout；JK=sum |
| GO 表示 | 本体编码 + center/log-offset 直接几何输入 |
| 输出 | 单共享 GO scorer |
| 精度 | CUDA BF16 autocast；关键稀疏计算及损失保留 FP32 |
| Optimizer | AdamW，eps=1e-8，betas=(.9,.999)，矩阵 weight decay=1e-4，向量/偏置不衰减 |
| 学习率 | 5 轮 warmup 至 .003，保持至第 10 轮，再 cosine 至第 200 轮的 .00003 |
| 梯度裁剪 | 全模型 norm 5 |
| 保存 | 每 5 轮固定 checkpoint，并保存 latest |

这沿用 tunedGNN large_graph/models.py 的残差后 BN 与 JK 思路，同时使用已有异构关系消息传递。蛋白质主干并非 pre-LN，也不是所有历史层直接参与每层残差。

官方参照：https://github.com/LUOyk1999/tunedGNN/blob/main/large_graph/models.py

不继续增大 batch：当前数据量下，每轮约 48 次 optimizer 更新，200 轮约 9600 次。确切次数以运行日志为准。用户提供的 Stage1 配置是 4 卡、每卡 6 core + 40 weak，即全局 184，且有 init_ckpt；其 600 epochs 不能与图模型按轮数直接等同。BF16 的实际提速与最大显存须由目标 A100 实测。

## 三类监督同时启用

1. **ASL**：gamma_neg=4、gamma_pos=0、clip=.05，focal 权重 detach。core 对 eligible seen GO 计算正负监督；weak 仅将 backbone 阈值选中的 B-positive 集合作为硬正标签，其他项先视为未知。
2. **选中 modelout 蒸馏**：每个 weak 蛋白选 teacher top2048 且 M≥.05，使用 sqrt(M) 置信权重。B-positive 中 M≥.5 的项额外纳入；M<.5 的 B-positive 不接收相反的 soft 约束。另在 B-negative 且 M≤.01 的项中选择学生分数最高的至多 256 个负项，使用 .25×sqrt(1-M) 权重。各项为 Bernoulli KL(M||q)，teacher detach；未知区域不再承受全列 dense KL。
3. **GO 层级一致性**：is_a / part_of 转为 child≤parent 的概率约束，去重 canonical 边，只监督两端均 seen 的边；alias-PU 排除项也排除出对应边损失。

core/weak 硬监督权重 .25/1，经角色归一化后在两角色均存在时为 .2/.8；蒸馏为独立 weak-mean、权重 1。层级项按角色加权的每行边均值计算，权重 .01。硬监督和蒸馏沿用固定 seen GO 分母，保证分块累积不会改变目标。

最终整体乘 `loss_scale=1000`。这是固定数值尺度选择；AdamW 对统一缩放近似不敏感，不能将其理解为“学习能力增强 1000 倍”。新 loss/grad 的数值不能直接和 v094 对比，需看 unscaled_loss 和各项 contrib。该设置与裁剪共同构成新实验配置，训练后不应随意更改。

这是一组联合策略实验，目标是集中改善监督和优化；其结果不能用于单独归因 ASL 的贡献，也无法预先保证超过指定性能线。

## 数据与运行边界

- 既有 v094 标签成员关系、候选 16 个预算和文件缓存直接复用；本次改变训练时的未知项与蒸馏解释。
- 新提供的 Stage1 ep600 参数用于确定迁移机制。本包未收到其完整模型/特征/概率文件，因此不会自动将旧缓存替换为新 Stage1 输出。
- ASL、蒸馏和层级项都遵守 seen 和 alias-PU 掩码。训练目标蛋白的标签边排除机制继续由原数据加载器执行。
- 初始化只读固定 v094 checkpoint；后续 v095 resume/evaluate 不需要重新打开初始化 checkpoint。
- 使用原 v094 数据契约以验证权重迁移；其中旧的 dense-supervision 文字表示来源数据契约。实际优化由 v095 冻结配置及 loss_definition 定义。
- 运行参数、数据身份、实现哈希及日程进入 checkpoint 契约。更换损失后不能沿用旧 optimizer 伪装成原实验 resume。
- 非 BF16 设备可在首次 train 时指定 `--precision fp32`；同一实验不能中途切换精度。

## 验证

包内 `tests/test_v095*.py` 覆盖 ASL、未知项及 teacher 冲突处理，完整/分块 batch 梯度等价、模拟 DDP 角色归一化，层级方向与别名去重、真实小型异构图前后向、BF16 dtype、优化器恢复、权重初始化与启动参数。交付时的实际测试结果记录在 `VALIDATION_V095.json`。

本地验证使用 CPU PyTorch；没有目标服务器的完整数据和 CUDA 设备，未执行真实 12800-batch GPU 训练，也不报告未经测量的速度或效果。
