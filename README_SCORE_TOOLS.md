# 已有概率的离线变换与融合评估

本包新增两个 CPU 离线工具文件，不改变 v092 模型、训练入口、配置、checkpoint、原评估概率或验收结果。文件名中的 v093 只是离线工具编号，不是新训练模型。

当前图模型的导出是 `sigmoid(logits)` 后的概率。再对这些概率做一次 sigmoid 是负对照，不是补上遗漏的 sigmoid。全局严格单调变换在精确算术下保持排序；对应映射阈值时预测集合也不变。0.001 固定阈值网格、原始0/1分数的端点限制和有限精度 ties 会带来指标差异。

## 安装与一条命令运行

下载并解压完整 ZIP，在解压目录运行：

```bash
/opt/conda/envs/pytorch2.4/bin/python install_score_tools.py \
  --project-root /home/dataset-local/data_local/shaojiangyi/latence-project --apply
```

在工程根目录评估已导出的 epoch60：

```bash
cd /home/dataset-local/data_local/shaojiangyi/latence-project

/opt/conda/envs/pytorch2.4/bin/python scripts/nbs/eval_nbs_score_tools_v093.py \
  all --epoch 60
```

不需要 CUDA、export、重新推理或重新训练。可以与 GPU 训练并行，但 CPU 排序和磁盘读取仍会消耗主机资源；多个诊断建议逐个运行。精确 AP 需要对约 3800 万个评分元素排序，耗时和内存不是零；工具逐方法处理并使用原项目的分块精确指标实现。

只做第一项分数变换时，将 `all` 改为 `transform`；只做混合时，改为 `mix`。更换 checkpoint 时只改 `--epoch`，该 epoch 必须已经完成正式 evaluate，并有 `metrics_v092/summary.json`。

默认输入沿用此前实验：geometry_single12800_scratch、bp_shared_full_task_top512_v071_v2，以及原 metadata。若用了自定义路径，显式传入 `--work-dir`、`--input-dir`、`--metadata-file`。不读取旧 shell 的 WORK_DIR/CONFIG 等变量。

## 变换方案

| 方法名 | 操作 | 用途 |
|---|---|---|
| B / G | 原 backbone / NBS 概率 | 必须先复现已有整体 Fmax 与 AP |
| G_sigmoid_again | 对 G 的概率再 sigmoid | 单调变换负对照；不预期提升理论排序 |
| G_global_minmax | 对整个 G 矩阵做一次 min–max | 全局分数刻度负对照 |
| G_row_max | 每个蛋白的 GO 分数除以本行最大值 | 检验蛋白之间置信度尺度的影响；零行保持零 |
| G_go_logit_zscore | 每列 GO 的 logit 标准化后再 sigmoid | 检验 GO 之间分数尺度的影响 |

逐 GO 标准化采用 `sigmoid((logit(p)-mean_g)/max(std_g,0.001))`，总体均值、标准差来自当前不含标签的评估分数。使用 float64、稳定 logit、仅端点裁剪。常数列映射到约0.5，低方差列有明确尺度下限。

**逐 GO 标准化是依赖当前无标签样本集合的诊断，不是独立开发集拟合出的正式校准器。**其分数会随整批输入的组成改变；若要部署或正式验证，应在独立校准集固定统计后，再用于测试。

逐 GO 单调变换在精确算术下保持该 GO 内蛋白排序，因此 per-GO AP / macro AP 应基本不变，却可能改变全局 micro AP 和 Fmax。实际浮点 ties 可能产生小差异，工具报告实际计算结果，不强行覆盖为理论值。

## 混合方案

令 alpha 为图模型 G 的权重：

| 方法名 | 固定规则 |
|---|---|
| prob_mix_g025 / g050 / g075 | `(1-alpha)*B + alpha*G`，alpha固定为0.25、0.5、0.75 |
| logit_mix_g050 | `sigmoid(0.5*logit(B)+0.5*logit(G))` |
| max_BG | `maximum(B,G)`，观察扩大覆盖与误报的权衡 |
| freq_protect_9 | core频次≤9的列原样使用B，其余列0.5B＋0.5G |
| freq_protect_99 | core频次≤99的列原样使用B，其余列0.5B＋0.5G |

alpha 数值越大，G 的占比越高。所有权重和门槛预先固定；工具不根据测试结果自动选最优，也不把混合结果写回原 no-fusion 验收。

logit 混合单独使用预先固定的对称 `epsilon=1e-6` 截断到 `[epsilon,1-epsilon]`，使0和1的极端概率对称处理，并记录被截断的数量；原始B/G与概率混合不做这种截断。逐GO标准化仍只处理精确端点，以尽量保留原始GO内排序。

频次门控只读取训练 core 的原 gold CSR，并按原 seed/holdout 重建训练 core 集合。频次和训练 core ID 的哈希必须与已生成的 v092 summary 一致；不使用测试阳性或 weak 伪标签频次来决定门控。

被保护列的 B 概率原样保留，因此该组 AP 也保持。但融合自身的整体 Fmax 阈值会变化，**不能承诺在新全局阈值下仍保留相同的 rare recall**。报告另给所有方法在 B 原整体阈值下的检出，以检查这一点。

## 输出

默认目录：

```
outputs/latence_nbs_experiments/v092/geometry_single12800_scratch/
  score_postprocess_epoch60/all/
```

- `report.json`：全部固定方法及来源、整体和分层指标；优先回传这个文件。
- `methods.csv`：整体 Fmax、precision、recall、micro AP、macro AP。
- `frequency_strata.csv`：core频次分层 AP / macro AP，以及各自 Fmax 阈值下 TP、预测数、pair P/R。
- `*_per_go_ap.npy`：每个原始 GO 列的 AP（百分数），没有测试阳性的列为 NaN。
- `inputs.json`：输入与代码身份，供重复运行和结果复核。

报告保留全部原始 GO 列，不做本体概率传播。protein Fmax 沿用严格 `>` 和0.001阈值网格；AP为精确非插值AP，非梯形PR-AUC。macro AP只平均有测试阳性的固定GO集合，所有方法使用相同集合，并报告列数。

频次分组0、1–5、1–9、10–99、100–999、≥1000；1–5嵌套在1–9中，不能相加。分组AP也不能直接相加分解整体AP。

每完成一个方法就保存进度。同一命令重复运行会核验来源并复用已完成方法；输入、代码或统计改变时，要求另用新的 `--output-dir`。原始 B/G 的 Fmax 和 AP 无法复现时会停止，不继续接受变换后的结果。

## 如何判读

1. 二次 sigmoid 或全局 min–max 只有很小的 Fmax 变化、AP基本不变：多半是网格或浮点效应。
2. 逐 GO 归一化改善 micro AP / Fmax，但 macro AP基本不变：标签间尺度存在影响；若 macro AP本身低，GO内部排序也仍有不足。
3. 融合超过两条单路：说明存在可用互补。若任何加入G的组合都低于B，当前G的增量信息有限，或分数尚未对齐。
4. 门控恢复稀有组AP并改善整体：证明现有标签上的互补，不证明共享G分支自身恢复了稀有或零样本能力。

这些是在当前测试集上的固定方案诊断。正式方法的参数和统计应在独立开发/校准集确定，然后冻结到测试集，不把本工具中的最大数值自动当成最终方法结果。

## 验证范围

22项小型CPU测试通过：15项变换、混合及指标测试，7项端到端CLI测试。原B/G指标与生产内核一致；macro AP及ties与sklearn对齐；覆盖原始ID重排、core holdout重现、CSR去重计数、分数不变、结果重用与来源冲突拒绝。测试不代表真实1800×21312矩阵已在本地执行，也不预先保证融合能超过B。

安装后在工程根目录可运行：

```bash
/opt/conda/envs/pytorch2.4/bin/python -m unittest discover -s tests -p 'test_score*v093.py' -v
```
