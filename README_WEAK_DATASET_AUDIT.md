# Weak 标签集审计与下一轮数据设计

本包完成当前阶段的只读诊断。它不生成未经验证的新伪标签，也不改变训练数据、模型、配置或运行中的训练。

## 本轮 report 的证据

输入为 epoch 60 的 `nbs-offline-score-tools-v093-r1` 完整 report。单位均为百分数。

| 方法 | protein Fmax | micro AP | macro AP |
|---|---:|---:|---:|
| Backbone B | 57.723149 | 49.450788 | 38.705788 |
| Stage2 G | 55.838449 | 48.465182 | 34.067570 |
| 0.75 B + 0.25 G | 58.121170 | 50.415164 | 38.912922 |
| 等权 logit 混合 | 57.965390 | 52.473104 | 39.917857 |
| ≤99 次列保留 B，其余等权概率混合 | 56.860754 | 48.771657 | 38.772453 |

重复 sigmoid、全局 min-max 和逐蛋白最大值归一化没有实质修复。G 在频次 1–5 组的 micro AP 为 3.394640，低于 B 的 23.262795；相应 macro AP 为 31.358987 与 39.382143。这说明除了 GO 间分数尺度，还有 GO 内部排序退化。逐 GO 单调变换无法恢复后者。

logit 混合的各频次组 micro/macro AP 均超过 B；它包含 ε=1e-6 的对称概率裁剪，G 有约 74.02% 的单元被裁剪。因此提升属于“裁剪＋融合”的完整规则，尚不能归因于概率校准或证明蒸馏有效。

硬频次切换保住部分列的原始分数和 AP，也不保证单一全局阈值下的 rare precision。频次分界、阈值与教师来源应在校准数据上确定。

以上支持先审计 weak 教师和监督构建，但不能仅据测试预测报告认定数据集就是主因。当前报告比较 B 与 Stage2 G，没有验证“低频 B、高频 Stage1 expert”的分工。

## 安装与运行

在解压后的包目录运行一次：

```bash
/opt/conda/envs/pytorch2.4/bin/python install_weak_dataset_audit.py \
  --project-root /home/dataset-local/data_local/shaojiangyi/latence-project --apply
```

随后在项目目录运行：

```bash
cd /home/dataset-local/data_local/shaojiangyi/latence-project
/opt/conda/envs/pytorch2.4/bin/python scripts/nbs/audit_nbs_weak_dataset.py
```

默认使用 `outputs/latence_nbs_experiments/v092/geometry_single12800_scratch/resolved_config.json` 指向的数据。CPU 运行，只依赖 NumPy 和 Python 标准库，不构建 GNN，不读取独立测试标签，也不获取训练锁。候选边规模很大，完整审计需要顺序读取磁盘；可先添加 `--skip-candidates` 获得标签覆盖报告。不同选项请使用不同 `--output-dir`。

完整参数：

```bash
python scripts/nbs/audit_nbs_weak_dataset.py --help
```

脚本输出报告以及 GO/频次分组覆盖表。需要反馈的是完整审计的 `report.json`、`frequency_strata.csv`，以及 GO 级别的 `go_coverage.csv`。脚本的成功运行只代表数据结构审计完成，不代表伪标签已被证实正确。

## 审计要回答的问题

1. **词表与频次。** 保持全部任务 GO 列；用训练 core 金标准重建频次，排除原 holdout。互斥分组为 0、1–5、6–9、10–99、100–999、≥1000，避免将 1–5 和 1–9 重复相加。
2. **Weak 正监督覆盖。** 每组伪正例数、覆盖 GO 数、零伪标签 GO 数、weak 标签数分布。训练仅使用有伪标签的 active weak 行，因此同时报告 active weak 和全部 weak 两个分母。
3. **候选证据覆盖。** 按现有排序规则重现候选 top16，比较导出候选与保留候选的低频组成，以及伪正例与候选的交集。top16 仅限制部分证据输入，不限制全列预测和全列监督。
4. **监督契约。** 记录实际 manifest 的教师来源、严格比较符、阈值及配置。原始阈值在写入 FP16 前执行，CSR 中的 0.5 可能来自略大于 0.5 的值；不得重新取阈值删掉这些成员。
5. **负项机制。** 当前 v092 的二值全列 BCE 通常将非伪正项作为 0；报告的潜在负项计数是排除别名掩码之前的数量，不等于真实生物学阴性，也不等于最终实际负梯度数量。

coverage 不能当作 precision/recall：weak 没有可直接用于此目的的可靠金标准。某 GO 在 weak 中少见也可能反映真实分布差异，不能按每 GO 配额强造伪正例。精度和教师阈值仍需要额外的有标签校准数据。

## 重构 weak 标签集的条件与设计

先在同一校准样本、同一 GO 顺序上比较 B、原始 expert E、当前 decoder modelout M；不要把 E 与 M 混称为 expert prob。若 Stage1 已见过全部 core，事后 Stage2 的 1024 holdout 不是 Stage1 的独立校准集。训练内最佳阈值只作探索；正式阈值尽量使用教师未见样本或折外预测，不使用独立测试标签选参数。

若结果支持分频教师：频次分组仅由训练 core 金标准确定；低频以 B 为主要候选教师，高频使用校准表现较好的 E/M 或固定融合规则；分组共享阈值。保留全部原始分数、硬标签和来源，core 金标准原样保留。

最小对照：原教师＋0.5；原教师＋分组阈值；分频教师＋0.5；分频教师＋分组阈值。初轮保持相同训练结构、loss 和预算，以检验数据本身的作用。连续分数的软监督/蒸馏或不确定弱标签掩码应作为后续单独对照，因为当前二值 loss 不会自动使用新 prob 值。

新数据版本需要显式的新来源/分组阈值契约、重新构建倒排索引及相关缓存，并启用新 run；不要伪装成旧 modelout>0.5 数据后继续 resume。

## 下一轮的成功标准

成功需同时满足整体性能提升与低频能力保留。在观察新测试结果之前，固定 1–5、1–9、10–99 各组 micro/macro AP 相对 B 的允许退化范围，以及预设工作点或匹配 precision 下的 recall 要求；不以整体 Fmax 改善抵消 rare 的实质退化。B 的参考值：1–5 micro/macro AP 为 23.262795/39.382143；1–9 为 27.471440/38.842662；10–99 为 32.586335/39.276718。稀有组的测试阳性数量少，微小波动需要配对不确定性分析，不能直接解释为稳定提升。

当前所有结果仅支持已有任务 GO 的诊断，不构成新 GO zero-shot 能力验证。
