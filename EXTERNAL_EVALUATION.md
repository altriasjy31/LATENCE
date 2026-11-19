# 外部模型统一评估

`eval_external_unseen_go_v095.py` 直接读取现有 `test_manifest.json`。既可用 `ind_test_exp`，也可用 population manifest 中的 `dark_weak_all_exp`、其他有注释的 cohort。零正例的 IPR cohort 仍输出覆盖和计数，指标留空。

两个导出器输出完整蛋白轴、可评分的 GO 轴和未截断的 float32 概率矩阵。缺失 GO 在 `coverage.tsv` 单列；不会填零或改成负例。任何缺失蛋白都会终止评估。蛋白顺序、GO 顺序可不同，程序按明确的 ID 对齐。

独立测试示例，在项目根目录执行，变量指向前面生成的实际目录。

```bash
python scripts/nbs/eval_external_unseen_go_v095.py \
  --test-manifest "$TEST/test_manifest.json" --cohort ind_test_exp \
  --predictions-dir "$MZSGO_IND_PRED" \
  --reference-predictions-dir "$PRED" \
  --output-dir "${MZSGO_IND_PRED}_eval"
```

完整 Dark-Weak 评估只需切换 manifest、cohort 和两套预测目录。

```bash
python scripts/nbs/eval_external_unseen_go_v095.py \
  --test-manifest "$POP/test_manifest.json" --cohort dark_weak_all_exp \
  --predictions-dir "$MZSGO_DW_PRED" \
  --reference-predictions-dir "$DW_PRED" \
  --output-dir "${MZSGO_DW_PRED}_eval"
```

如果完整 Dark-Weak 的 EXP、IPR、union 和历史对照在同一 `test_manifest.json` 中共享完整蛋白轴，同一份分数可以直接复用，分别更改 `--cohort` 与新的 `--output-dir` 即可。程序校验绑定的 test manifest 哈希、导出原始 cohort 和目标 cohort 的蛋白轴；证据标签不同不要求重新推理。不同蛋白轴或不同测试 manifest 不能按此方式混用。

DeepGO 结果使用相同命令，更换 `--predictions-dir`。`--reference-predictions-dir` 可省略，也可以使用另一外部模型目录。LATENCE 预测仍经过原 v095 的检查，不会给外部模型伪造 LATENCE 的配置、ontology 或 checkpoint 身份。

主要输出如下。

| 文件或字段 | 含义 |
|---|---|
| `coverage.tsv` | 每个请求 GO 的可评分覆盖和原始观测正例数 |
| `metrics.json` → `external_supported` | 外部模型在自身可评分候选上的结果 |
| `metrics.json` → `reference_supported` | 对照模型在自身可评分候选上的结果 |
| `metrics.json` → `external_common` / `reference_common` | 相同 GO 交集、相同完整蛋白池上的成对比较 |
| `paired_protein_to_GO.tsv` | 每个共同正例的蛋白→GO 名次、并列范围及差值 |
| `paired_GO_to_protein.tsv` | 每个共同正例的 GO→蛋白名次、并列范围及差值 |
| 各 scope 下的明细 TSV | 逐蛋白、逐 GO、逐正例指标 |

`external_rank_improvement = reference_midrank - external_midrank`，正数表示外部模型把该观测正例排得更靠前。不同候选轴的 supported 结果只作覆盖诊断；方法优劣优先比较 common 结果，并同时报告被覆盖限制排除的正例数、蛋白数。

蛋白→GO 包含 P@K、R@K、F1@K、MRR、Fmax、AP。为了与原 v095 可比，Fmax/AP 的计算域是“有至少一个可评分观测正例的 eligible 蛋白”，字段名为 `classification_positive_proteins`，不代表全体蛋白的特异性。全体 eligible 蛋白的 P@K 分母另列在 `precision_all_evaluable_rows_proxy@K`。每个 scope 同时报告完整队列、可核对身份的队列、支持的正例以及因 GO 覆盖移除的正例数量。

GO→蛋白默认按每个 GO 在完整蛋白队列中排名，包含 P@K、R@K、F1@K、观测 AP 和逐正例名次。使用 `--protein-pool evaluable` 可以选择身份可核对的子集；该选择会明确记录。没有观测正例的蛋白不会从 full 排名池中删除。

所有未注释配对均为未标注；precision、F1、AP、Fmax 是相对于已观测标签的代理指标。Top-K 和 MRR 沿用原程序的精确并列规则，AP 同分者一起进入阈值，不靠 ID 顺序拆分。Fmax 的最优阈值只用于描述本次测试，不作为独立验证所得的部署阈值。模型能给新 GO 打分本身不证明没有训练期暴露，应结合导出 provenance 与训练数据审计。

验证命令如下，仅需 NumPy。

```bash
python tests/check_external_unseen_eval.py
```

合成检查覆盖双轴乱序、精确并列、GO 缺列、零覆盖、零正例、完整蛋白轴校验、文件哈希、测试绑定，以及严格 v095 对照预测的读取。这里的合成结果不是任何真实模型的性能。
