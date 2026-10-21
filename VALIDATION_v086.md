# v086 验证记录

本记录区分代码正确性与任务性能。没有在本地训练完整 BP 图，也没有新增服务器性能结果。

## 已完成

- 同一次 CPU 回归：**197 passed，29 个 PyTorch CPU autocast 弃用提示，29.52 秒**。
- 其中原 v085 指标、评估、导出、调度、训练恢复、开发集和 shell 回归 94 项；新增模型 18 项，runner/eval/export/shell 77 项，loader 8 项。
- 原上传完整项目的 335 个文件与原 ZIP 逐字节比较，全部一致。所有实现变更都位于新 v086 文件。
- 新 loader 的 AST 对照显示：原 build 过程仅增加 v086 消费者契约检查，原监督来源校验只扩展版本白名单；没有修改底层数据或采样算法。
- Python 编译和 shell 语法检查通过。
- 在从原 ZIP 重建的独立项目副本上，安装器检查 58 个既有依赖、增加 19 个版本化文件；重复安装无写入，335 个原文件保持一致。同名 v086 内容冲突、旧依赖被改动均会拒绝安装。

## 覆盖的实际风险

- legacy 与 v084 公共参数初始化、非零 correction head 下输出/梯度一致；preln 公共初始化一致，初始 G=B。
- 两跳信息传播、anchor 不读第三跳、真实小型采样图上外部查询的 batch 划分不变。
- 空关系、低置信度、所有关系和两个 LN 的非零梯度；不读取监督目标或 E/M 连续概率作为新前向输入。
- DropEdge 层/步/rank 的确定性与独立全局 RNG；只改变 PP 传播，不改变 direct core vote。
- GO 分块 checkpoint 的输出/梯度/RNG；CPU BF16 的细小 residual 保留。
- 三个 encoder preset 的小图 fresh train/断点恢复；改变版本、encoder、日程、源码、数据等契约时拒绝恢复。
- 小图真实导出与标准指标、缓存重算、E/M/B/G 引用与预测来源约束、v086 manifest 身份；不伪装旧版本。
- 新 stage 与二值 pseudo 的来源审核；旧 loader 仍保持原 gate。
- shell 默认预算、显式评估、引用文件传递与路径处理。

## 在服务器运行的测试命令

使用已经安装项目依赖的 Python。在项目根目录执行；无需安装新的 DGL/PyG 版本来使用本版 encoder。

```bash
PYTHONPATH="$PWD:$PWD/nbs_models/nbs_protein_go:$PWD/nbs_models/nbs_protein_go/tests" \
  PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  python -m pytest -q \
  tests/test_full_task_metrics_v085.py \
  tests/test_full_task_eval_v085.py \
  tests/test_run_nbs_v085_shell.py \
  tests/test_full_task_export_v085.py \
  nbs_models/nbs_protein_go/tests/test_full_task_schedule_v085.py \
  nbs_models/nbs_protein_go/tests/test_full_task_runner_v085.py \
  nbs_models/nbs_protein_go/tests/test_full_task_development_v085.py \
  nbs_models/nbs_protein_go/tests/test_full_task_model_v086.py \
  nbs_models/nbs_protein_go/tests/test_full_task_runner_v086.py \
  nbs_models/nbs_protein_go/tests/test_local_loader_v086.py \
  tests/test_full_task_eval_v086.py \
  tests/test_full_task_export_v086.py \
  tests/test_run_nbs_v086_shell.py
```

本地使用的版本包括 PyTorch 2.4.1 CPU。测试复用了项目现有的小图 fixtures；不会说明服务器上的文件路径、CUDA 算子或资源一定可用。

## 尚未验证

- 服务器 CUDA、两卡 NCCL/DDP、真实 549,722 蛋白图和完整 21,312 BP 输出下的新模型训练。
- 服务器真实 E/M/B/G 矩阵的再次独立重算；本轮分析使用已上传的服务器标准指标和来源记录。
- 归一化或 DropEdge 的准确率、泛化、速度提升；新的独立开发集与多种子确认。
- BN、sum/关系权重、JK、GAT、额外 decoder dropout、新 loss、细分 weak 输入干预和 PU 快照执行器。它们没有隐含在本版实现中。

按 README 先进行单卡 smoke，再做固定预算的配对实验。没有可信 dev 时，core holdout 只能 monitor，不能把 ind_test 接入选择循环。
