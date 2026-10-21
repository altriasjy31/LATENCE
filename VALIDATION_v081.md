# v0.8.1 验证记录

验证环境：CPU PyTorch 2.4.1；没有CUDA设备或用户的完整训练数据。

## 自动验证

- 69项通过，1项跳过（原模型/训练/评估测试及本次新增参照测试的合计）。
- 跳过项：双进程Gloo训练；执行环境禁止创建Gloo TCP通信，未把它记为通过。
- shell通过`bash -n`，新增Python运行模块通过语法检查。

测试范围：

1. 保留完整GO任务列，包括canonical/alt-ID共用本体行的情况。
2. 图辅助损失从第一步进入weak/core路径；辅助图分数不读取直接base、expert或目标标签。
3. 候选边随机丢弃与标签无关；同时移除聚合与精确候选属性。
4. GO query影响邻居读出；SDPA/math结果一致；分块checkpoint前向、梯度及随机状态一致。
5. 空邻居、全正例、短PU行、混合精度的小残差均有限。
6. 结构支持对hard/background/medium/ranking/KL的负向梯度衰减不被归一化抵消；不增加伪正例。
7. 固定toy优化恢复多个正例的相对排序；辅助图排序也得到优化。
8. 训练→保存→恢复，与连续训练具有一致参数和loss；改变监督契约或base clip会拒绝恢复。
9. 新入口baseline组与原v080在同一fixture上产生相同参数更新；loss组ranking接通。
10. 真实2,903列CC维度的导出和既有评估器运行完成；E/M独立、行列重排、错误ID及缺失参照均按契约处理。
11. 预测复用验证代码来源；损坏的归纳邻居缓存重建；重复邻居不重复计算结构支持。
12. 执行项目实际V3 query decoder定义，导出结果与 `sigmoid(qout["logits"])` 相同，并确认不同于省略anchor/gate的裸delta重建。
13. 实际Stage1 exporter参数解析、checkpoint/args合并及严格加载；用小型backbone替身验证独立MSA索引覆盖、checkpoint参数优先级、缺失decoder拒绝及模型配置哈希检查。
14. external经float16的数值路径、独立输入/MSA/external三种不同蛋白顺序、GO列重排、字节ID与alt-ID列保留。
15. 模型只接收不含标注的MSA选择数据；参照缓存复用发生在Stage1模型加载之前，源文件或输出变化会使缓存失效。
16. B不一致时不发布参照、不改写旧B；刷新失败保留旧参照；错误输出目录不能覆盖其他工作文件。
17. shell系列评估先导出一次再运行三个NBS检查点，手动成对参照跳过导出，单份参照在推理前报错，训练不触发测试参照。
18. 四组报告绑定生成manifest、E/M及ID输出哈希、原输入manifest与B来源。

## 在项目中复现

如环境已安装pytest，在项目根目录运行：

```bash
PYTHONPATH="$PWD:$PWD/nbs_models/nbs_protein_go:$PWD/nbs_models/nbs_protein_go/tests${PYTHONPATH:+:$PYTHONPATH}" \
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python -m pytest -q \
  nbs_models/nbs_protein_go/tests/test_full_task_model_v081.py \
  nbs_models/nbs_protein_go/tests/test_full_task_loss_v081.py \
  nbs_models/nbs_protein_go/tests/test_full_task_evidence_v081.py \
  nbs_models/nbs_protein_go/tests/test_full_task_runner_v081.py \
  nbs_models/nbs_protein_go/tests/test_full_task_data_v080.py \
  tests/test_full_task_eval_v081.py \
  tests/test_stage1_references_v081.py \
  tests/test_v081_reference_shell.py
```

训练环境如允许Gloo通信，双进程测试会实际运行。本包未宣称GPU/DDP或真实Stage1 checkpoint完整推理已验证，也未宣称真实数据性能已提升。仅补齐评估参照时，直接运行README中的 `references` / `evaluate`，不必重训。首次训练新loss/graph方案时，先做20步GPU smoke。
