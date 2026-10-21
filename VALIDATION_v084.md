# v084 验证记录

本更新在 CPU PyTorch 2.4.1 上检查；没有完整 BP 数据、CUDA/NCCL 或双卡 A100 的实测结果。

## 实际检查内容

- 多源池：原生关系方向、显式反向边、去重、固定/动态对照、相同缺边增强、完整缓存哈希。
- 监督隔离：全部当前监督种子及 holdout 的 gold/pseudo 从输入和固定 loss-support 中移除；修改 seed 标签不改变前向图。
- 图学习：模型在仅 GO→core₂→core₁→种子路径携带信号的小图上学到区别；两层五类关系获得梯度。
- 数值和输出：初始输出等于 B、完整 GO 列保留、别名不合并、FP32 残差、缺边、AMP、chunk 和激活重计算。
- 推理一致性：真实采样器和模型的独立预测不随 batch 分组变化；第三跳不会意外进入 anchor query。
- 续训：固定和动态两组 4 步与 2＋2 步的模型、优化器、RNG、蛋白队列、loss 精确一致；跨轮批内无重复蛋白。
- 端到端：真实小型 CC 2,903 GO 数据训练两步，导出 full/pp_off，运行真实 local_micro 评估子进程；E/M/B/G 来源分开，预测等于直接模型输出，缓存复用不重做模型预测。
- 配置和入口：两份正式配置通过旧 loader 的监督来源检查；shell 路径带空格、Stage1 默认路径与覆盖、旧版本拒绝、六组干预来源一致性。
- 安装：在独立 v083 项目副本安装，验证包哈希、窄范围 loader 补丁、用户其他修改保留、备份及重复安装。

## 测试命令

项目已有 PyTorch、NumPy、pytest 及原评估依赖时：

```bash
PYTHONPATH="$PWD:$PWD/nbs_models/nbs_protein_go:$PWD/nbs_models/nbs_protein_go/tests" OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python -m pytest -q   nbs_models/nbs_protein_go/tests/test_full_task_data_v084.py   nbs_models/nbs_protein_go/tests/test_full_task_model_v084.py   nbs_models/nbs_protein_go/tests/test_full_task_runner_v084.py   tests/test_full_task_eval_v084.py   tests/test_run_nbs_v084_shell.py   tests/test_full_task_export_v084.py
```

在从 ZIP 安装得到的独立 v083 项目副本上，完整测试结果：**62 passed，12.29 秒，退出码 0**。包完整性、Python 语法、shell 语法、安装备份及重复安装检查也通过。个别 PyTorch CPU autocast 弃用提示来自原依赖，不代表测试失败。

本测试证明实现与运行流程通过所列检查，不证明实际 BP 性能提升。真实资源消耗、采样质量与性能应以两组20步 smoke、600步 pilot 为准。原 Stage1 helper.py 未提供，因此主表指标保持原实现，未声称修复此前的 Fmax/AUPRC 口径疑问。
