# v085 验证记录

## 已执行

2026-09-18，Python + PyTorch 2.4.1 CPU，独立指标、日程/runner、开发集、shell/evaluator及真实小图导出测试合计 **94 passed**。测试报告出现13条PyTorch CPU autocast弃用警告，无失败。另从ZIP解压并安装到独立v084副本后，再次通过94项测试（13.72秒）；安装重复执行、原有文件逐个哈希保持不变、ZIP完整性与shell语法均通过检查。

覆盖内容：

- 原helper函数最小复现；标准指标与独立暴力计算、sklearn AP/PR曲线对照；同分数跨chunk、0/1、空gold与阈值边界。
- warmup/constant/cosine固定horizon、恢复契约；4步连续与2+2恢复的模型/优化器/日程/RNG/蛋白队列一致。
- 难gold按窗口/各rank总量计算，空难例覆盖率null；角色覆盖和遍历记录。
- 开发集Stage1/Stage2/test ID隔离、来源及行列校验、标签不入forward、开发checkpoint guard。
- 真正的小图两步训练→CC全2903列归纳预测→真实评估子进程→E/M/B/G标准化指标。
- 已有预测缓存不再次forward；显式缓存重算不torch.load/checkpoint、不覆盖原结果；文件被改动时拒绝继续。
- Shell参数/路径含空格/预置对照、来源绑定及标准指标契约比较。

合成规模资源检查：1800×21312（38,361,600个位置），全部新增指标CPU约12.07秒、进程峰值约828.7MiB。随机标签/随机分数，仅用于资源可行性，不代表真实模型性能。

## 可复现测试命令

在已安装v084+v085的项目目录，使用现有具有torch、numpy、scikit-learn、pytest的环境：

```bash
PYTHONPATH="$PWD:$PWD/nbs_models/nbs_protein_go:$PWD/nbs_models/nbs_protein_go/tests${PYTHONPATH:+:$PYTHONPATH}" \
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python -m pytest -q \
  tests/test_full_task_metrics_v085.py \
  tests/test_full_task_eval_v085.py \
  tests/test_run_nbs_v085_shell.py \
  tests/test_full_task_export_v085.py \
  nbs_models/nbs_protein_go/tests/test_full_task_schedule_v085.py \
  nbs_models/nbs_protein_go/tests/test_full_task_runner_v085.py \
  nbs_models/nbs_protein_go/tests/test_full_task_development_v085.py
```

## 尚未执行

没有用户服务器数据或GPU，未执行全BP训练、A100显存测量、DDP端到端训练或真实预测新指标重算；未证明参数改动提高泛化。开发集工具不能证明用户提供的Stage1训练ID列表完整，也不能替代同源/时间隔离设计。
