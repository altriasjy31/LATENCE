# NBS P0 采样优化与分类器重构诊断：修订版

本包修订上传的 fast 采样器、入口、基准测试和对齐诊断。它不改变模型结构、损失、学习率、batch 或已有配置；混合评分头和注意力聚合尚未实现。

## 安装位置

推荐沿用现有项目的 `scripts/nbs`。也支持将本包新增脚本放在 `nbs/scripts`：安装时选择对应目录，shell 会从自身所在目录寻找 fast Python 入口。原始 `scripts/nbs/train_nbs_full_task_v089.py`、`...v090.py` 等依赖仍须保留原位，不能整体移动旧工程。

采样器固定安装到 `nbs_models/nbs_protein_go/nbs_pg/full_task_sampler_fast.py`。

解压后，在本包目录运行：

```bash
python install_p0_revision.py \
  --project-root /home/dataset-local/data_local/shaojiangyi/latence-project \
  --script-dir scripts/nbs

# 上一步预检通过后安装；若需要 nbs/scripts，两个命令都改该参数。
python install_p0_revision.py \
  --project-root /home/dataset-local/data_local/shaojiangyi/latence-project \
  --script-dir scripts/nbs --apply
```

安装器校验原始运行依赖及包内文件。已上传版本的 fast 文件可更新，更新前自动备份；不识别的本地修改会保留并报错，重复安装相同版本不会改写。它不会覆盖原始版本的 runner、model、data、config 或 checkpoint。

## 修复内容

- **采样器**：保留身份过滤、正 confidence、去重、补位、RNG 和边顺序；修复重复邻居带 NaN 时与原实现不同的错误行为；属性在原精度验证后再转 float32；按 2048 个接收节点分块，避免一次展开所有节点的候选池。
- **基准测试**：从配置读取真实本地 macro batch、world size 和 accumulation，排除全局所有监督种子；支持各 rank、多 step、train/eval 和外部查询推理比较。数组与 tensor 按 dtype、shape、字节比较，包括 signed zero。
- **计时报告**：预热、重复和中位数仅指 CPU batch 构建。未比较 full size 时，`full_size.compared=false`、`full_size.identical=null`；顶层 `identical` 只覆盖明确列出的比较案例。
- **启动器**：默认目录为 `v089_fast` / `v090_fast`，支持两种新增脚本位置；清理残留分布式环境；wrapper 重复安装/导入不会递归包装。
- **对齐诊断**：分类器 weight 与 bias 均纳入重构；canonical GO aliases 同组留出；岭回归超参数由训练 GO 内部选择；报告重构一致性及已见 GO 的诊断结果，不称为零样本评估。

## 1. 先做真实数据等价性与计时

从项目根目录运行。以下 `SCRIPTS` 必须与安装选择一致：

```bash
cd /home/dataset-local/data_local/shaojiangyi/latence-project
export PYTHON_BIN=/opt/conda/envs/pytorch2.4/bin/python
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
SCRIPTS=scripts/nbs
INPUT_DIR=outputs/latence_nbs_eval/bp_shared_full_task_top512_v071_v2

# v089：双卡形状的全局监督排除、两个 rank、完整本地规模。
# 原实现较慢；该命令不是三步 smoke，需等待多次构图完成。
"$PYTHON_BIN" "$SCRIPTS/bench_nbs_sampler_fast.py" \
  --config nbs_models/nbs_protein_go/configs/bp_full_task_v0.8.9_mb12800_ddp_ep100.json \
  --input-dir "$INPUT_DIR" --steps 1,2 --all-ranks --full-original --full-eval \
  --warmup 1 --repeats 3 \
  --output outputs/latence_nbs_diagnostics/sampler_fast_r2/v089.json

# v090：六层单卡 12800，规模自动来自配置，不再误用 6400 默认值。
"$PYTHON_BIN" "$SCRIPTS/bench_nbs_sampler_fast.py" \
  --config nbs_models/nbs_protein_go/configs/bp_full_task_v0.9.0_mb12800_single_ep100_6layer_lr001_weak.json \
  --input-dir "$INPUT_DIR" --steps 1,2 --full-original --full-eval \
  --warmup 1 --repeats 3 \
  --output outputs/latence_nbs_diagnostics/sampler_fast_r2/v090.json
```

预算紧张时可先去掉 `--full-original`、使用 `--steps 1 --warmup 0 --repeats 1` 做快速检查。此时只证明 subset 上已列出的案例相同，不能声称已验证完整规模。

结果应满足 `identical=true`；完整规模还应满足 `full_size.compared=true`、`full_size.identical=true`、`full_size_matches_config=true`。基准使用确定性的合成种子集合模拟 runner 形状，不是逐步重放某个真实训练历史。

`paired_timing.median_build_speedup` 是同规模 CPU 构图加速比，不是训练整步或六层模型的加速比。默认重复计时针对 subset；`--full-original` 只增加完整规模首个 step 的 train 比较，`--full-eval` 增加完整规模 eval 比较。若要对原实现完整规模预热并重复计时，再显式添加 `--time-full-original`，计算成本更高。完整训练速度需要另测前向、反向、传输及分布式开销。

## 2. 分类器—GO 表示重构诊断

先用原始采样器运行；完成相应真实数据等价性检查后，可将参数改为 `--sampler fast`。这一步读取服务器上的原始 GO 几何和 checkpoint，不修改它们。

```bash
export CUDA_VISIBLE_DEVICES=0  # 改为实际空闲物理卡
"$PYTHON_BIN" "$SCRIPTS/diagnose_nbs_alignment_v089.py" \
  --config outputs/latence_nbs_experiments/v089/mb12800_fresh_ep100/resolved_config.json \
  --checkpoint outputs/latence_nbs_experiments/v089/mb12800_fresh_ep100/nbs_epoch70.pt \
  --compare-prob outputs/latence_nbs_experiments/v089/mb12800_fresh_ep100/eval_step3360/full/nbs_ind_test_prob.f32.npy \
  --input-dir "$INPUT_DIR" \
  --metadata-file /home/dataset-assist-0/datafile/latence-dataset/unidata_with_exp_train_pseudo.pkl \
  --output-dir outputs/latence_nbs_diagnostics/alignment_v089_ep70_r2 \
  --device cuda:0 --sampler original
```

提供 `--compare-prob` 可检查重算路径与实际导出的一致性。超出脚本容差时会停止解释，而不是继续输出看似有效的对齐结论。未提供该文件时，导出一致性明确记为未检查。

重点比较 ontology encoding、原盒几何、frequency-only 和置乱对照。R² 不是空间语义成立与否的充分条件；替换分类器后的预测变化也应一起看。当前编码器已接受这些 GO 的监督，因此本报告属于已见 GO 的分类器重构诊断，不是严格零样本性能。

## 3. 可选：新目录短训练检查

仅在真实数据比较通过后运行：

```bash
export CUDA_VISIBLE_DEVICES=0
export CONFIG=nbs_models/nbs_protein_go/configs/bp_full_task_v0.9.0_mb12800_single_ep100_6layer_lr001_weak.json
export NUM_GPUS=1 DEVICE=cuda:0
export WORK_DIR=outputs/latence_nbs_experiments/v090_fast/mb12800_6layer_probe_r2
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
unset INIT_FROM ENCODER_FROM RESUME STEPS STOP_STEP STOP_EPOCH CHECKPOINT DEV_MANIFEST PRESET
unset ABLATION METRIC_BACKEND AUPRC_MODE
STOP_STEP=3 bash "$SCRIPTS/run_nbs_v090_single12800_fast.sh" train
```

三步只能检查能否运行，不能估计稳态吞吐。若需要训练整步计时，可在同一 fast 实验中以 `RESUME="$WORK_DIR/latest.pt" STOP_STEP=10` 接续，再观察后续窗口。

## 旧 checkpoint 的重要边界

旧 v089/v090 继续使用原始入口 resume/export；不要把原始 checkpoint 直接交给 fast `RESUME`，也不要覆盖旧运行源码来绕过检查。当前包没有实现保留 optimizer、scheduler、RNG 的跨实现迁移。v089 原始 checkpoint 的 `INIT_FROM` 也不能据此承诺兼容 fast 哈希检查。

fast 同版本 checkpoint 可在原 fast work directory 严格 resume/export。已有长训无需为了这次 P0 安装而中断。后续若要把未完成旧长训迁移到 fast，需要独立、可追溯的状态迁移工具；本包不把迁移伪装成普通 resume。

## 本地验证与服务器验证

包内测试只依赖 Python、NumPy 和标准库：

```bash
python -m unittest discover -s tests -v
```

覆盖三/六层、fixed/dynamic、train/eval、RNG/顺序、重复/非有限属性、CSR、分块、入口隔离、全局排除和回归诊断 helper。测试使用随包保留的原方法参考及受控 stub，不等同于真实 PyTorch/GPU 集成测试。

当前制作环境没有 PyTorch、训练数据或 GPU，因此尚未获得服务器上的加速倍数、完整推理等价结果或任何新性能提升。下一次只需返回 `v089.json` / `v090.json`、`alignment.json` / `summary.md`，以及可选短训 history，即可继续做有依据的实验决策。
