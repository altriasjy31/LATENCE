# v094：统一 Backbone 硬标签，Modelout 全列软监督

本版是新 Stage2 模型与新监督目标。安装在已完成 v093 与 GO alias 修复的项目中，新增 v094 文件；不覆盖 v093 源码、旧数据或 checkpoint。

## 实验动机与边界

v093 epoch20/25/30 的 Fmax 为58.0167/58.0068/57.9389，micro AP 为49.4896/49.2314/49.0018；base为57.7231/49.4508。结果不支持仅靠继续延长该运行取得稳定整体收益，但也不能据此认定标签是唯一原因。

原v093在1–999频率组已经使用B硬标签。本轮不仅将高频硬标签也统一到B，还改变了软监督覆盖范围、weak/core权重及最终评分路径，属于整体新实验，不能解释为单项标签消融。

| 项目 | v094定义 |
|---|---|
| Weak硬标签 | 所有频率均使用 B > 训练core内拟合的分组阈值 |
| 阈值目标 | 每个频率组 pooled micro-F1；排除1024蛋白holdout及未见GO；不使用独立测试标签 |
| 最佳阈值含义 | 仅为上述训练core、频率组及网格内最优；Stage1见过该core，不是独立泛化校准 |
| Weak软监督 | 原始modelout概率M，对全部seen GO计算Bernoulli KL，包括B硬负例 |
| Strong监督 | 原core gold保留，降低其相对损失权重 |
| Rare损失 | 移除，不再蒸馏保留B的rare输出 |
| 候选选择 | 普通global top16，无rare/medium配额；16是图采样预算，输出和损失仍覆盖全部任务GO |
| 模型输出 | 唯一共享protein–GO评分器；无B logits相加、无额外残差头、无双头辅助损失 |
| GO特征 | 沿用center/log-offset直接输入和本体编码路径 |
| 初始化 | Stage2随机初始化，复用已保存Stage1表征与图；不从v093 checkpoint续训 |

M是第一阶段实际导出的modelout，包含其expert/base组合及解码修正，不等同于原始expert概率。M仅进入训练损失，不进入模型forward，也不改变候选选择。预测启动仍校验原训练文件以核对契约，因此应保留M源文件。原Stage1特征与候选图本身仍含上游信息。

## 损失和权重

默认配置：

```json
"loss": {
  "core_weight": 0.25,
  "weak_weight": 1.0,
  "modelout_soft_weight": 1.0
}
```

当两种角色都存在时：

\[
L=0.2\,\overline{\mathrm{BCE}}_{core}
 +0.8\,\overline{\mathrm{BCE}}_{weak,Bhard}
 +1.0\,\overline{\mathrm{KL}(M\|q)}_{weak}.
\]

80/20仅描述主BCE的角色比例；加上独立weak均值的软项后，形式系数是weak1.8、core0.2，约90/10。这不是梯度或性能贡献比例。相对v093的主BCE，weak/core系数比提高16倍。

对于普通weak配对，忽略alias排除等差异，有效目标为(0.8*yB+M)/1.8。因此B和M可以冲突，M原始概率允许小于0.5乃至0；M还可在B未选中的功能上提供监督。该设定并不等于简单把所有伪正例分数推高。

硬BCE继续保留canonical alias的PU排除；独立dense M监督只屏蔽unseen，不套用硬标签PU掩码。所有weak蛋白都参与训练，包括B硬标签集合为空的蛋白。

## 安装与运行

解压后进入本包目录：

```bash
/opt/conda/envs/pytorch2.4/bin/python install_v094_weak_teacher.py \
  --project-root /home/dataset-local/data_local/shaojiangyi/latence-project --apply
```

回到项目根目录：

```bash
cd /home/dataset-local/data_local/shaojiangyi/latence-project
/opt/conda/envs/pytorch2.4/bin/python scripts/nbs/run_nbs_v094.py check
/opt/conda/envs/pytorch2.4/bin/python scripts/nbs/run_nbs_v094.py build
/opt/conda/envs/pytorch2.4/bin/python scripts/nbs/run_nbs_v094.py smoke --gpu 0
/opt/conda/envs/pytorch2.4/bin/python scripts/nbs/run_nbs_v094.py train --gpu 0 --stop-epoch 20
```

`build`默认先重新校准全部频率（包括1000+），再构建数据。无需重新运行Stage1，但原B/M导出矩阵需存在。校准和构建仅用CPU。

默认路径：

- 原始source：`outputs/latence_nbs_experiments/v092/geometry_single12800_scratch`
- 新数据：`outputs/latence_nbs_experiments/v094/backbone_hard_modelout_soft`
- 新训练：`outputs/latence_nbs_experiments/v094/weak_teacher_single12800`
- Smoke：新训练目录名追加`_smoke`，与正式训练分开。

默认3层、hidden320、单卡batch10240weak+2560core、accum2，固定200epoch调度，每5epoch保存；stop-epoch仅停止节点，不缩短调度。v094的早期评估不再继承base预测起点。

另一个GPU评估，eval任务之间串行以兼容旧输入缓存：

```bash
/opt/conda/envs/pytorch2.4/bin/python scripts/nbs/run_nbs_v094.py evaluate --gpu 1 --epoch 5
```

报告：`<work>/eval_step<实际step>/full/metrics_v094/summary.json`。训练日志保留总loss、主BCE及独立modelout KL。终端旧引擎打印的`BCE=`是总loss，真实主BCE看`main_binary`。普通resume严格冻结配置、数据与代码。

```bash
/opt/conda/envs/pytorch2.4/bin/python scripts/nbs/run_nbs_v094.py resume --gpu 0 --stop-epoch 30
```

需要改变权重时，在新数据/运行目录的build阶段显式设置，例如`--core-weight 0.25 --weak-weight 1 --modelout-soft-weight 1`。首次build会冻结配置；已有不同配置的目录不会被覆盖。权重选项不会在resume时悄悄生效。

## 构建前后检查

`unified_dataset_manifest.json`记录来源哈希、阈值及分频覆盖。`teacher_disagreement`和每条`frequency_strata`记录：

- B硬正例且M<0.5的数量；
- B硬负例且M>0.5的数量；
- B硬正例上的M均值。

M的0.5仅用于报告分歧，不决定新硬标签。新CSR的probability字段存M，membership由B决定；dense软监督直接读取原M矩阵，避免复制几十GB文件。

## 未见GO功能

已见和未见GO均走同一个共享评分器。原本体注册表中的合法GO可按列表请求：

```bash
/opt/conda/envs/pytorch2.4/bin/python scripts/nbs/run_nbs_v094.py predict-go \
  --gpu 1 --epoch 5 --go-ids-file requested_go_v094.example.txt
```

示例GO是已见BP，仅测试接口。真正新增本体节点仍需要提供锚定于旧坐标的几何/关系bundle，新增GO特征不会由本脚本自动生成。`--label-holdout-fraction`支持新实验的Stage2标签排除：同时排除hard、soft和标签消息中的未见监督；上游Stage1可能见过这些标签，因此不宣称完整pipeline零样本。当前46个自然zero GO在独立测试中无正例，不能评价zero-shot准确性。

## 验证范围

本地20项CPU测试通过，21项需要PyTorch的模型、runtime及phase-engine测试因环境无PyTorch跳过。已验证自动校准→构建→配置→复用的真实schema小样本CLI流程，原数据不变。

服务器的`check`将执行真实Torch测试，包括teacher-only前向、Bnegative/Mpositive监督、unseen零梯度、微批累计、无双头、动态GO和连续训练/断点续训的模型、optimizer、scheduler等值检查。实际大数据/GPU运行仍需以服务器`check`和`smoke`为准，性能改善尚未经新训练验证。
